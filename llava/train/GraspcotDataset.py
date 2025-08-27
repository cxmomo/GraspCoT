import os
import copy
import torch
import pickle
import numpy as np
import re

from torch.utils.data import Dataset
from pytorchse3.se3 import se3_log_map

from llava.constants import (IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, 
                             DEFAULT_IM_END_TOKEN, DEFAULT_VIDEO_TOKEN, DEFAULT_IMAGE_PATCH_TOKEN, 
                             DEFAULT_VID_START_TOKEN, DEFAULT_VIDEO_PATCH_TOKEN, DEFAULT_VID_END_TOKEN,
                             DEFAULT_LOC_START_TOKEN, DEFAULT_LOC_END_TOKEN, DEFAULT_BOX_TOKEN,
                             UNSUPERVISED_TOKEN_INDEX, DEFAULT_UNSUPERVISED_TOKEN, TEPORARY_IGNORE_INDEX)

from llava import conversation as conversation_lib
from llava.model import *
from llava.mm_utils import tokenizer_image_token, map_obj, PlainBoxFormatter, tokenizer_special_token_v2

from typing import Dict, Optional, Sequence, List
import transformers
import tokenizers

import mmengine
import random
from tqdm import tqdm
from llava.train.image import Image

from packaging import version
IS_TOKENIZER_GREATER_THAN_0_14 = version.parse(tokenizers.__version__) >= version.parse('0.14')


MAX_WIDTH = 0.202   # maximum width of gripper 2F-140

img_w, img_h = 336, 336


response_model = ["I'll retrieve <obj> for you momentarily.", \
    "I'll grab <obj> as you need.", \
    "Alright, I'll bring you <obj> right now to meet your requirement.", \
    "No problem. I'll quickly obtain <obj> for you.", \
    "I'll fetch <obj>.", \
    "Okay, I'll go and pick up <obj> and return to you soon.", \
    "I'll be happy to get <obj> for you.", \
    "I'll seize <obj> without delay.", \
    "Okay. I'll secure <obj>.", \
    "Sure thing. I'll lay my hands on <obj> and bring to you pronto.", \
    "Maybe you need <obj>.", \
    "<obj>."]


request_model = [" If you are a grasping bot, determine what you should help me grasp in this scene."]
obj_num_model = ["The total number is ", \
    "The number is ", \
    ""]

number_model = ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", \
    "eleven", "twelve", "thirteen", "fourteen", "fifteen"]

num_response = len(response_model)
num_request = len(request_model)
num_obj_num = len(obj_num_model)

def get_rgb(rgb_file_path, rot=0, zoom=1.0, normalise=True):

    rgb_img = Image.from_file(rgb_file_path) # 0-255
    # import matplotlib.pyplot as plt
    # plt.imsave(f'image_{i}.png', image)
    # Jacquard try
    rgb_img.rotate(rot)
    rgb_img.zoom(zoom)
    rgb_img.resize((img_w, img_h))
    if normalise:
        rgb_img.normalise() # 255 到-1~1
        rgb_img.img = rgb_img.img.transpose((2, 0, 1))
    return rgb_img.img

def add_period(sentence):
    punctuation_marks = ['.', '?', '!']
    if sentence and sentence[-1] not in punctuation_marks:
        sentence += '.'
    return sentence

def _tokenize_fn(strings: Sequence[str],
                 tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ) for text in strings
    ]
    input_ids = labels = [
        tokenized.input_ids[0] for tokenized in tokenized_list
    ]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item()
        for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def _mask_targets(target, tokenized_lens, speakers):
    # cur_idx = 0
    cur_idx = tokenized_lens[0]
    tokenized_lens = tokenized_lens[1:]
    target[:cur_idx] = IGNORE_INDEX
    for tokenized_len, speaker in zip(tokenized_lens, speakers):
        if speaker == "human":
            target[cur_idx+2:cur_idx + tokenized_len] = IGNORE_INDEX
        cur_idx += tokenized_len


def _add_speaker_and_signal(header, source, get_conversation=True):
    """Add speaker and start/end signal on each round."""
    BEGIN_SIGNAL = "### "
    END_SIGNAL = "\n"
    conversation = header
    for sentence in source:
        from_str = sentence["from"]
        if from_str.lower() == "human":
            from_str = conversation_lib.default_conversation.roles[0]
        elif from_str.lower() == "gpt":
            from_str = conversation_lib.default_conversation.roles[1]
        else:
            from_str = 'unknown'
        sentence["value"] = (BEGIN_SIGNAL + from_str + ": " +
                             sentence["value"] + END_SIGNAL)
        if get_conversation:
            conversation += sentence["value"]
    conversation += BEGIN_SIGNAL
    return conversation


def preprocess_multimodal(
    sources: Sequence[str],
    is_multimodal: bool = False,
    mm_use_im_start_end: bool = False
) -> Dict:
    # is_multimodal = data_args.is_multimodal
    if not is_multimodal:
        return sources

    for source in sources:
        for sentence in source:
            if DEFAULT_IMAGE_TOKEN in sentence['value'] or DEFAULT_VIDEO_TOKEN in sentence['value']:
                sentence['value'] = sentence['value'].replace(DEFAULT_VIDEO_TOKEN, DEFAULT_IMAGE_TOKEN)
                sentence['value'] = sentence['value'].replace(DEFAULT_IMAGE_TOKEN, '').strip()
                sentence['value'] = DEFAULT_IMAGE_TOKEN + '\n' + sentence['value']
                sentence['value'] = sentence['value'].strip()
                if "mmtag" in conversation_lib.default_conversation.version:
                    sentence['value'] = sentence['value'].replace(DEFAULT_IMAGE_TOKEN, '<Image>' + DEFAULT_IMAGE_TOKEN + '</Image>')
            # Here we replace the <video> token with <image> token to reduce the coding 
            replace_token, video_replace_token = DEFAULT_IMAGE_TOKEN, DEFAULT_IMAGE_TOKEN
            if mm_use_im_start_end: # false
                replace_token = DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
            sentence["value"] = sentence["value"].replace(DEFAULT_IMAGE_TOKEN, replace_token)
    return sources
    

def preprocess_target_prompts(
    sources: Sequence[str],
    targets: Sequence,
    # data_args: DataArguments
    is_multimodal: bool = False
) -> Dict:
    # is_multimodal = data_args.is_multimodal
    if not is_multimodal:
        return sources

    for idx, source in enumerate(sources):
        target = targets[idx]
        if target is not None and 'boxes' in target:
            boxes = target['boxes']
            clicks = []
            for box in boxes:
                click = [round(coord, 3) for coord in box[:3]]
                clicks.append(click)
        else:
            clicks = []
        for sentence in source:
            words = sentence['value']
            boxes_seq = sentence.get('boxes_seq', None)
            if boxes_seq is not None:
                boxes = boxes_seq[0]
                objs_num = len(boxes)
                obj_placeholder =  DEFAULT_BOX_TOKEN + ', '
                objs_str = obj_placeholder * objs_num
                objs_str = objs_str.rstrip(', ')
                converted = words.replace(DEFAULT_BOX_TOKEN, objs_str)
                words = converted
            if boxes_seq is not None:
                sentence['value'] = words
    return sources, clicks

def preprocess_llama_2(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations

    if has_image:
        input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.LLAMA_2

    # Mask targets
    sep = "[/INST] "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_v1(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())   # list of combination conversation(including <image>/n)

    # Tokenize conversations
    # conversations[0] = "§ § § " + conversations[0]
    if has_image:
        input_ids = torch.stack([tokenizer_special_token_v2(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.TWO

    sep = conv.sep + conv.roles[1] + ": "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_special_token_v2(rou, tokenizer))
                instruction_len = len(tokenizer_special_token_v2(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            if i != 0 and not tokenizer.legacy and IS_TOKENIZER_GREATER_THAN_0_14:
                round_len -= 1
                instruction_len -= 1

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        # a = target[target==UNSUPERVISED_TOKEN_INDEX]
        target[target==UNSUPERVISED_TOKEN_INDEX] = TEPORARY_IGNORE_INDEX
        # target[target==UNSUPERVISED_TOKEN_INDEX] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_mpt(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations

    if has_image:
        input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()
    assert conv.sep_style == conversation_lib.SeparatorStyle.MPT

    # Mask targets
    sep = conv.sep + conv.roles[1]
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep)
        re_rounds = [conv.sep.join(rounds[:3])]
        for conv_idx in range(3, len(rounds), 2):
            re_rounds.append(conv.sep.join(rounds[conv_idx:conv_idx+2]))
        cur_len = 0
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(re_rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 1
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 1

            if i != 0 and getattr(tokenizer, 'legacy', False) and IS_TOKENIZER_GREATER_THAN_0_14:
                round_len += 1
                instruction_len += 1

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_plain(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    # add end signal and concatenate together
    conversations = []
    for source in sources:
        assert len(source) == 2
        assert DEFAULT_IMAGE_TOKEN in source[0]['value']
        source[0]['value'] = DEFAULT_IMAGE_TOKEN
        conversation = source[0]['value'] + source[1]['value'] + conversation_lib.default_conversation.sep
        conversations.append(conversation)
    # tokenize conversations
    input_ids = [tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations]
    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        tokenized_len = len(tokenizer_image_token(source[0]['value'], tokenizer))
        target[:tokenized_len] = IGNORE_INDEX

    return dict(input_ids=input_ids, labels=targets)


def preprocess(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False
) -> Dict:
    """
    Given a list of sources, each is a conversation list. This transform:
    1. Add signal '### ' at the beginning each sentence, with end signal '\n';
    2. Concatenate conversations together;
    3. Tokenize the concatenated conversation;
    4. Make a deepcopy as the target. Mask human words with IGNORE_INDEX.
    """
    if conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.PLAIN:
        return preprocess_plain(sources, tokenizer)
    if conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.LLAMA_2:
        return preprocess_llama_2(sources, tokenizer, has_image=has_image)
    if conversation_lib.default_conversation.version.startswith("v1"):
        return preprocess_v1(sources, tokenizer, has_image=has_image)
    if conversation_lib.default_conversation.version == "mpt":
        return preprocess_mpt(sources, tokenizer, has_image=has_image)
    # add end signal and concatenate together
    conversations = []
    for source in sources:
        header = f"{conversation_lib.default_conversation.system}\n\n"
        conversation = _add_speaker_and_signal(header, source)
        conversations.append(conversation)
    # tokenize conversations
    def get_tokenize_len(prompts):
        return [len(tokenizer_image_token(prompt, tokenizer)) for prompt in prompts]

    if has_image:
        input_ids = [tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations]
    else:
        conversations_tokenized = _tokenize_fn(conversations, tokenizer)
        input_ids = conversations_tokenized["input_ids"]

    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        if has_image:
            tokenized_lens = get_tokenize_len([header] + [s["value"] for s in source])
        else:
            tokenized_lens = _tokenize_fn([header] + [s["value"] for s in source], tokenizer)["input_ids_lens"]
        speakers = [sentence["from"] for sentence in source]
        _mask_targets(target, tokenized_lens, speakers)

    return dict(input_ids=input_ids, labels=targets)


def mask_text_llava_compatible(text, tokenizer):
    """
    通过分词结果直接替换每个Token为单位占位符，确保长度一致
    Args:
        text: 输入字符串
        tokenizer: LLaVA/Vicuna的分词器
    Returns:
        masked_text: 替换后的字符串
        is_ok: 是否长度一致
        original_tokens: 原始分词结果
        masked_tokens: 替换后分词结果
    """
    original_tokens = tokenizer.tokenize(text)

    masked_text = "_ " * len(original_tokens)

    masked_text = masked_text[:-1]

    masked_tokens = tokenizer.tokenize(masked_text)
    
    return masked_text

class GraspcotDataset_Train(Dataset):
    """
    Data loading class for training.
    """
    def __init__(self, dataset_path: str, tokenizer= transformers.PreTrainedTokenizer):
        """
        dataset_path (str): path to the dataset
        """
        super().__init__()
        self.dataset_path = dataset_path
        
        self.tokenizer = tokenizer
        self.ann_file = [f"data/grasp_anything/grasp_anything_infos_train_{str(i)}.pkl" for i in range(8)]
        self.dialogue_file = ["data/grasp_anything/dialogues/dialogue_infos_train_all.pkl"]
        # self._load()
        self.data_infos = self.load_annotations(self.ann_file)
        self.dialogue_infos = self.load_annotations(self.dialogue_file)
        self.aligned_infos = self.align_annotations()


    def load_annotations(self, ann_file_list):
        """Load annotations from ann_file.

        Args:
            ann_file (str): Path of the annotation file.

        Returns:
            list[dict]: List of annotations.
        """
        # loading data from a file-like object needs file format

        for i, ann_file in enumerate(ann_file_list):
            if i==0:
                data_infos = mmengine.load(ann_file, file_format='pkl')
            else:
                tmp_data = mmengine.load(ann_file, file_format='pkl')
                data_infos['infos'] += tmp_data['infos']
        return data_infos

    def align_annotations(self):
        aligned_infos = dict(version=self.dialogue_infos['version'])
        ann_infos = self.data_infos['infos']
        dialogue_infos = self.dialogue_infos['infos']


        aligned_list = []
        scene_tokens_ann = set()
        scene_tokens_dialogue = set()

        for ann_info in ann_infos:
            scene_token = ann_info.get('scene_token')
            scene_tokens_ann.add(scene_token)

        for dialogue_info in dialogue_infos:
            scene_token = dialogue_info.get('scene_token')
            if scene_token:
                scene_tokens_dialogue.add(scene_token)

        common_scene_tokens = scene_tokens_ann & scene_tokens_dialogue

        combined_dict_cache = {}
        for scene_token in common_scene_tokens:
            combined_dict_cache[scene_token] = {}

        for ann_info in ann_infos:
            scene_token = ann_info.get('scene_token')
            if scene_token in common_scene_tokens:
                combined_dict_cache[scene_token].update(ann_info)

        for dialogue_info in dialogue_infos:
            scene_token = dialogue_info.get('scene_token')
            if scene_token in common_scene_tokens:
                combined_dict_cache[scene_token].update(dialogue_info)

        for combined_dict in combined_dict_cache.values():
            aligned_list.append(combined_dict)

        aligned_infos['infos'] = aligned_list
        return aligned_infos


    def get_data_info(self, index):
        info = self.aligned_infos['infos'][index]
        
        scene = info['scene_token']
        try:
            with open(f"data/grasp_anything/scene_description/{scene}.pkl", "rb") as f:
                scene_description = pickle.load(f)
            scene_description_str = scene_description[0]
            obj_name_list = scene_description[1]
        except:
            obj_name_list = []

        pc_path = info['pc_path']
        img_path = info['img_path']
        depth_path = info['depth_path']

        ext = torch.from_numpy(info['pose']).to(torch.bfloat16)
        intrinsic = torch.from_numpy(info['intrinsic']).to(torch.bfloat16)

        gs_prompts_list = info['gs_prompts']
        grasps = info['gs']
        gs_labels = info['gs_labels']

        dialogs = info['dialogues']
        dialog_objs = info['objects']
        dialog_certaintys = info['certainty']

        pc = np.load(f"{self.dataset_path}/pc/{scene}.npy")
        pc = torch.from_numpy(pc).to(torch.bfloat16)

        rgb = []
        for i in range(4):
            rgb_i = get_rgb(f"{self.dataset_path}/" + img_path + f"_{str(i)}.png")
            rgb.append(torch.from_numpy(rgb_i).to(torch.bfloat16))
        rgb = torch.stack(rgb)

        depth = np.load(f"{self.dataset_path}/" + depth_path + ".npy")
        depth = torch.from_numpy(depth).to(torch.bfloat16) / 255

        n_dialog = len(dialogs)
        mean_certainty = [sum(sub_list) / len(sub_list) for sub_list in dialog_certaintys]

        valid_dialog_ids = []
        for i_c, certain in enumerate(mean_certainty):
            if certain!= 0:
                valid_dialog_ids.append(i_c)

        query_list, query_obj_list, response_list = [], [], []
        
        conversations = []
        mask_conversations = []

        request_str = "<image>" + "Describe the scene in detail."
        response_str = scene_description_str
        mask_response_str = mask_text_llava_compatible(scene_description_str, self.tokenizer)

        request = {
            "from": "human",
            "value": request_str
            }
        response = {
            "from": "gpt",
            "value": response_str
            }

        conversations += [request, response]
        mask_conversations += [request, {"from": "gpt", "value": mask_response_str}]

        query_list.append("Describe the scene in detail.")
        response_list.append(response_str)

        if len(valid_dialog_ids)>0:
            dialog_id = random.sample(valid_dialog_ids, 1)[0]
            dialog_list = dialogs[dialog_id]
            used_dia_objs_ori = dialog_objs[dialog_id]
            used_dia_objs = [dia_obj for dia_obj in used_dia_objs_ori]

            used_obj_names = [obj_name_list[obj_id] for obj_id in used_dia_objs]
            response_template = response_model[random.randint(0, num_response-1)]
            
            if len(used_dia_objs)>1:
                str_used_obj_names = ""
                for qq, obj_name in enumerate(used_obj_names):
                    if qq == 0:
                        str_used_obj_names = str_used_obj_names + "the " + obj_name
                    else:
                        str_used_obj_names = str_used_obj_names + ", the " + obj_name
            else:
                assert len(used_dia_objs) == 1
                str_used_obj_names = "the " + used_obj_names[0]

            request_str = add_period(dialog_list[0]).replace("Human: ", "") + request_model[random.randint(0, num_request-1)]
            response_str = response_template.replace("<obj>", str_used_obj_names)
            # mask_response_str = response_str.replace(str_used_obj_names, mask_text_llava_compatible(str_used_obj_names, self.tokenizer))
        else:
            n_obj = len(gs_prompts_list)
            obj_ids_list = list(range(n_obj))
            pos_obj_id = random.sample(obj_ids_list, 1)[0]
            used_dia_objs = [pos_obj_id]
            used_obj_names = [obj_name_list[obj_id] for obj_id in used_dia_objs]
            str_used_obj_names = used_obj_names[0]

            request_str = gs_prompts_list[pos_obj_id]
            response_str = "Okay, I'll bring you "+str_used_obj_names+" right now."
        
        assert len(used_dia_objs) >= 1
        mask_response_str = response_str.replace(str_used_obj_names, mask_text_llava_compatible(str_used_obj_names, self.tokenizer))

        request = {
            "from": "human",
            "value": request_str
            }
        response = {
            "from": "gpt",
            "value": response_str
            }

        conversation = [request, response]

        conversations += conversation
        mask_conversations += [request, {"from": "gpt", "value": mask_response_str}]

        query_list.append(request_str)
        response_list.append(response_str)

        query_obj_list = used_obj_names

        request_str = "How many objects need to be grasped in this scene?"
        # if len(used_dia_objs)>=10:
        #     print(used_obj_names)
        #     print(dialog_list[0])
        obj_num = number_model[len(used_dia_objs)-1]
        response_str = obj_num_model[random.randint(0, num_obj_num-1)]+ obj_num +"."
        mask_response_str = response_str.replace(obj_num, mask_text_llava_compatible(obj_num, self.tokenizer))

        request = {
            "from": "human",
            "value": request_str
            }
        response = {
            "from": "gpt",
            "value": response_str
            }

        conversations += [request, response]
        mask_conversations += [request, {"from": "gpt", "value": mask_response_str}]
        
        query_list.append(request_str)
        response_list.append(response_str)

        request_str = "For each target object in the scene, describe its material property by selecting adjectives from: soft, hard, rigid, flexible, brittle, ductile, tough, elastic, viscoelastic, etc. " \
            + "For each object, provide the output as: \"[object_name]: [adjective1], [adjective2];\". Separate multiple objects with semicolons (;). If no adjectives apply, write: \"[object_name]: none;\"."

        request = {
            "from": "human",
            "value": request_str
            }

        response_str = "<unsupervised>" * 30
        mask_response_str = "_ " * 30
        mask_response_str = mask_response_str[:-1]

        response = {
            "from": "gpt",
            "value": response_str
            }

        conversations += [request, response]
        mask_conversations += [request, {"from": "gpt", "value": mask_response_str}]
        
        query_list.append(request_str)
        response_list.append(response_str)

        request_str = "For each target object in the scene, describe its surface property by selecting adjectives from: granular, smooth, grooved, porous, fibrous, coarse, gritty, polished, pitted, uneven, serrated, sticky, slippery, tacky, lubricated, hydrophobic, etc. " \
            + "For each object, provide the output as: \"[object_name]: [adjective1], [adjective2];\". Separate multiple objects with semicolons (;). If no adjectives apply, write: \"[object_name]: none;\"."
        response_str = "<unsupervised>" * 30
        mask_response_str = "_ " * 30
        mask_response_str = mask_response_str[:-1]

        request = {
            "from": "human",
            "value": request_str
            }
        response = {
            "from": "gpt",
            "value": response_str
            }

        conversations += [request, response]
        mask_conversations += [request, {"from": "gpt", "value": mask_response_str}]

        query_list.append(request_str)
        response_list.append(response_str)

        request_str = "For each target object in the scene, describe its approximate shape by selecting adjectives from: cylindrical, planar, asymmetric, spherical, protrusion, recess, flat, perforated, layered, smooth, textured, jagged, organic, etc. " \
            + "For each object, provide the output as: \"[object_name]: [adjective1], [adjective2];\". Separate multiple objects with semicolons (;). If no adjectives apply, write: \"[object_name]: none;\"."
        response_str = "<unsupervised>" * 30
        mask_response_str = "_ " * 30
        mask_response_str = mask_response_str[:-1]

        request = {
            "from": "human",
            "value": request_str
            }
        response = {
            "from": "gpt",
            "value": response_str
            }

        conversations += [request, response]
        mask_conversations += [request, {"from": "gpt", "value": mask_response_str}]

        query_list.append(request_str)
        response_list.append(response_str)

        request_str = "To grasp targert objects in this scene, select an appropriate verb for each object from: clamp, pinch, snap, pluck, squeeze, tweeze, retrieve, lift, grip, scoop, hook, etc. " \
            + "For each object, provide the output as: \"[object_name]: [verb];\". Separate multiple objects with semicolons (;). If no verbs apply, write: \"[object_name]: none;\"."
        response_str = "<unsupervised>" * 30
        mask_response_str = "_ " * 30
        mask_response_str = mask_response_str[:-1]

        request = {
            "from": "human",
            "value": request_str
            }
        response = {
            "from": "gpt",
            "value": response_str
            }

        conversations += [request, response]
        mask_conversations += [request, {"from": "gpt", "value": mask_response_str}]

        query_list.append(request_str)
        response_list.append(response_str)

        pos_grasps = []
        pos_gs_labels = []
        pos_grasp_ids = []
        for k, obj_id in enumerate(used_dia_objs):
            pos_grasps.append(grasps[obj_id])
            pos_gs_labels.append(gs_labels[obj_id]-1)
            pos_grasp_ids.append(scene+"_"+str(obj_id))

        pos_gs = torch.cat(pos_grasps, dim=0)
        pos_gs_label = torch.cat(pos_gs_labels, dim=0)


        sources = preprocess_multimodal(
            copy.deepcopy([conversations]),
            is_multimodal=True)

        mask_sources = preprocess_multimodal(
            copy.deepcopy([mask_conversations]),
            is_multimodal=True)

        data_dict = preprocess(sources, tokenizer=self.tokenizer, has_image=True)
        mask_data_dict = preprocess(mask_sources, tokenizer=self.tokenizer, has_image=True)

        input_ids = mask_data_dict["input_ids"][0]
        labels = data_dict["labels"][0]

        input_dict = dict(
            grasp_ids=pos_grasp_ids,
            input_ids=input_ids, 
            labels=labels,
            scene = scene, 
            pc = pc,
            pc_path = pc_path,
            image = rgb,
            depth = depth, 
            grasps = pos_gs, 
            gs_labels = pos_gs_label.squeeze(1), 
            pose = ext, 
            intrinsic = intrinsic
        )

        return input_dict

            
    def __getitem__(self, index):
        """
        index (int): the element index
        """
        data_dict = self.get_data_info(index)
        return data_dict

    def __len__(self):
        return len(self.aligned_infos['infos'])
    

class GraspcotDataset_Test(Dataset):
    """
    Data loading class for training.
    """
    def __init__(self, dataset_path: str, tokenizer= transformers.PreTrainedTokenizer):
        """
        dataset_path (str): path to the dataset
        """
        super().__init__()
        self.dataset_path = dataset_path
        
        self.tokenizer = tokenizer
        
        self.ann_file = [f"data/grasp_anything/grasp_anything_infos_val_{str(i)}.pkl" for i in range(2)]
        self.dialogue_file = ["data/grasp_anything/dialogues/dialogue_infos_val_all.pkl"]
        
        self.data_infos = self.load_annotations(self.ann_file)
        self.dialogue_infos = self.load_annotations(self.dialogue_file)
        self.aligned_infos = self.align_annotations()
    

    def load_annotations(self, ann_file_list):
        """Load annotations from ann_file.

        Args:
            ann_file (str): Path of the annotation file.

        Returns:
            list[dict]: List of annotations.
        """
        # loading data from a file-like object needs file format

        for i, ann_file in enumerate(ann_file_list):
            if i==0:
                data_infos = mmengine.load(ann_file, file_format='pkl')
            else:
                tmp_data = mmengine.load(ann_file, file_format='pkl')
                data_infos['infos'] += tmp_data['infos']
        return data_infos

    def align_annotations(self):
        aligned_infos = dict(version=self.dialogue_infos['version'])
        ann_infos = self.data_infos['infos']
        dialogue_infos = self.dialogue_infos['infos']

        # 用于存储筛选后的元素
        aligned_list = []
        scene_tokens_ann = set()
        scene_tokens_dialogue = set()

        # 先遍历ann_infos提取所有的scene_token，存入scene_tokens_ann集合
        for ann_info in ann_infos:
            scene_token = ann_info.get('scene_token')
            if scene_token:
                scene_tokens_ann.add(scene_token)

        # 再遍历dialogue_infos提取所有的scene_token，存入scene_tokens_dialogue集合
        for dialogue_info in dialogue_infos:
            scene_token = dialogue_info.get('scene_token')
            if scene_token:
                scene_tokens_dialogue.add(scene_token)

        # 找出两个集合中共同的scene_token
        common_scene_tokens = scene_tokens_ann & scene_tokens_dialogue

        # 创建一个字典，用于临时存储同一个scene_token对应的合并后的字典数据
        combined_dict_cache = {}
        for scene_token in common_scene_tokens:
            combined_dict_cache[scene_token] = {}

        # 遍历ann_infos，将对应scene_token的键值对合并到combined_dict_cache中
        for ann_info in ann_infos:
            scene_token = ann_info.get('scene_token')
            if scene_token in common_scene_tokens:
                combined_dict_cache[scene_token].update(ann_info)

        # 遍历dialogue_infos，将对应scene_token的键值对合并到combined_dict_cache中
        for dialogue_info in dialogue_infos:
            scene_token = dialogue_info.get('scene_token')
            if scene_token in common_scene_tokens:
                combined_dict_cache[scene_token].update(dialogue_info)

        # 将合并后的字典数据添加到aligned_list
        for combined_dict in combined_dict_cache.values():
            aligned_list.append(combined_dict)

        aligned_infos['infos'] = aligned_list
        return aligned_infos


    def get_data_info(self, index):
        info = self.aligned_infos['infos'][index]
        # dialogue_info = self.dialogue_infos['infos'][index]
        
        scene = info['scene_token']
        try:
            with open(f"data/grasp_anything/scene_description/{scene}.pkl", "rb") as f:
                scene_description = pickle.load(f)
            scene_description_str = scene_description[0]
            obj_name_list = scene_description[1]
        except:
            obj_name_list = []

        pc_path = info['pc_path']
        img_path = info['img_path']
        depth_path = info['depth_path']

        ext = torch.from_numpy(info['pose']).to(torch.bfloat16)
        intrinsic = torch.from_numpy(info['intrinsic']).to(torch.bfloat16)

        gs_prompts_list = info['gs_prompts']
        grasps = info['gs']
        gs_labels = info['gs_labels']

        dialogs = info['dialogues']
        dialog_objs = info['objects']
        dialog_certaintys = info['certainty']

        pc = np.load(f"{self.dataset_path}/pc/{scene}.npy")
        pc = torch.from_numpy(pc).to(torch.bfloat16)

        rgb = []
        for i in range(4):
            rgb_i = get_rgb(f"{self.dataset_path}/" + img_path + f"_{str(i)}.png")
            rgb.append(torch.from_numpy(rgb_i).to(torch.bfloat16))
        rgb = torch.stack(rgb)

        depth = np.load(f"{self.dataset_path}/" + depth_path + ".npy")
        depth = torch.from_numpy(depth).to(torch.bfloat16) / 255

        n_dialog = len(dialogs)
        mean_certainty = [sum(sub_list) / len(sub_list) for sub_list in dialog_certaintys]

        valid_dialog_ids = []
        for i_c, certain in enumerate(mean_certainty):
            if certain!= 0:
                valid_dialog_ids.append(i_c)

        query_list, query_obj_list, response_list = [], [], []
        
        conversations = []
        mask_conversations = []

        request_str = "<image>" + "Describe the scene in detail."
        response_str = scene_description_str
        mask_response_str = mask_text_llava_compatible(scene_description_str, self.tokenizer)

        request = {
            "from": "human",
            "value": request_str
            }
        response = {
            "from": "gpt",
            "value": response_str
            }

        conversations += [request, response]
        mask_conversations += [request, {"from": "gpt", "value": mask_response_str}]

        query_list.append("Describe the scene in detail.")
        response_list.append(response_str)

        if len(valid_dialog_ids)>0:
            dialog_id = random.sample(valid_dialog_ids, 1)[0]
            dialog_list = dialogs[dialog_id]
            used_dia_objs_ori = dialog_objs[dialog_id]
            used_dia_objs = [dia_obj for dia_obj in used_dia_objs_ori]

            used_obj_names = [obj_name_list[obj_id] for obj_id in used_dia_objs]
            response_template = response_model[random.randint(0, num_response-1)]
            
            if len(used_dia_objs)>1:
                str_used_obj_names = ""
                for qq, obj_name in enumerate(used_obj_names):
                    if qq == 0:
                        str_used_obj_names = str_used_obj_names + "the " + obj_name
                    else:
                        str_used_obj_names = str_used_obj_names + ", the " + obj_name
            else:
                assert len(used_dia_objs) == 1
                str_used_obj_names = "the " + used_obj_names[0]

            request_str = add_period(dialog_list[0]).replace("Human: ", "") + request_model[random.randint(0, num_request-1)]
            response_str = response_template.replace("<obj>", str_used_obj_names)
            # mask_response_str = response_str.replace(str_used_obj_names, mask_text_llava_compatible(str_used_obj_names, self.tokenizer))
        else:
            n_obj = len(gs_prompts_list)
            obj_ids_list = list(range(n_obj))
            pos_obj_id = random.sample(obj_ids_list, 1)[0]
            used_dia_objs = [pos_obj_id]
            used_obj_names = [obj_name_list[obj_id] for obj_id in used_dia_objs]
            str_used_obj_names = used_obj_names[0]

            request_str = gs_prompts_list[pos_obj_id]
            response_str = "Okay, I'll bring you "+str_used_obj_names+" right now."
        
        assert len(used_dia_objs) >= 1
        mask_response_str = response_str.replace(str_used_obj_names, mask_text_llava_compatible(str_used_obj_names, self.tokenizer))

        request = {
            "from": "human",
            "value": request_str
            }
        response = {
            "from": "gpt",
            "value": response_str
            }

        conversation = [request, response]

        conversations += conversation
        mask_conversations += [request, {"from": "gpt", "value": mask_response_str}]

        query_list.append(request_str)
        response_list.append(response_str)

        query_obj_list = used_obj_names

        request_str = "How many objects need to be grasped in this scene?"
        if len(used_dia_objs)>=10:
            print(used_obj_names)
            print(dialog_list[0])
        obj_num = number_model[len(used_dia_objs)-1]
        response_str = obj_num_model[random.randint(0, num_obj_num-1)]+ obj_num +"."
        mask_response_str = response_str.replace(obj_num, mask_text_llava_compatible(obj_num, self.tokenizer))

        request = {
            "from": "human",
            "value": request_str
            }
        response = {
            "from": "gpt",
            "value": response_str
            }

        conversations += [request, response]
        mask_conversations += [request, {"from": "gpt", "value": mask_response_str}]
        
        query_list.append(request_str)
        response_list.append(response_str)

        request_str = "For each target object in the scene, describe its material property by selecting adjectives from: soft, hard, rigid, flexible, brittle, ductile, tough, elastic, viscoelastic, etc. " \
            + "For each object, provide the output as: \"[object_name]: [adjective1], [adjective2];\". Separate multiple objects with semicolons (;). If no adjectives apply, write: \"[object_name]: none;\"."

        request = {
            "from": "human",
            "value": request_str
            }

        response_str = "<unsupervised>" * 30
        mask_response_str = "_ " * 30
        mask_response_str = mask_response_str[:-1]

        response = {
            "from": "gpt",
            "value": response_str
            }

        conversations += [request, response]
        mask_conversations += [request, {"from": "gpt", "value": mask_response_str}]
        
        query_list.append(request_str)
        response_list.append(response_str)

        request_str = "For each target object in the scene, describe its surface property by selecting adjectives from: granular, smooth, grooved, porous, fibrous, coarse, gritty, polished, pitted, uneven, serrated, sticky, slippery, tacky, lubricated, hydrophobic, etc. " \
            + "For each object, provide the output as: \"[object_name]: [adjective1], [adjective2];\". Separate multiple objects with semicolons (;). If no adjectives apply, write: \"[object_name]: none;\"."
        response_str = "<unsupervised>" * 30
        mask_response_str = "_ " * 30
        mask_response_str = mask_response_str[:-1]

        request = {
            "from": "human",
            "value": request_str
            }
        response = {
            "from": "gpt",
            "value": response_str
            }

        conversations += [request, response]
        mask_conversations += [request, {"from": "gpt", "value": mask_response_str}]

        query_list.append(request_str)
        response_list.append(response_str)

        request_str = "For each target object in the scene, describe its approximate shape by selecting adjectives from: cylindrical, planar, asymmetric, spherical, protrusion, recess, flat, perforated, layered, smooth, textured, jagged, organic, etc. " \
            + "For each object, provide the output as: \"[object_name]: [adjective1], [adjective2];\". Separate multiple objects with semicolons (;). If no adjectives apply, write: \"[object_name]: none;\"."
        response_str = "<unsupervised>" * 30
        mask_response_str = "_ " * 30
        mask_response_str = mask_response_str[:-1]

        request = {
            "from": "human",
            "value": request_str
            }
        response = {
            "from": "gpt",
            "value": response_str
            }

        conversations += [request, response]
        mask_conversations += [request, {"from": "gpt", "value": mask_response_str}]

        query_list.append(request_str)
        response_list.append(response_str)

        request_str = "To grasp targert objects in this scene, select an appropriate verb for each object from: clamp, pinch, snap, pluck, squeeze, tweeze, retrieve, lift, grip, scoop, hook, etc. " \
            + "For each object, provide the output as: \"[object_name]: [verb];\". Separate multiple objects with semicolons (;). If no verbs apply, write: \"[object_name]: none;\"."
        response_str = "<unsupervised>" * 30
        mask_response_str = "_ " * 30
        mask_response_str = mask_response_str[:-1]

        request = {
            "from": "human",
            "value": request_str
            }
        response = {
            "from": "gpt",
            "value": response_str
            }

        conversations += [request, response]
        mask_conversations += [request, {"from": "gpt", "value": mask_response_str}]

        query_list.append(request_str)
        response_list.append(response_str)

        pos_grasps = []
        pos_gs_labels = []
        pos_grasp_ids = []
        for k, obj_id in enumerate(used_dia_objs):
            pos_grasps.append(grasps[obj_id])
            pos_gs_labels.append(gs_labels[obj_id]-1)
            pos_grasp_ids.append(scene+"_"+str(obj_id))

        pos_gs = torch.cat(pos_grasps, dim=0)
        pos_gs_label = torch.cat(pos_gs_labels, dim=0)


        sources = preprocess_multimodal(
            copy.deepcopy([conversations]),
            is_multimodal=True)

        mask_sources = preprocess_multimodal(
            copy.deepcopy([mask_conversations]),
            is_multimodal=True)

        data_dict = preprocess(sources, tokenizer=self.tokenizer, has_image=True)
        mask_data_dict = preprocess(mask_sources, tokenizer=self.tokenizer, has_image=True)

        input_ids = mask_data_dict["input_ids"][0]
        labels = data_dict["labels"][0]
        
        input_dict = dict(
            query=query_list,
            query_obj=query_obj_list,
            response=response_list,
            grasp_ids=pos_grasp_ids,
            input_ids=input_ids, 
            labels=labels,
            scene=scene, 
            pc=pc,
            pc_path=pc_path,
            image=rgb,
            depth=depth, 
            grasps=pos_gs, 
            gs_labels = pos_gs_label.squeeze(1), 
            pose = ext, 
            intrinsic = intrinsic
        )

        return input_dict
            
    def __getitem__(self, index):
        """
        index (int): the element index
        """
        data_dict = self.get_data_info(index)
        return data_dict

    def __len__(self):
        return len(self.aligned_infos['infos'])


tokenizer = transformers.AutoTokenizer.from_pretrained(
    # model_args.model_name_or_path,
    "./pretrained_llms/llava-3d-7b",
    model_max_length=4096,
    padding_side="right",
    use_fast=False,
)

if __name__ == "__main__":
    train_dataset = GraspcotDataset_Train("data/grasp_anything")