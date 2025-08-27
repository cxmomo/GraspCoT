import argparse
import torch

from llava.model.builder import load_pretrained_grasp_model
from llava.utils import disable_torch_init
from llava.mm_utils import (
    get_model_name_from_path,
)

from llava.train.GraspcotDataset import GraspcotDataset_Test
from llava.train.train import DataCollatorForSupervisedDataset

from PIL import Image

import requests
from PIL import Image
from io import BytesIO
import transformers
from typing import Dict

import os
import pickle
from tqdm import tqdm

import random
from llava import conversation as conversation_lib


def precess_data(tokenizer, instance):

    input_ids, labels = tuple(instance[key] for key in ("input_ids", "labels"))
    input_ids = input_ids.unsqueeze(0).cuda()
    labels = labels.unsqueeze(0).cuda() 
    input_ids = input_ids[:, :tokenizer.model_max_length]
    labels = labels[:, :tokenizer.model_max_length]
    batch = dict(
        input_ids=input_ids,
        labels=labels,
        attention_mask=input_ids.ne(tokenizer.pad_token_id),
    )
    
    # ==========Too many videos or images may lead to OOM, so we encode them one by one======================
    if 'image' in instance:
        images = instance["image"].unsqueeze(0).to(torch.bfloat16).cuda()
        batch['images'] = images  # (B, V, H, W)

    if 'depth' in instance:
        depths = instance["depth"].unsqueeze(0).to(torch.bfloat16).cuda()
        poses = instance["pose"].unsqueeze(0).to(torch.bfloat16).cuda()

        intrinsics = instance["intrinsic"].unsqueeze(0).to(torch.bfloat16).cuda()
        
        batch['depths'] = depths  # (B, V, H, W)
        batch['poses'] = poses  # (B, V, 4, 4)
        batch['intrinsics'] = intrinsics  # (B, 4, 4)

        # batch['clicks'] = clicks  # (num_clicks, 3)
    if 'pure_img' in instance:
        pure_imgs = instance["pure_img"].unsqueeze(0).to(torch.bfloat16).cuda()
        batch['pure_imgs'] = pure_imgs # (B, 3, 168, 168)
    return batch


def image_parser(args):
    out = args.image_file.split(args.sep)
    return out


def load_image(image_file):
    if image_file.startswith("http") or image_file.startswith("https"):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")
    return image


def load_images(image_files):
    out = []
    for image_file in image_files:
        image = load_image(image_file)
        out.append(image)
    return out

def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer,
                                data_path) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    eval_dataset = GraspcotDataset_Test(data_path, tokenizer=tokenizer)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=None,
                eval_dataset=eval_dataset,
                data_collator=data_collator)

def eval_model(args):
    # Model
    disable_torch_init()

    torch_dtype = torch.float32
    if args.precision == "bf16":
        torch_dtype = torch.bfloat16
    elif args.precision == "fp16":
        torch_dtype = torch.half

    model_name = get_model_name_from_path(args.model_path)
    print("********************* load model Start *********************")
    tokenizer, model, processor, context_len = load_pretrained_grasp_model(
        args.model_path, args.model_base, model_name, use_flash_attn=True, torch_dtype=torch_dtype
    )
    print("********************* load model End *********************")

    model.eval()

    if "llama-2" in model_name.lower():
        conv_mode = "llava_llama_2"
    elif "mistral" in model_name.lower():
        conv_mode = "mistral_instruct"
    elif "v1.6-34b" in model_name.lower():
        conv_mode = "chatml_direct"
    elif "v1" in model_name.lower():
        conv_mode = "llava_v1"
    elif "3d" in model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in model_name.lower():
        conv_mode = "mpt"
    else:
        conv_mode = "llava_v0"

    tokenizer.pad_token = tokenizer.unk_token
    if conv_mode in conversation_lib.conv_templates:
        conversation_lib.default_conversation = conversation_lib.conv_templates[conv_mode]
    else:
        conversation_lib.default_conversation = conversation_lib.conv_templates["vicuna_v1"]

    test_dataset = GraspcotDataset_Test(args.data_path, tokenizer=tokenizer)
    data_lens = len(test_dataset)

    model.config.use_dialogue = True

    num_val = 20000
    len_data = len(test_dataset)
    data_list = list(range(len_data))
    selected_data = random.sample(data_list, num_val)

    grasp_dict = {}
    for i in tqdm(selected_data):

        instance = test_dataset.__getitem__(i)
        gt_grasps = instance["grasps"]
        pc_path = instance["pc_path"]
        scene = instance["scene"]
        inputs = precess_data(tokenizer, instance)

        with torch.inference_mode():
            grasp_outs, outputs = model(**inputs)
            grasps = model.predict_grasps(grasp_outs)
            pred_grasps, scores, labels = grasps[0]

        grasp_dict[scene]=dict(gen_grasps=pred_grasps, gt_grasps=gt_grasps, scores=scores, pc_path=pc_path, scene=scene)

    with open(os.path.join(args.output_dir, "scenegrasp_gen_data.pkl"), 'wb') as f:
        pickle.dump(grasp_dict, f)
    
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--data-path", type=str, required=True) 
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--sep", type=str, default=",")
    parser.add_argument(
        "--precision",
        default="bf16",
        type=str,
        choices=["fp32", "bf16", "fp16"],
        help="precision for inference",
    )
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    args = parser.parse_args()

    eval_model(args)
