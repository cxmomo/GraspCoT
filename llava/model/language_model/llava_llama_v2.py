#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss

from transformers import AutoConfig, AutoModelForCausalLM, \
                         LlamaConfig, LlamaModel, LlamaForCausalLM

from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput

from ..llava_arch import LlavaMetaModel, LlavaMetaForCausalLM

from llava.constants import IGNORE_INDEX, GRASP_TOKEN_INDEX, TEPORARY_IGNORE_INDEX

import torch.nn.functional as F

from .modeling_llama_v2 import LlamaForCausalLM_v2
from ..petr_head_v2 import PETRHead_v2

point_cloud_range = [-1.0, -1.0, 0, 1.0, 1.0, 1]
voxel_size = [0.01, 0.01, 0.1]


class LlavaConfig(LlamaConfig):
    model_type = "llava_llama"


class LlavaLlamaModel(LlavaMetaModel, LlamaModel):
    config_class = LlavaConfig

    def __init__(self, config: LlamaConfig):
        super(LlavaLlamaModel, self).__init__(config)


class LlavaLlamaForCausalLM_v2(LlamaForCausalLM_v2, LlavaMetaForCausalLM):
    config_class = LlavaConfig

    def __init__(self, config):
        super().__init__(config)
        self.model = LlavaLlamaModel(config)
        self.pretraining_tp = config.pretraining_tp
        self.vocab_size = config.vocab_size

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.compress_proj = nn.MaxPool1d(kernel_size=9, stride=9)

        pts_bbox_head=dict(
            torch_dtype=torch.bfloat16,
            num_classes=1,
            in_channels=4096,
            embed_dims=512,
            num_query=1000,
            LID=True,
            with_position=True,
            with_multiview=True,
            position_range=[-1.5, -1.5, -0.5, 1.5, 1.5, 1.5],
            normedlinear=False,
            code_size=7,
            code_weights=[1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 1.0],
            bbox_coder=dict(
                type='GraspCoder',
                post_center_range=[-1.5, -1.5, -0.5, 1.5, 1.5, 1.5],
                pc_range=point_cloud_range,
                max_num=600,
                score_threshold=0.0,
                num_classes=1),
            loss_cls=dict(
                type='mmdet.FocalLoss',
                use_sigmoid=True,
                gamma=2.0,
                alpha=0.25,
                loss_weight=1.0),
            loss_bbox=dict(type='mmdet.L1Loss', loss_weight=1.0),
            loss_iou=dict(type='mmdet.GIoULoss', loss_weight=0.0))
        # model training and testing settings
        train_cfg=dict(
            grid_size=[512, 512, 1],
            voxel_size=voxel_size,
            point_cloud_range=point_cloud_range,
            out_size_factor=4,
            assigner=dict(
                type='HungarianAssigner3D',
                cls_cost=dict(type='FocalLossCost', weight=1.0),
                reg_cost=dict(type='BBox3DL1Cost', weight=1.0),
                iou_cost=dict(
                    type='IoUCost', weight=0.0
                ),  # Fake cost. Just to be compatible with DETR head.
                pc_range=point_cloud_range))

        pts_bbox_head.update(train_cfg=train_cfg)
        self.det_head = PETRHead_v2(**pts_bbox_head)
        # Initialize weights and apply final processing
        self.post_init()


    def init_weights(self):
        """Initialize weights of the transformer head."""
        # The initialization for transformer is important
        super().init_weights()
        self.det_head.init_weights()

    def get_model(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        pure_imgs: Optional[torch.FloatTensor] = None,
        depths: Optional[torch.FloatTensor] = None,
        pcs: Optional[List[torch.FloatTensor]] = None,
        poses: Optional[torch.FloatTensor] = None,
        intrinsics: Optional[torch.FloatTensor] = None,
        lengths: Optional[torch.FloatTensor] = None,
        input_token_len: Optional[int] = None,
        clicks: Optional[List[List[float]]] = None,
        image_sizes: Optional[List[List[int]]] = None,
        return_dict: Optional[bool] = None,
        generate=False,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        if inputs_embeds is None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                images,
                pure_imgs,
                depths,
                poses,
                intrinsics,
                lengths,
                pcs,
                clicks,
                image_sizes
            )

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )

        hidden_states = outputs[0]
        B, _, C = hidden_states.shape

        if generate:
            if self.config.pretraining_tp > 1:
                lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.config.pretraining_tp, dim=0)
                logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(self.config.pretraining_tp)]
                logits = torch.cat(logits, dim=-1)
            else:
                logits = self.lm_head(hidden_states)
            logits = logits.float()
            loss = None            
 
            if not return_dict:
                output = (logits,) + outputs[1:]
                return (loss,) + output if loss is not None else output

            return CausalLMOutputWithPast(
                loss=loss,
                logits=logits,
                past_key_values=outputs.past_key_values,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
            )

        dialog_labels, dialog_hidden_states = [], []
        for i in range(B):
            valid_inds = (labels[i]!=GRASP_TOKEN_INDEX) & (labels[i]!=IGNORE_INDEX)
            new_labels = labels[i][valid_inds].contiguous()
            new_labels[new_labels==TEPORARY_IGNORE_INDEX] = IGNORE_INDEX
            dialog_labels.append(new_labels)
            dialog_hidden_states.append(hidden_states[i][valid_inds].clone().contiguous())


        dialog_max_len = max(x.shape[0] for x in dialog_labels)
        dialog_hidden_states_padded = []
        dialog_labels_padded = torch.full((B, dialog_max_len), IGNORE_INDEX, dtype=dialog_labels[0].dtype, device=dialog_labels[0].device)

        for i, (cur_new_embed, cur_new_labels) in enumerate(zip(dialog_hidden_states, dialog_labels)):
            cur_len = cur_new_embed.shape[0]
            if getattr(self.config, 'tokenizer_padding_side', 'right') == "left":
                dialog_hidden_states_padded.append(torch.cat((
                    torch.zeros((dialog_max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device),
                    cur_new_embed
                ), dim=0))
                if cur_len > 0:
                    dialog_labels_padded[i, -cur_len:] = cur_new_labels
            else:
                dialog_hidden_states_padded.append(torch.cat((
                    cur_new_embed,
                    torch.zeros((dialog_max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)
                ), dim=0))
                if cur_len > 0:
                    dialog_labels_padded[i, :cur_len] = cur_new_labels

        dialog_hidden_states = torch.stack(dialog_hidden_states_padded, dim=0)
        dialog_hidden_states = torch.nan_to_num(dialog_hidden_states)

        grasp_outs = self.det_head(hidden_states, attention_mask)

        if not self.config.use_dialogue:
            return grasp_outs, None

        if self.config.pretraining_tp > 1:
            lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.config.pretraining_tp, dim=0)
            logits = [F.linear(dialog_hidden_states, lm_head_slices[i]) for i in range(self.config.pretraining_tp)]
            logits = torch.cat(logits, dim=-1)
        else:
            logits = self.lm_head(dialog_hidden_states)
        logits = logits.float()


        loss = None
        if dialog_labels_padded is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = dialog_labels_padded[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        loss *= 0.1
        
        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output + grasp_outs if loss is not None else output + grasp_outs
        
        dialog_outs = CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

        return grasp_outs, dialog_outs

    @torch.no_grad()
    def predict_grasps(
        self,
        outs
    ):
        grasp_list = self.det_head.get_bboxes(
            outs)
        return grasp_list

AutoConfig.register("llava_llama", LlavaConfig)
AutoModelForCausalLM.register(LlavaConfig, LlavaLlamaForCausalLM_v2)




