# ------------------------------------------------------------------------
# Copyright (c) 2021 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from DETR3D (https://github.com/WangYueFt/detr3d)
# Copyright (c) 2021 Wang, Yue
# ------------------------------------------------------------------------
# Modified from mmdetection3d (https://github.com/open-mmlab/mmdetection3d)
# Copyright (c) OpenMMLab. All rights reserved.
# ------------------------------------------------------------------------
import torch
import torch.nn.functional as F
from mmdet.models.task_modules import BaseBBoxCoder

from mmengine.registry import TASK_UTILS

def denormalize_grasp(normalized_grasps):
    """ denormalize bboxes
        Args:
            normalized_bboxes (Tensor): boxes with normalized coordinate
                (cx,cy,L,W,cz,H,sin(φ),cos(φ),v_x,v_y).
                All in range [0, 1] and shape [num_query, 10].
            pc_range (List): Perception range of the detector
        Returns:
            denormalized_bboxes (Tensor): boxes with unnormalized
                coordinates (cx,cy,cz,L,W,H,φ,v_x,v_y). Shape [num_gt, 9].
    """
    # rotation
    x_rot = normalized_grasps[..., 0:1]
    
    y_rot = normalized_grasps[..., 1:2]

    z_rot = normalized_grasps[..., 2:3]

    cx = normalized_grasps[..., 3:4]
    cy = normalized_grasps[..., 4:5]
    cz = normalized_grasps[..., 5:6]

    w = normalized_grasps[..., 6:7]

    denormalized_grasps = torch.cat([x_rot, y_rot, z_rot, cx, cy, cz, w], dim=-1)

    return denormalized_grasps


@TASK_UTILS.register_module()
class GraspCoder(BaseBBoxCoder):
    """Bbox coder for NMS-free detector.

    Args:
        pc_range (list[float]): Range of point cloud.
        post_center_range (list[float]): Limit of the center.
            Default: None.
        max_num (int): Max number to be kept. Default: 100.
        score_threshold (float): Threshold to filter boxes based on score.
            Default: None.
        code_size (int): Code size of bboxes. Default: 9
    """

    def __init__(self,
                 pc_range,
                 post_center_range=None,
                 max_num=1000,
                 score_threshold=None,
                 num_classes=1):

        self.pc_range = pc_range
        self.post_center_range = post_center_range
        self.max_num = max_num
        self.score_threshold = score_threshold
        self.num_classes = num_classes

    def encode(self):
        pass

    def decode_single(self, cls_scores, bbox_preds):
        """Decode bboxes.

        Args:
            cls_scores (Tensor): Outputs from the classification head, \
                shape [num_query, cls_out_channels]. Note \
                cls_out_channels should includes background.
            bbox_preds (Tensor): Outputs from the regression \
                head with normalized coordinate format \
                (cx, cy, w, l, cz, h, rot_sine, rot_cosine, vx, vy). \
                Shape [num_query, 9].
        Returns:
            list[dict]: Decoded boxes.
        """
        max_num = cls_scores.shape[0]

        cls_scores = cls_scores.sigmoid()
        scores, indexes = cls_scores.view(-1).topk(max_num)
        labels = indexes % self.num_classes
        bbox_index = indexes // self.num_classes
        bbox_preds = bbox_preds[bbox_index]

        final_box_preds = denormalize_grasp(bbox_preds)
        final_scores = scores
        final_preds = labels

        # use score threshold
        if self.score_threshold is not None:
            thresh_mask = final_scores > self.score_threshold
        if self.post_center_range is not None:
            self.post_center_range = torch.tensor(
                self.post_center_range, device=scores.device)

            mask = (final_box_preds[..., 3:6] >=
                    self.post_center_range[:3]).all(1)
            mask &= (final_box_preds[..., 3:6] <=
                     self.post_center_range[3:]).all(1)

            if self.score_threshold:
                mask &= thresh_mask

            boxes3d = final_box_preds[mask]
            scores = final_scores[mask]
            labels = final_preds[mask]
            predictions_dict = {
                'bboxes': boxes3d,
                'scores': scores,
                'labels': labels
            }

        else:
            raise NotImplementedError(
                'Need to reorganize output as a batch, only '
                'support post_center_range is not None for now!')
        return predictions_dict

    def decode(self, preds_dicts):
        """Decode bboxes.

        Args:
            all_cls_scores (Tensor): Outputs from the classification head, \
                shape [nb_dec, bs, num_query, cls_out_channels]. Note \
                cls_out_channels should includes background.
            all_bbox_preds (Tensor): Sigmoid outputs from the regression \
                head with normalized coordinate format \
                (cx, cy, w, l, cz, h, rot_sine, rot_cosine, vx, vy). \
                Shape [nb_dec, bs, num_query, 9].
        Returns:
            list[dict]: Decoded boxes.
        """
        all_cls_scores = preds_dicts['all_cls_scores'][-1]
        all_bbox_preds = preds_dicts['all_bbox_preds'][-1]

        batch_size = all_cls_scores.size()[0]
        predictions_list = []
        for i in range(batch_size):
            predictions_list.append(
                self.decode_single(all_cls_scores[i], all_bbox_preds[i]))
        return predictions_list