# Copyright (c) OpenMMLab. All rights reserved.
import warnings
import torch
from torch import Tensor
from mmdet.models.task_modules import AssignResult
from mmengine.structures import InstanceData

from mmdet.structures.bbox import BaseBoxes, cat_boxes
from mmdet.models.task_modules.samplers import BaseSampler
from mmdet.utils import util_mixins

class GraspSamplingResult(util_mixins.NiceRepr):
    """Bbox sampling result.

    Args:
        pos_inds (Tensor): Indices of positive samples.
        neg_inds (Tensor): Indices of negative samples.
        priors (Tensor): The priors can be anchors or points,
            or the bboxes predicted by the previous stage.
        gt_bboxes (Tensor): Ground truth of bboxes.
        assign_result (:obj:`AssignResult`): Assigning results.
        gt_flags (Tensor): The Ground truth flags.
        avg_factor_with_neg (bool):  If True, ``avg_factor`` equal to
            the number of total priors; Otherwise, it is the number of
            positive priors. Defaults to True.

    Example:
        >>> # xdoctest: +IGNORE_WANT
        >>> from mmdet.models.task_modules.samplers.sampling_result import *  # NOQA
        >>> self = GraspSamplingResult.random(rng=10)
        >>> print(f'self = {self}')
        self = <GraspSamplingResult({
            'neg_inds': tensor([1,  2,  3,  5,  6,  7,  8,
                                9, 10, 11, 12, 13]),
            'neg_priors': torch.Size([12, 4]),
            'num_gts': 1,
            'num_neg': 12,
            'num_pos': 1,
            'avg_factor': 13,
            'pos_assigned_gt_inds': tensor([0]),
            'pos_inds': tensor([0]),
            'pos_is_gt': tensor([1], dtype=torch.uint8),
            'pos_priors': torch.Size([1, 4])
        })>
    """

    def __init__(self,
                 pos_inds: Tensor,
                 neg_inds: Tensor,
                 priors: Tensor,
                 gt_bboxes: Tensor,
                 assign_result: AssignResult,
                 gt_flags: Tensor,
                 avg_factor_with_neg: bool = True) -> None:
        self.pos_inds = pos_inds
        self.neg_inds = neg_inds
        self.num_pos = max(pos_inds.numel(), 1)
        self.num_neg = max(neg_inds.numel(), 1)
        self.avg_factor_with_neg = avg_factor_with_neg
        self.avg_factor = self.num_pos + self.num_neg \
            if avg_factor_with_neg else self.num_pos
        self.pos_priors = priors[pos_inds]
        self.neg_priors = priors[neg_inds]
        self.pos_is_gt = gt_flags[pos_inds]

        self.num_gts = gt_bboxes.shape[0]
        self.pos_assigned_gt_inds = assign_result.gt_inds[pos_inds] - 1
        self.pos_gt_labels = assign_result.labels[pos_inds]

        self.box_dim = 10
        if gt_bboxes.numel() == 0:
            # hack for index error case
            assert self.pos_assigned_gt_inds.numel() == 0
            self.pos_gt_bboxes = gt_bboxes.view(-1, self.box_dim)
        else:
            if len(gt_bboxes.shape) < 2:
                gt_bboxes = gt_bboxes.view(-1, self.box_dim)
            self.pos_gt_bboxes = gt_bboxes[self.pos_assigned_gt_inds.long()]

    @property
    def priors(self):
        """torch.Tensor: concatenated positive and negative priors"""
        return cat_boxes([self.pos_priors, self.neg_priors])

    @property
    def bboxes(self):
        """torch.Tensor: concatenated positive and negative boxes"""
        warnings.warn('DeprecationWarning: bboxes is deprecated, '
                      'please use "priors" instead')
        return self.priors

    @property
    def pos_bboxes(self):
        warnings.warn('DeprecationWarning: pos_bboxes is deprecated, '
                      'please use "pos_priors" instead')
        return self.pos_priors

    @property
    def neg_bboxes(self):
        warnings.warn('DeprecationWarning: neg_bboxes is deprecated, '
                      'please use "neg_priors" instead')
        return self.neg_priors

    def to(self, device):
        """Change the device of the data inplace.

        Example:
            >>> self = GraspSamplingResult.random()
            >>> print(f'self = {self.to(None)}')
            >>> # xdoctest: +REQUIRES(--gpu)
            >>> print(f'self = {self.to(0)}')
        """
        _dict = self.__dict__
        for key, value in _dict.items():
            if isinstance(value, (torch.Tensor, BaseBoxes)):
                _dict[key] = value.to(device)
        return self

    def __nice__(self):
        data = self.info.copy()
        data['pos_priors'] = data.pop('pos_priors').shape
        data['neg_priors'] = data.pop('neg_priors').shape
        parts = [f"'{k}': {v!r}" for k, v in sorted(data.items())]
        body = '    ' + ',\n    '.join(parts)
        return '{\n' + body + '\n}'

    @property
    def info(self):
        """Returns a dictionary of info about the object."""
        return {
            'pos_inds': self.pos_inds,
            'neg_inds': self.neg_inds,
            'pos_priors': self.pos_priors,
            'neg_priors': self.neg_priors,
            'pos_is_gt': self.pos_is_gt,
            'num_gts': self.num_gts,
            'pos_assigned_gt_inds': self.pos_assigned_gt_inds,
            'num_pos': self.num_pos,
            'num_neg': self.num_neg,
            'avg_factor': self.avg_factor
        }


# @TASK_UTILS.register_module()
class GraspSampler(BaseSampler):
    """A pseudo sampler that does not do sampling actually."""

    # TODO: This is a temporary grasp sampler.

    def __init__(self, **kwargs):
        pass

    def _sample_pos(self, **kwargs):
        """Sample positive samples."""
        raise NotImplementedError

    def _sample_neg(self, **kwargs):
        """Sample negative samples."""
        raise NotImplementedError

    def sample(self, assign_result: AssignResult, pred_instances: InstanceData,
               gt_instances: InstanceData, *args, **kwargs) -> GraspSamplingResult:
        """Directly returns the positive and negative indices  of samples.

        Args:
            assign_result (:obj:`AssignResult`): Bbox assigning results.
            pred_instances (:obj:`InstaceData`): Instances of model
                predictions. It includes ``priors``, and the priors can
                be anchors, points, or bboxes predicted by the model,
                shape(n, 4).
            gt_instances (:obj:`InstaceData`): Ground truth of instance
                annotations. It usually includes ``bboxes`` and ``labels``
                attributes.

        Returns:
            :obj:`GraspSamplingResult`: sampler results
        """
        gt_bboxes = gt_instances.grasps_6d
        priors = pred_instances.priors

        pos_inds = torch.nonzero(
            assign_result.gt_inds > 0, as_tuple=False).squeeze(-1).unique()
        neg_inds = torch.nonzero(
            assign_result.gt_inds == 0, as_tuple=False).squeeze(-1).unique()

        gt_flags = priors.new_zeros(priors.shape[0], dtype=torch.uint8)
        sampling_result = GraspSamplingResult(
            pos_inds=pos_inds,
            neg_inds=neg_inds,
            priors=priors,
            gt_bboxes=gt_bboxes,
            assign_result=assign_result,
            gt_flags=gt_flags,
            avg_factor_with_neg=False)
        return sampling_result
