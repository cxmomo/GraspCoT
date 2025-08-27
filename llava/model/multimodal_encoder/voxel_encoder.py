import torch
import torch.nn as nn
from torch.nn import functional as F

from torch.autograd import Function
from torch.nn.modules.utils import _pair

from mmdet3d.registry import MODELS
from mmdet3d.models.data_preprocessors.voxelize import VoxelizationByGridShape, dynamic_scatter_3d
from .unproject import voxelize as voxelize_for_scatter

class PillarVoxelTower(nn.Module):
    def __init__(self, args):
        super().__init__()
        point_cloud_range = [-0.5, -0.5, 0, 0.5, 0.5, 1]
        # change for real
        self.voxel_type = 'hard'
        voxel_layer=dict(
            max_num_points=32,
            point_cloud_range=point_cloud_range,
            voxel_size=[0.025, 0.025, 1.0],
            max_voxels=(16000, 40000))

        voxel_encoder=dict(
            type='PillarFeatureNet',
            in_channels=6,
            feat_channels=[256],
            with_distance=False,
            voxel_size=[0.025, 0.025, 1.0],
            point_cloud_range=point_cloud_range)

        middle_encoder=dict(
            type='PointPillarsScatter', in_channels=256, output_shape=[40, 40])

        self.voxel_layer = VoxelizationByGridShape(**voxel_layer)
        self.voxel_encoder = MODELS.build(voxel_encoder)
        self.middle_encoder = MODELS.build(middle_encoder)     
        self.conv_proj = nn.Sequential(
            nn.Conv2d(256, 1024, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(1024, 4096, 1, stride=2, padding=0))


    @torch.no_grad()
    def forward(self, points):

        points_new = [pc.to(torch.float32) for pc in points]
        voxel_dict = self.voxelize(points_new)
        voxel_features = self.voxel_encoder(voxel_dict['voxels'].to(torch.bfloat16),
                                            voxel_dict['num_points'],
                                            voxel_dict['coors'])
        batch_size = voxel_dict['coors'][-1, 0].item() + 1

        spatial_features = self.middle_encoder(voxel_features, voxel_dict['coors'],
                            batch_size)


        spatial_features = self.conv_proj(spatial_features)
        B, C, H, W = spatial_features.shape
        spatial_features = spatial_features.view(B, C, H* W)
        new_spatial_features, batch_offset = [], []
        for i in range(B):
            valid_inds = (spatial_features[i].sum(0) != 0)
            valid_spatial_features = spatial_features[i][:, valid_inds]
            new_spatial_features.append(valid_spatial_features.permute(1,0).to(torch.bfloat16))
            batch_offset.append(valid_spatial_features.shape[-1]+1)

        return new_spatial_features, batch_offset



    @torch.no_grad()
    def voxelize(self, points):
        """Apply voxelization to point cloud.

        Args:
            points (List[Tensor]): Point cloud in one data batch.
            data_samples: (list[:obj:`Det3DDataSample`]): The annotation data
                of every samples. Add voxel-wise annotation for segmentation.

        Returns:
            Dict[str, Tensor]: Voxelization information.

            - voxels (Tensor): Features of voxels, shape is MxNxC for hard
              voxelization, NxC for dynamic voxelization.
            - coors (Tensor): Coordinates of voxels, shape is Nx(1+NDim),
              where 1 represents the batch index.
            - num_points (Tensor, optional): Number of points in each voxel.
            - voxel_centers (Tensor, optional): Centers of voxels.
        """

        voxel_dict = dict()

        if self.voxel_type == 'hard':
            voxels, coors, num_points, voxel_centers = [], [], [], []
            for i, res in enumerate(points):
                res_voxels, res_coors, res_num_points = self.voxel_layer(res)
                res_voxel_centers = (
                    res_coors[:, [2, 1, 0]] + 0.5) * res_voxels.new_tensor(
                        self.voxel_layer.voxel_size) + res_voxels.new_tensor(
                            self.voxel_layer.point_cloud_range[0:3])
                res_coors = F.pad(res_coors, (1, 0), mode='constant', value=i)
                voxels.append(res_voxels)
                coors.append(res_coors)
                num_points.append(res_num_points)
                voxel_centers.append(res_voxel_centers)

            voxels = torch.cat(voxels, dim=0)
            coors = torch.cat(coors, dim=0)
            num_points = torch.cat(num_points, dim=0)
            voxel_centers = torch.cat(voxel_centers, dim=0)

            voxel_dict['num_points'] = num_points
            voxel_dict['voxel_centers'] = voxel_centers
        elif self.voxel_type == 'dynamic':
            coors = []
            # dynamic voxelization only provide a coors mapping
            for i, res in enumerate(points):
                res_coors = self.voxel_layer(res)
                res_coors = F.pad(res_coors, (1, 0), mode='constant', value=i)
                coors.append(res_coors)
            voxels = torch.cat(points, dim=0)
            coors = torch.cat(coors, dim=0)

        else:
            raise ValueError(f'Invalid voxelization type {self.voxel_type}')

        voxel_dict['voxels'] = voxels
        voxel_dict['coors'] = coors

        return voxel_dict

    @property
    def dtype(self):
        return self.dtype

    @property
    def device(self):
        return self.device



class SecondVoxelTower(nn.Module):
    def __init__(self, args):
        super().__init__()
        point_cloud_range = [-0.5, -0.5, 0, 0.5, 0.5, 1]
        self.voxel_type = 'hard'
        voxel_layer=dict(
            max_num_points=10,
            point_cloud_range=point_cloud_range,
            voxel_size=[0.0125, 0.0125, 0.0125],
            max_voxels=(16000, 40000))

        voxel_encoder=dict(type='HardSimpleVFE')
        
        middle_encoder=dict(
            type='SparseEncoder',
            in_channels=3,
            output_channels=4096,
            # output_channels=2048,
            sparse_shape=[80, 80, 80],
            order=('conv', 'norm', 'act'))

        self.voxel_layer = VoxelizationByGridShape(**voxel_layer)
        self.voxel_encoder = MODELS.build(voxel_encoder)
        self.middle_encoder = MODELS.build(middle_encoder)     



    @torch.no_grad()
    def forward(self, points):

        voxel_dict = self.voxelize(points)
        voxel_features = self.voxel_encoder(voxel_dict['voxels'],
                                            voxel_dict['num_points'],
                                            voxel_dict['coors'])
        batch_size = voxel_dict['coors'][-1, 0].item() + 1
        spatial_features = self.middle_encoder(voxel_features.to(torch.float32), voxel_dict['coors'].to(torch.int32),
                            batch_size)
        
        B, C, D, H, W = spatial_features.shape
        spatial_features = spatial_features.view(B, C, D * H* W)
        new_spatial_features, batch_offset = [], []
        for i in range(B):
            valid_inds = spatial_features[i].sum(0) > 0
            valid_spatial_features = spatial_features[i][:, valid_inds]
            new_spatial_features.append(valid_spatial_features.permute(1,0).to(torch.bfloat16))
            batch_offset.append(valid_spatial_features.shape[-1]+1)

        return new_spatial_features, batch_offset



    @torch.no_grad()
    def voxelize(self, points):
        """Apply voxelization to point cloud.

        Args:
            points (List[Tensor]): Point cloud in one data batch.
            data_samples: (list[:obj:`Det3DDataSample`]): The annotation data
                of every samples. Add voxel-wise annotation for segmentation.

        Returns:
            Dict[str, Tensor]: Voxelization information.

            - voxels (Tensor): Features of voxels, shape is MxNxC for hard
              voxelization, NxC for dynamic voxelization.
            - coors (Tensor): Coordinates of voxels, shape is Nx(1+NDim),
              where 1 represents the batch index.
            - num_points (Tensor, optional): Number of points in each voxel.
            - voxel_centers (Tensor, optional): Centers of voxels.
        """

        voxel_dict = dict()

        if self.voxel_type == 'hard':
            voxels, coors, num_points, voxel_centers = [], [], [], []
            for i, res in enumerate(points):
                res_voxels, res_coors, res_num_points = self.voxel_layer(res)
                res_voxel_centers = (
                    res_coors[:, [2, 1, 0]] + 0.5) * res_voxels.new_tensor(
                        self.voxel_layer.voxel_size) + res_voxels.new_tensor(
                            self.voxel_layer.point_cloud_range[0:3])
                res_coors = F.pad(res_coors, (1, 0), mode='constant', value=i)
                voxels.append(res_voxels)
                coors.append(res_coors)
                num_points.append(res_num_points)
                voxel_centers.append(res_voxel_centers)

            voxels = torch.cat(voxels, dim=0)
            coors = torch.cat(coors, dim=0)
            num_points = torch.cat(num_points, dim=0)
            voxel_centers = torch.cat(voxel_centers, dim=0)

            voxel_dict['num_points'] = num_points
            voxel_dict['voxel_centers'] = voxel_centers
        elif self.voxel_type == 'dynamic':
            coors = []
            # dynamic voxelization only provide a coors mapping
            for i, res in enumerate(points):
                res_coors = self.voxel_layer(res)
                res_coors = F.pad(res_coors, (1, 0), mode='constant', value=i)
                coors.append(res_coors)
            voxels = torch.cat(points, dim=0)
            coors = torch.cat(coors, dim=0)

        else:
            raise ValueError(f'Invalid voxelization type {self.voxel_type}')

        voxel_dict['voxels'] = voxels
        voxel_dict['coors'] = coors

        return voxel_dict

    # @property
    # def dummy_feature(self):
    #     return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return self.dtype

    @property
    def device(self):
        return self.device


