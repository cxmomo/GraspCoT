# Copyright (c) OpenMMLab. All rights reserved.
import os
from os import path as osp
import numpy as np


import pickle

from pytorchse3.se3 import se3_log_map

from llava.model import *

import torch
from mmengine import dump, load

from PIL import Image, ImageDraw

import argparse
from tqdm import tqdm

import cv2
from sklearn.cluster import KMeans



# x: right, y: up, z: forward

# front：fov_degrees = 0, f = 1, R = [[1, 0, 0], [0, np.cos(theta), np.sin(theta)], [0, -np.sin(theta), np.cos(theta)]], t = [0.0, -0.5, 1.0]
# bottom right: fov_degrees = 90, f = 1, R = [[np.cos(theta), 0, np.sin(theta)], [0, 1, 0], [-np.sin(theta), 0, np.cos(theta)]], t = [-0.5, 0.0, 0.5]
# BEV: fov_degrees = 180, f = 1.5, R = [[np.cos(theta), 0, np.sin(theta)], [0, 1, 0], [-np.sin(theta), 0, np.cos(theta)]], t = [0, 0.0, 0.5]
# top：fov_degrees = 270, f = 2.0, R = [[1, 0, 0], [0, np.cos(theta), np.sin(theta)], [0, -np.sin(theta), np.cos(theta)]], t = [0.0, 0.5, 1.0]

max_num = 100

MAX_WIDTH = 0.202   # maximum width of gripper 2F-140
width, height = 336, 336
seperate_num = 100000

def project_points(pc_xyz, K, ext):
    pc_homogeneous = torch.cat((pc_xyz, torch.ones(pc_xyz.shape[0], 1, dtype=torch.float32)), dim=1)
    proj_matrix = K @ ext
    proj_points = proj_matrix @ pc_homogeneous.t()
    proj_points[:2, :] = proj_points[:2, :] / (proj_points[2, :]+1e-4) 
    return proj_points.t()


def points2deprgb(points, pc_rgb, height, width):
    points[:, 0] = points[:, 0] * width
    points[:, 1] = points[:, 1] * height
    depth_map = torch.zeros((height, width), dtype=torch.float32)
    img_rgb = torch.zeros((height, width, 3), dtype=torch.float32)
    coor = torch.round(points[:, :2])
    depth = points[:, 2]
    kept1 = (coor[:, 0] >= 0) & (coor[:, 0] < width) & (
        coor[:, 1] >= 0) & (coor[:, 1] < height)
    coor, depth, rgb = coor[kept1], depth[kept1], pc_rgb[kept1]
    ranks = coor[:, 0] + coor[:, 1] * width
    sort = (ranks + depth / 100.).argsort()
    coor, depth, rgb, ranks = coor[sort], depth[sort], rgb[sort], ranks[sort]

    kept2 = torch.ones(coor.shape[0], device=coor.device, dtype=torch.bool)
    kept2[1:] = (ranks[1:] != ranks[:-1])
    coor, depth, rgb = coor[kept2], depth[kept2], rgb[kept2]
    coor = coor.to(torch.long)
    depth_map[coor[:, 1], coor[:, 0]] = depth
    img_rgb[coor[:, 1], coor[:, 0]] = rgb

    depth_map = (depth_map * 255).to(torch.uint8)
    img_rgb = (img_rgb * 255).to(torch.uint8)
    return depth_map, img_rgb


def pruning_grasps(grasps, rot_weight=1.0, trans_weight=3.0, w_weight=0.1, max_num=100):

    rots = grasps[:, :3]
    trans = grasps[:, 3:6]
    widths = grasps[:, 6]

    rot_distance = torch.norm(rots[:, None, :] - rots[None, :, :], dim=2)
    trans_distance = torch.norm(trans[:, None, :] - trans[None, :, :], dim=2)
    width_distance = torch.abs(widths[:, None] - widths[None, ])

    similarity_matrix = rot_weight * rot_distance + trans_weight * trans_distance + w_weight * width_distance

    kmeans = KMeans(n_clusters=max_num, random_state=0, n_init='auto').fit(grasps.detach().numpy())
    cluster_labels = kmeans.labels_

    selected_grasps, selected_ids = [], []
    for cluster_id in range(max_num):
        cluster_indices = np.where(cluster_labels == cluster_id)[0]
        if len(cluster_indices) == 1:
            selected_grasps.append(grasps[cluster_indices[0]])
            selected_ids.append(cluster_indices[0])
            continue
       
        grasps_incluster = grasps[cluster_indices]
        similarity_sum = torch.sum(similarity_matrix[cluster_indices, :][:, cluster_indices], dim=1)
        min_similarity_index = torch.argmin(similarity_sum, dim=0)
        selected_grasps.append(grasps_incluster[min_similarity_index])
        selected_ids.append(int(cluster_indices[min_similarity_index]))
    selected_grasps_tensor = torch.stack(selected_grasps)
    selected_ids = torch.Tensor(selected_ids).to(torch.int64)
    return selected_grasps_tensor

def _fill_grasp_trainval_infos(id=0, version="train", pruning=False):
    """Generate the train/val infos from the raw data.

    Args:
        nusc (:obj:`NuScenes`): Dataset class in the nuScenes dataset.
        train_scenes (list[str]): Basic information of training scenes.
        val_scenes (list[str]): Basic information of validation scenes.
        test (bool, optional): Whether use the test mode. In test mode, no
            annotations can be accessed. Default: False.
        max_sweeps (int, optional): Max number of sweeps. Default: 10.

    Returns:
        tuple[list[dict]]: Information of training set and validation set
            that will be saved to the info file.
    """
    infos = []
    
    filenames = sorted(os.listdir("data/grasp_anything/pc"))

    print("Processing dataset for {} set!".format(version))

    id = 0 # glyou debug
    if version == "train":
        part_filenames = filenames[id*seperate_num:min((id+1)*seperate_num, int(len(filenames)*4/5))]   # 80% scenes for training
    else:
        part_filenames = filenames[int(len(filenames)*4/5)+id*seperate_num:min(int(len(filenames)*4/5)+(id+1)*seperate_num, len(filenames))]   # 20% scenes for val

    for jj, filename in enumerate(tqdm(part_filenames)):
        scene, _ = os.path.splitext(filename)
        try: 
            with open(f"data/grasp_anything/grasp_prompt/{scene}.pkl", "rb") as f:
                prompts = pickle.load(f)
        except:
            continue

        num_objects = len(prompts)
        gs_list = []
        gs_label_list = []

        pos_prompt_list = []
        for i in range(num_objects):
            try:
                with open(f"data/grasp_anything/grasp/{scene}_{i}", "rb") as f:
                    Rts, ws = pickle.load(f)
            except:
                continue
            
            gs = torch.from_numpy(np.concatenate((se3_log_map(torch.from_numpy(Rts)).numpy(), 2*ws[:, None]/MAX_WIDTH-1.0), axis=-1)).to(torch.float32)
            gs_labels = torch.ones_like(gs[..., :1], dtype=torch.int64)

            if pruning:
                num_grasps = len(gs)
                if num_grasps<=max_num:
                    gs_list.append(gs)
                    gs_label_list.append(gs_labels)
                    continue
                assert gs.dim() == 2
                pruned_grasps = pruning_grasps(gs, max_num=max_num)
                gs_list.append(pruned_grasps)
                gs_label_list.append(gs_labels[:len(pruned_grasps)])
            else:
                gs_list.append(gs)
                gs_label_list.append(gs_labels)

            pos_prompt_list.append(prompts[i])

        if len(gs_list)==0:
            continue

        pc = np.load(f"data/grasp_anything/pc/{scene}.npy")
        
        pc_ori = torch.from_numpy(pc).to(torch.float32)
        pc_xyz = pc_ori[..., :3] # 0-1
        pc_rgb = pc_ori[..., 3:] # 0-1
        
        n_view = 4
        fov_degree_list = [0, 90, 180, 270]
        
        f_list = [1, 1, 1.5, 2.0]

        t_list = [[0.0, -0.5, 1.0],
                  [-0.5, 0.0, 0.5],
                  [0, 0.0, 0.5],
                  [0, 0.5, 1.0],
                ]
        
        theta_list = []
        for k in range(n_view):
            theta_list.append(np.radians(90 - fov_degree_list[k] / 2))
        
        R_list = [
            [[1, 0, 0], [0, np.cos(theta_list[0]), np.sin(theta_list[0])], [0, -np.sin(theta_list[0]), np.cos(theta_list[0])]],
            [[np.cos(theta_list[1]), 0, np.sin(theta_list[1])], [0, 1, 0], [-np.sin(theta_list[1]), 0, np.cos(theta_list[1])]],
            [[np.cos(theta_list[2]), 0, np.sin(theta_list[2])], [0, 1, 0], [-np.sin(theta_list[2]), 0, np.cos(theta_list[2])]],
            [[1, 0, 0], [0, np.cos(theta_list[3]), np.sin(theta_list[3])], [0, -np.sin(theta_list[3]), np.cos(theta_list[3])]],
            ]


        depth_map_list, ext_list, K_list = [], [], []
        for k in range(n_view):
            f = f_list[k]

            R = torch.tensor(R_list[k], dtype=torch.float32)
            t = torch.tensor(t_list[k], dtype=torch.float32)

            cx, cy = 0.5, 0.5
            K = torch.tensor([[f, 0, cx, 0],
                            [0, f, cy, 0],
                            [0, 0,  1, 0]], dtype=torch.float32)

            ext = torch.eye(4, dtype=torch.float32)
            ext[:3, :3] = R
            ext[:3, 3] = t
            
            proj_points = project_points(pc_xyz, K, ext)
            
            depth_map, img_rgb = points2deprgb(proj_points, pc_rgb, width, height)
            
            if not os.path.exists(f"data/grasp_anything/rgb/{scene}_{str(k)}.png"):
                cv2.imwrite(f"data/grasp_anything/rgb/{scene}_{str(k)}.png", cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
            
            # img_rgb_list.append(img_rgb) # 0-255
            depth_map_list.append(depth_map) # 0-255
            ext_list.append(torch.linalg.inv(ext))
            K_list.append(K)
        # img_rgb_all = torch.stack(img_rgb_list).numpy()
        depth_map_all = torch.stack(depth_map_list).numpy()
        ext_all = torch.stack(ext_list).to(torch.float32).numpy()
        K_all =  torch.stack(K_list).to(torch.float32).numpy()
        
        if not os.path.exists(f"data/grasp_anything/depth/{scene}.npy"):
            np.save(f"data/grasp_anything/depth/{scene}.npy", depth_map_all)

        info = {
            'scene_token': scene,
            'gs_prompts': pos_prompt_list,
            'gs': gs_list,
            'gs_labels': gs_label_list,
            'pc_path': filename,
            'img_path': f"rgb/{scene}",
            'depth_path': f"depth/{scene}",
            'pose': ext_all,
            'intrinsic': K_all,
        }
        infos.append(info)

    return infos


def create_grasp_infos(root_path,
                          info_prefix,
                          version='train',
                          pruning=True,
                          id=0):
    """Create info file of nuscene dataset.

    Given the raw data, generate its related info file in pkl format.

    Args:
        root_path (str): Path of the data root.
        info_prefix (str): Prefix of the info file to be generated.
        version (str, optional): Version of the data.
            Default: 'v1.0-trainval'.
        max_sweeps (int, optional): Max number of sweeps.
            Default: 10.
    """

    infos = _fill_grasp_trainval_infos(id=id, version=version, pruning=pruning)

    metadata = dict(version=version)

    print('{} sample: {}'.format(version, len(infos)))
    data = dict(infos=infos, metadata=metadata)
    info_path = osp.join(root_path,
                            '{}_infos_{}_'.format(info_prefix, version)+str(id)+'.pkl')
    dump(data, info_path)
    print('Finish {}_infos_{}_'.format(info_prefix, version)+str(id))



def parse_args():
    parser = argparse.ArgumentParser(description="Create grasp data! ")
    parser.add_argument('--version', required=True)
    parser.add_argument('--pruning', action="store_true")
    parser.add_argument("--id", type=int, help="dataset is too big and needs ids to seperate.")
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    create_grasp_infos("data/grasp_anything/",
                          "grasp_anything",
                          version=args.version,
                          pruning=args.pruning,
                          id=args.id)
