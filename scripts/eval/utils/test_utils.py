import torch
from pytorchse3.se3 import se3_log_map, se3_exp_map
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
import trimesh
import open3d as o3d

MAX_WIDTH = 0.202
NORMAL_WIDTH = 0.140


def create_gripper_marker(width_scale=1.0, color=[0, 255, 0], tube_radius=0.005, sections=6):
    """Create a 3D mesh visualizing a parallel yaw gripper. It consists of four cylinders.

    Args:
        width_scale (float, optional): Scale of the grasp with w.r.t. the normal width of 140mm, i.e., 0.14. 
        color (list, optional): RGB values of marker.
        tube_radius (float, optional): Radius of cylinders.
        sections (int, optional): Number of sections of each cylinder.

    Returns:
        trimesh.Trimesh: A mesh that represents a simple parallel yaw gripper.
    """
    cfl = trimesh.creation.cylinder(
        radius=tube_radius,
        sections=sections,
        segment=[
            [7.10000000e-02*width_scale, -7.27595772e-12, 1.154999996e-01],
            [7.10000000e-02*width_scale, -7.27595772e-12, 1.959999998e-01],
        ],
    )
    cfr = trimesh.creation.cylinder(
        radius=tube_radius,
        sections=sections,
        segment=[
            [-7.100000e-02*width_scale, -7.27595772e-12, 1.154999996e-01],
            [-7.100000e-02*width_scale, -7.27595772e-12, 1.959999998e-01],
        ],
    )
    cb1 = trimesh.creation.cylinder(
        radius=tube_radius, sections=sections, segment=[[0, 0, 0], [0, 0, 1.154999996e-01]]
    )
    cb2 = trimesh.creation.cylinder(
        radius=tube_radius,
        sections=sections,
        segment=[[-7.100000e-02*width_scale, 0, 1.154999996e-01], [7.100000e-02*width_scale, 0, 1.154999996e-01]],
    )

    tmp = trimesh.util.concatenate([cb1, cb2, cfr, cfl])
    tmp.visual.face_colors = color

    return tmp


def earth_movers_distance(train_grasps, gen_grasps):
    """
    Compute Earth Mover's Distance between two sets of vectors.
    """
    # Ensure the input sets have the same dimensionality
    assert train_grasps.shape[1] == gen_grasps.shape[1]
    if np.isnan(train_grasps).any() or np.isnan(gen_grasps).any():
        raise ValueError("NaN values exist!")
    # Calculate pairwise distances between vectors in the sets
    distances = np.linalg.norm(train_grasps[:, np.newaxis] - gen_grasps, axis=-1)
    # Solve the linear sum assignment problem
    _, assignment = linear_sum_assignment(distances)
    # Compute the total Earth Mover's Distance
    emd = distances[np.arange(len(assignment)), assignment].mean()
    return emd

def coverage_rate(train_grasps, gen_grasps, dist_thr=0.4):
    """
    Function to compute the coverage rate metric.
    """
    assert train_grasps.shape[1] == gen_grasps.shape[1]
    if np.isnan(train_grasps).any() or np.isnan(gen_grasps).any():
        raise ValueError("NaN values exist!")
    dist = cdist(train_grasps, gen_grasps)
    rate = np.sum(np.any(dist <= dist_thr, axis=1)) / train_grasps.shape[0]
    return rate


def preprocess_point_cloud(pc, voxel_size=0.02, nb_neighbors=15, std_ratio=2.0):
    """
    点云预处理函数，使用体素下采样和统计滤波去噪，并创建 KD 树

    :param pc: 输入的点云数据，numpy 数组
    :param voxel_size: 体素大小
    :param nb_neighbors: 统计滤波的邻居数量
    :param std_ratio: 标准差倍数
    :return: 下采样和去噪后的点云数据，open3d 中的 KD 树
    """
    # 将输入点云数据转换为 open3d.geometry.PointCloud 对象
    pcd_cpu = o3d.geometry.PointCloud()
    pcd_cpu.points = o3d.utility.Vector3dVector(pc[:, :3])

    pcd_cpu, _ = pcd_cpu.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    
    # 体素下采样
    pcd_cpu = pcd_cpu.voxel_down_sample(voxel_size=voxel_size)
    
    # 统计滤波去噪
    pcd_cpu = o3d.geometry.PointCloud(pcd_cpu)  # 明确转换为 open3d.geometry.PointCloud

    return np.asarray(pcd_cpu.points)

def collision_check(pc, gripper):
    """
    Function to check collision between a point cloud and a gripper.
    """
    return np.sum(gripper.contains(pc)) > 0

def earth_movers_distance_single(train_grasps, gen_grasps):
    """
    Compute Earth Mover's Distance between two sets of vectors.
    """
    # Ensure the input sets have the same dimensionality
    assert train_grasps.shape[1] == gen_grasps.shape[1]
    if np.isnan(train_grasps).any() or np.isnan(gen_grasps).any():
        raise ValueError("NaN values exist!")
    # Calculate pairwise distances between vectors in the sets
    distances = np.linalg.norm(gen_grasps[:, np.newaxis] - train_grasps, axis=-1)
    # Solve the linear sum assignment problem
    indice, assignment = linear_sum_assignment(distances)
    gen_grasps = gen_grasps[indice]
    # Compute the total Earth Mover's Distance
    emd = distances[np.arange(len(assignment)), assignment]
    return emd, gen_grasps


def collision_free_rate(pc, grasps, gt_grasps):
    """
    Function to compute the collision rate metric.
    pc' size: N x 6
    """
    distance, grasps = earth_movers_distance_single(gt_grasps, grasps)
    coff = np.abs(np.sqrt(7.7149)-distance)/np.sqrt(7.7149)

    if np.isnan(grasps).any():
        return None
    pc = pc[:, :3]  # use only the coordinates of the point cloud
    pc = preprocess_point_cloud(pc)
    Rts, ws = se3_exp_map(torch.from_numpy(grasps[:, :-1])).numpy(), (grasps[:, -1] + 1.0)*MAX_WIDTH/2
    grippers = [create_gripper_marker(width_scale=w/NORMAL_WIDTH).apply_transform(Rt) for w, Rt in zip(ws, Rts)]
    collision_free_rate = np.mean(np.array([(1.0-collision_check(pc, gripper)) * coff[i] for i, gripper in enumerate(grippers)]))
    return collision_free_rate


def collision_free_rate_improved(pc, grasps):
    pass