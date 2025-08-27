import pickle
import numpy as np
from utils import *
import argparse
from tqdm import tqdm
from utils.test_utils import earth_movers_distance, coverage_rate, collision_free_rate

import torch


dataset_path = "data/grasp_anything"

pc_range = [-1.0, -1.0, 0, 1.0, 1.0, 1]
rot_range = [-1, -1, 0, 1, 1, torch.pi]
w_range = [-1, 1]

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a model")
    parser.add_argument("--data", type=str, help="path to the data")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    with open(args.data, "rb") as f:
       generated_data = pickle.load(f)
    
    cvr_list_04, cvr_list_03, cvr_list_02 = [], [], []
    for scene, datapoint in tqdm(generated_data.items()):
        gt_grasps = datapoint["gt_grasps"].cpu().numpy()
        gen_grasps = datapoint["gen_grasps"].cpu().numpy()

        valud_inds = (gen_grasps[:, 0]>rot_range[0]) & (gen_grasps[:, 0]<rot_range[3]) \
            & (gen_grasps[:, 1]>rot_range[1]) & (gen_grasps[:, 1]<rot_range[4]) \
            & (gen_grasps[:, 2]>rot_range[2]) & (gen_grasps[:, 2]<rot_range[5]) \
            & (gen_grasps[:, 3]>pc_range[0]) & (gen_grasps[:, 3]<pc_range[3]) \
            & (gen_grasps[:, 4]>pc_range[1]) & (gen_grasps[:, 4]<pc_range[4]) \
            & (gen_grasps[:, 5]>pc_range[2]) & (gen_grasps[:, 5]<pc_range[5]) \
            & (gen_grasps[:, 6]>w_range[0]) & (gen_grasps[:, 6]<w_range[1])

        gen_grasps = gen_grasps[valud_inds][:600]
  
        cvr_i_04 = coverage_rate(gt_grasps, gen_grasps, dist_thr=0.4)
        cvr_i_03 = coverage_rate(gt_grasps, gen_grasps, dist_thr=0.3)
        cvr_i_02 = coverage_rate(gt_grasps, gen_grasps, dist_thr=0.2)

        if cvr_i_04 is not None:
            cvr_list_04.append(cvr_i_04)
        if cvr_i_03 is not None:
            cvr_list_03.append(cvr_i_03)
        if cvr_i_02 is not None:
            cvr_list_02.append(cvr_i_02)
    cvr04 = np.array(cvr_list_04)
    cvr03 = np.array(cvr_list_03)
    cvr02 = np.array(cvr_list_02)
    print(f"Average CR 0.4: {np.mean(cvr04)}")
    print(f"Average CR 0.3: {np.mean(cvr03)}")
    print(f"Average CR 0.2: {np.mean(cvr02)}")

    
    emd_list = []
    for scene, datapoint in tqdm(generated_data.items()):
        gt_grasps = datapoint["gt_grasps"].cpu().numpy()
        gen_grasps = datapoint["gen_grasps"].cpu().numpy()

        valud_inds = (gen_grasps[:, 0]>rot_range[0]) & (gen_grasps[:, 0]<rot_range[3]) \
            & (gen_grasps[:, 1]>rot_range[1]) & (gen_grasps[:, 1]<rot_range[4]) \
            & (gen_grasps[:, 2]>rot_range[2]) & (gen_grasps[:, 2]<rot_range[5]) \
            & (gen_grasps[:, 3]>pc_range[0]) & (gen_grasps[:, 3]<pc_range[3]) \
            & (gen_grasps[:, 4]>pc_range[1]) & (gen_grasps[:, 4]<pc_range[4]) \
            & (gen_grasps[:, 5]>pc_range[2]) & (gen_grasps[:, 5]<pc_range[5]) \
            & (gen_grasps[:, 6]>w_range[0]) & (gen_grasps[:, 6]<w_range[1])

        gen_grasps = gen_grasps[valud_inds][:600]

        emd_i = earth_movers_distance(gt_grasps, gen_grasps)
        if emd_i is not None:
            emd_list.append(emd_i)
    emds = np.array(emd_list)
    print(f"Average EMD: {np.mean(emds)}") 

    cfr_list = []
    for scene, datapoint in tqdm(generated_data.items()):
        pc_path = datapoint["pc_path"]
        pc = np.load(f"{dataset_path}/pc/{scene}.npy")

        gt_grasps = datapoint["gt_grasps"].cpu().numpy()
        gen_grasps = datapoint["gen_grasps"].cpu().numpy()

        valud_inds = (gen_grasps[:, 0]>rot_range[0]) & (gen_grasps[:, 0]<rot_range[3]) \
            & (gen_grasps[:, 1]>rot_range[1]) & (gen_grasps[:, 1]<rot_range[4]) \
            & (gen_grasps[:, 2]>rot_range[2]) & (gen_grasps[:, 2]<rot_range[5]) \
            & (gen_grasps[:, 3]>pc_range[0]) & (gen_grasps[:, 3]<pc_range[3]) \
            & (gen_grasps[:, 4]>pc_range[1]) & (gen_grasps[:, 4]<pc_range[4]) \
            & (gen_grasps[:, 5]>pc_range[2]) & (gen_grasps[:, 5]<pc_range[5]) \
            & (gen_grasps[:, 6]>w_range[0]) & (gen_grasps[:, 6]<w_range[1])

        gen_grasps = gen_grasps[valud_inds][:600]

        if gen_grasps.shape[0]==0:
            continue

        cfr_i = collision_free_rate(pc, gen_grasps, gt_grasps)
        # cfr = collision_free_rate(pc, gt_grasps)
        if cfr_i is not None:
            cfr_list.append(cfr_i)
    cfr = np.array(cfr_list)
    print(f"Average CFR: {np.mean(cfr)}")   