"""
Convert foundationpose predictions to the same format as predictions from our method
"""

import os
import numpy as np
import torch
import json
import argparse
import shutil
TMF_TO_REALSENSE_TF = np.array(
    [
        [-1.0, 0.0, 0.0, 0.06452],
        [0.0, -1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.021676],
        [0.0, 0.0, 0.0, 1.0],
    ]
)

DSET_PATH = "/home/carter/projects/SPAD-6D-Pose-Capture/data/captures/YCB/mustard_refined_gt"
FOUNDATIONPOSE_PREDS_PATH = "6d_pose_estimation/results/vis_for_paper/mustard_foundationpose/foundationpose_preds"

OBJ_POSE_IDX = 1
VIEWPOINT_IDXS = range(15)

def convert_foundationpose_preds(dset_path, foundationpose_preds_path):

    output = {}

    for viewpoint_idx in VIEWPOINT_IDXS:
        with open(os.path.join(foundationpose_preds_path, f"{viewpoint_idx+1:06d}.txt"), "r") as f:
            pred = np.loadtxt(f)

        with open(os.path.join(dset_path, f"{OBJ_POSE_IDX:03d}", "tmf.json"), "r") as f:
            tmf_data = json.load(f)
        tmf_pose = np.array(tmf_data[viewpoint_idx]["pose"])

        cam_pose = tmf_pose @ TMF_TO_REALSENSE_TF

        pred_global = cam_pose @ pred

        pred_translation = pred_global[:3, 3]
        pred_rot_6d = matrix_to_rotation_6d(torch.from_numpy(pred_global[:3, :3])[None, ...])[0]

        print("viewpoint_idx", viewpoint_idx)
        print("pred_translation", pred_translation)
        print("pred_rot_6d", pred_rot_6d)

        output[f"{viewpoint_idx+1:06d}"] = {
            "pred_translation": pred_translation.tolist(),
            "pred_rot_6d": pred_rot_6d.tolist(),
            "viewpoint_idx": viewpoint_idx,
            "obj_pose_idx": OBJ_POSE_IDX,
            "dset_path": dset_path,
            "foundationpose_preds_path": foundationpose_preds_path,
        }

        with open(FOUNDATIONPOSE_PREDS_PATH + "_converted.json", "w") as f:
            json.dump(output, f, indent=4)


def matrix_to_rotation_6d(matrix: torch.Tensor) -> torch.Tensor:
    """
    Converts rotation matrices to 6D rotation representation by Zhou et al. [1]
    by dropping the last row. Note that 6D representation is not unique.
    Args:
        matrix: batch of rotation matrices of size (*, 3, 3)

    Returns:
        6D rotation representation, of size (*, 6)

    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks.
    IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    Retrieved from http://arxiv.org/abs/1812.07035
    """
    batch_dim = matrix.size()[:-2]
    return matrix[..., :2, :].clone().reshape(batch_dim + (6,))


if __name__ == "__main__":
    convert_foundationpose_preds(DSET_PATH, FOUNDATIONPOSE_PREDS_PATH)