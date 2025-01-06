"""
Evaluate a 6D pose estimation model (works on real or simulated data).
"""

import argparse
import datetime
import json
import logging
import os
import sys

import yaml
from data_loader import PoseEstimation6DDataset
from model import PoseEstimation6DModel
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import torch
import trimesh
from torch import nn, optim
from torch.utils.data import DataLoader

from hand_pose_estimation.utils.utils import matrix_to_rotation_6d, rotation_6d_to_matrix

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

start_time = datetime.datetime.now()


def test(
    dset_path: str,
    dset_type: str,
    output_dir: str,
    batch_size: int,
    training_path: str,
    noise_level: float = 0.00,
    rot_type: str = "6d",
    save_opt: bool = False,
    obj_path: str = "",
) -> None:
    """
    Test a 6D pose estimation model (works on real or simulated data)

    Args:
        dset_path (str): Path to the dataset
        dset_type (str): Type of dataset, must be 'sim' or 'real'
        output_dir (str): Directory to save model checkpoints and logs
        batch_size (int): Batch size
        training_path (str): Path to the directory containing the trained model
        noise_level (float): Noise level to add to the input data
        rot_type (str): Type of rotation representation to use
        save_opt (bool): Whether to save the data for optimization
        obj_path (str): Path to the object mesh file

    Returns:
        None
    """

    logging.basicConfig(
        filename=f"{output_dir}/testing_log.log",
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        filemode="w",
    )

    # load obj points
    obj_mesh = trimesh.load(obj_path)
    obj_points = obj_mesh.sample(1000)
    obj_points = torch.tensor(obj_points, dtype=torch.float32).to(device)

    # load dataset
    test_dataset = PoseEstimation6DDataset(dset_path, dset_type=dset_type, split="test")

    testloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    print(f"Testing dataset has {len(test_dataset)} samples")

    data = np.load(f"{training_path}/mean_std_data.npz")
    MEAN = torch.tensor(data["mean"]).to(device)
    STD = torch.tensor(data["std"]).to(device)

    def normalize_hists(hists):
        return (hists - MEAN) / (STD + 3e-9)

    epoch_data = []

    # load model
    model = PoseEstimation6DModel(device=device).to(device)
    model.load_state_dict(torch.load(f"{training_path}/model_final.pth"))

    l1_loss = nn.L1Loss()
    l2_loss = nn.MSELoss()

    # evaluate model
    total_batches = 1.0 * (len(testloader))
    with tqdm(total=total_batches, desc="Total Progress", unit="batch") as pbar:

        model.eval()

        for batch_idx, loaded_data in enumerate(testloader):
            if dset_type == "sim":
                ori_hists, labels = loaded_data
            elif dset_type == "real":
                ori_hists, labels, filenames = loaded_data

            ori_hists = torch.tensor(ori_hists).float().to(device)
            hists = normalize_hists(ori_hists).float()
            labels = torch.tensor(labels).float().to(device)

            outputs = model(hists)

            if rot_type == "6d":
                labels_rot_6d = matrix_to_rotation_6d(labels[:, :3, :3])
                labels_trans = labels[:, :3, 3].reshape(labels.size(0), 3)

                outputs_rot_6d = outputs[:, :6]
                outputs_trans = outputs[:, 6:9]

                labels_rot_matrix = labels[:, :3, :3]
                outputs_rot_matrix = rotation_6d_to_matrix(outputs[:, :6])

                labels_obj_pc = torch.matmul(obj_points, labels_rot_matrix.transpose(1, 2))
                outputs_obj_pc = torch.matmul(obj_points, outputs_rot_matrix.transpose(1, 2))
            else:
                raise Exception("support only 6d rotation type")

            rot_loss = l1_loss(outputs_rot_6d, labels_rot_6d)
            # rot_loss = torch.mean(torch.norm(outputs_rot_matrix - labels_rot_matrix, dim=1))
            trans_loss = l1_loss(outputs_trans, labels_trans)
            pc_loss = l2_loss(outputs_obj_pc, labels_obj_pc)

            loss = 0.5 * rot_loss + 0.5 * trans_loss + 0.1 * pc_loss

            for k in range(outputs_rot_6d.size()[0]):
                data = {
                    "prediction_rot_6d": outputs_rot_6d[k].tolist(),
                    "target_rot_6d": labels_rot_6d[k].tolist(),
                    "prediction_trans": outputs_trans[k].tolist(),
                    "target_trans": labels_trans[k].tolist(),
                    "filename": filenames[k] if dset_type == "real" else "",
                    "batch": batch_idx,
                    "dataset": "test",
                    "dataset_type": dset_type,
                }
                if save_opt:
                    data["hists"] = ori_hists[k].tolist()

            epoch_data.append(data)

            # Update progress bar
            pbar.update(1)
            pbar.set_postfix(
                epoch=1,
                test_loss=loss.item(),
                rot_loss=rot_loss.item(),
                pc_loss=pc_loss.item(),
                trans_loss=trans_loss.item(),
            )

    # save model output data
    with open(f"{output_dir}/model_output_data_final.json", "w") as f:
        json.dump(epoch_data, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test a pose estimation model")
    parser.add_argument(
        "-o",
        "--opts",
        type=str,
        help="Path to YAML file containing training options",
        required=True,
    )
    args = parser.parse_args()

    # The opts filename will be used as the directory name for output
    opts_fname = os.path.basename(args.opts).split(".")[0]

    # Open and read in opts YAML file
    with open(args.opts, "r") as f:
        opts = yaml.safe_load(f)

    # Create output directory and copy yaml tile to it
    output_dir = f"6d_pose_estimation/results/eval/{opts_fname}"
    os.makedirs(output_dir, exist_ok=True)
    with open(f"{output_dir}/opts.yaml", "w") as f:
        yaml.dump(opts, f)

    test(
        opts["dset_path"],
        opts["dset_type"],
        output_dir,
        training_path=opts["training_path"],
        batch_size=opts["batch_size"],
        rot_type=opts["rot_type"],
        save_opt=opts["save_opt"],
        obj_path=opts["obj_path"],
    )
