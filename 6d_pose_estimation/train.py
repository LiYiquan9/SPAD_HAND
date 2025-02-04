"""
Train a 6D pose estimation model.
"""

import argparse
import datetime
import json
import logging
import os
import sys

import numpy as np
import torch
import trimesh
import yaml
from data_loader import PoseEstimation6DDataset
from model import PoseEstimation6DModel
from torch import nn, optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from tqdm import tqdm
from util import knn_one_point

import wandb

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from hand_pose_estimation.utils.utils import matrix_to_rotation_6d, rotation_6d_to_matrix

wandb.init(project="spad_6d_pose_estimation", name="spad_6d_pose_estimator_training", dir="data")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

start_time = datetime.datetime.now()


def train(
    dset_path: str,
    dset_type: str,
    output_dir: str,
    epochs: int,
    batch_size: int,
    save_model_interval: int,
    noise_level: float = 0.00,
    include_hist_idxs: list | str = "all",
    test_interval: int = 1,
    rot_type: str = "6d",
    obj_path: str = "",
    check_real: bool = False,
    real_path: str = "",
    loss_type: str = "rot_trans_pcd",
) -> None:
    """
    Train a 6D pose estimation model (works on real or simulated data)

    Args:
        dset_path (str): Path to the dataset
        dset_type (str): Type of the dataset. Must be "real" or "sim"
        output_dir (str): Directory to save model checkpoints and logs
        epochs (int): Number of epochs to train for
        batch_size (int): Batch size
        save_model_interval (int): How often to save model checkpoints
        noise_level (float): Standard deviation of noise to add to input histograms
        test_interval (int): How often to test the model
        rot_type (str): Type of rotation representation to use
        obj_path (str): Path to the object mesh
        check_real (bool): Whether to check the model on real data during training
        real_path (str): Path to the real data, if check_real is True
        loss (str): Type of loss to use

    Returns:
        None
    """

    logging.basicConfig(
        filename=f"{output_dir}/training_log.log",
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        filemode="w",
    )

    # load obj points
    obj_mesh = trimesh.load(obj_path)
    obj_points = obj_mesh.sample(1000)
    obj_points = torch.tensor(obj_points, dtype=torch.float32).to(device)

    # load dataset
    train_dataset = PoseEstimation6DDataset(
        dset_path, dset_type=dset_type, split="train", include_hist_idxs=include_hist_idxs
    )
    test_dataset = PoseEstimation6DDataset(
        dset_path, dset_type=dset_type, split="test", include_hist_idxs=include_hist_idxs
    )

    trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    testloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    if check_real:
        real_test_dataset = PoseEstimation6DDataset(
            real_path,
            dset_type="real",
            split="test",
            test_portion=1.0,
            include_hist_idxs=include_hist_idxs,
        )
        real_testloader = DataLoader(real_test_dataset, batch_size=batch_size, shuffle=True)

    print(f"Training dataset has {len(train_dataset)} samples")
    print(f"Testing dataset has {len(test_dataset)} samples")

    def self_norm(hists):
        per_hist_mean = hists.mean(dim=-1, keepdim=True)
        per_hist_std = hists.std(dim=-1, keepdim=True)
        return (hists - per_hist_mean) / (per_hist_std + 3e-9)

    epoch_data = []

    # load model
    model = PoseEstimation6DModel(device=device, num_cameras=train_dataset.histograms.shape[1]).to(
        device
    )
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=5000, gamma=0.8)
    l1_loss = nn.L1Loss()
    l2_loss = nn.MSELoss()

    # train model
    total_batches = epochs * (len(trainloader) + len(testloader) // test_interval)
    with tqdm(total=total_batches, desc="Total Progress", unit="batch") as pbar:
        for epoch in range(epochs):
            model.train()

            for batch_idx, (hists, labels) in enumerate(trainloader):
                noise = torch.randn(hists.size()).to(device) * noise_level

                hists = torch.tensor(hists).float().to(device)
                hists = hists + noise

                hists = self_norm(hists).float()

                labels = torch.tensor(labels).float().to(device)

                optimizer.zero_grad()
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

                wandb.log(
                    {
                        "train_rot_loss": rot_loss.item(),
                        "train_trans_loss": trans_loss.item(),
                        "train_pc_loss": pc_loss.item(),
                        "train_total_loss": loss.item(),
                    }
                )

                loss.backward()
                optimizer.step()

                data = {
                    "prediction_rot_6d": outputs_rot_6d[0].tolist(),
                    "target_rot_6d": labels_rot_6d[0].tolist(),
                    "prediction_trans": outputs_trans[0].tolist(),
                    "target_trans": labels_trans[0].tolist(),
                    "epoch": epoch,
                    "batch": batch_idx,
                    "dataset": "train",
                }
                epoch_data.append(data)

                # Update progress bar
                pbar.update(1)
                pbar.set_postfix(
                    epoch=epoch + 1,
                    train_loss=loss.item(),
                    rot_loss=rot_loss.item(),
                    pc_loss=pc_loss.item(),
                    trans_loss=trans_loss.item(),
                )

            scheduler.step()

            if (epoch + 1) % test_interval == 0:
                model.eval()

                for batch_idx, (hists, labels) in enumerate(testloader):
                    hists = torch.tensor(hists).float().to(device)
                    hists = self_norm(hists).float()
                    # hists = normalize_hists(hists).float()
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
                        outputs_obj_pc = torch.matmul(
                            obj_points, outputs_rot_matrix.transpose(1, 2)
                        )
                    else:
                        raise Exception("support only 6d rotation type")

                    rot_loss = l1_loss(outputs_rot_6d, labels_rot_6d)
                    # rot_loss = torch.mean(torch.norm(outputs_rot_matrix - labels_rot_matrix, dim=1))
                    trans_loss = l1_loss(outputs_trans, labels_trans)
                    pc_loss = l2_loss(outputs_obj_pc, labels_obj_pc)

                    loss = 0.5 * rot_loss + 0.5 * trans_loss + 0.1 * pc_loss

                    wandb.log(
                        {
                            "test_rot_loss": rot_loss.item(),
                            "test_trans_loss": trans_loss.item(),
                            "test_pc_loss": pc_loss.item(),
                            "test_total_loss": loss.item(),
                        }
                    )

                    data = {
                        "prediction_rot_6d": outputs_rot_6d[0].tolist(),
                        "target_rot_6d": labels_rot_6d[0].tolist(),
                        "prediction_trans": outputs_trans[0].tolist(),
                        "target_trans": labels_trans[0].tolist(),
                        "epoch": epoch,
                        "batch": batch_idx,
                        "dataset": "test",
                    }
                    epoch_data.append(data)

                    # Update progress bar
                    pbar.update(1)
                    pbar.set_postfix(
                        epoch=epoch + 1,
                        test_loss=loss.item(),
                        rot_loss=rot_loss.item(),
                        pc_loss=pc_loss.item(),
                        trans_loss=trans_loss.item(),
                    )

                    if check_real:
                        for batch_idx, (hists, labels, filenames) in enumerate(real_testloader):
                            hists = torch.tensor(hists).float().to(device)
                            hists = self_norm(hists).float()
                            # hists = normalize_hists(hists).float()
                            labels = torch.tensor(labels).float().to(device)

                            outputs = model(hists)

                            if rot_type == "6d":
                                labels_rot_6d = matrix_to_rotation_6d(labels[:, :3, :3])
                                labels_trans = labels[:, :3, 3].reshape(labels.size(0), 3)

                                outputs_rot_6d = outputs[:, :6]
                                outputs_trans = outputs[:, 6:9]

                                labels_rot_matrix = labels[:, :3, :3]
                                outputs_rot_matrix = rotation_6d_to_matrix(outputs[:, :6])

                                labels_obj_pc = torch.matmul(
                                    obj_points, labels_rot_matrix.transpose(1, 2)
                                )
                                outputs_obj_pc = torch.matmul(
                                    obj_points, outputs_rot_matrix.transpose(1, 2)
                                )
                            else:
                                raise Exception("support only 6d rotation type")

                            if loss_type == "rot_trans_pcd":
                                rot_loss = l1_loss(outputs_rot_6d, labels_rot_6d)
                                # rot_loss = torch.mean(torch.norm(outputs_rot_matrix - labels_rot_matrix, dim=1))
                                trans_loss = l1_loss(outputs_trans, labels_trans)
                                pc_loss = l2_loss(outputs_obj_pc, labels_obj_pc)

                                loss = 0.5 * rot_loss + 0.5 * trans_loss + 0.1 * pc_loss

                            elif loss_type == "ADD_S":
                                loss = torch.norm(
                                    calculate_ADD_S(outputs_obj_pc, labels_obj_pc), p=1
                                )  # l1

                            else:
                                raise ValueError(f"Invalid loss type: {loss_type}")

                            wandb.log(
                                {
                                    "real_test_rot_loss": rot_loss.item(),
                                    "real_test_trans_loss": trans_loss.item(),
                                    "real_test_pc_loss": pc_loss.item(),
                                    "real_test_total_loss": loss.item(),
                                }
                            )

                            for k in range(outputs_rot_6d.size()[0]):
                                data = {
                                    "prediction_rot_6d": outputs_rot_6d[k].tolist(),
                                    "target_rot_6d": labels_rot_6d[k].tolist(),
                                    "prediction_trans": outputs_trans[k].tolist(),
                                    "target_trans": labels_trans[k].tolist(),
                                    "filename": filenames[k],
                                    "epoch": epoch,
                                    "batch": batch_idx,
                                    "dataset": "real_test",
                                }
                                epoch_data.append(data)
                            epoch_data.append(data)

                            # Update progress bar
                            pbar.update(1)
                            pbar.set_postfix(
                                epoch=epoch + 1,
                                test_loss=loss.item(),
                                rot_loss=rot_loss.item(),
                                pc_loss=pc_loss.item(),
                                trans_loss=trans_loss.item(),
                            )

            if (epoch + 1) % save_model_interval == 0:
                model_save_path = f"{output_dir}/model_{epoch}.pth"
                torch.save(model.state_dict(), model_save_path)
                logging.info(f"Model state dictionary saved to {model_save_path}")

                with open(f"{output_dir}/model_output_data_{epoch}.json", "w") as f:
                    json.dump(epoch_data, f, indent=4)

    # save model
    model_save_path = f"{output_dir}/model_final.pth"
    torch.save(model.state_dict(), model_save_path)
    logging.info(f"Final model state dictionary saved to {model_save_path}")
    with open(f"{output_dir}/model_output_data_final.json", "w") as f:
        json.dump(epoch_data, f, indent=4)


def calculate_ADD_S(pred, gt_xyz):
    # E6SD expects a different dim order, so it's easiest to just re-order here and keep the rest
    # of the fn the same
    gt_xyz = gt_xyz.permute(0, 2, 1)
    pred = pred.permute(0, 2, 1)

    # below is unchanged from E6SD
    num_valid, _, num_points = gt_xyz.size()
    # inputs should be (batch, n, 3) and (batch, m, 3)
    inds = knn_one_point(pred.permute(0, 2, 1), gt_xyz.permute(0, 2, 1))  # num_valid, num_points
    inds = inds.view(num_valid, 1, num_points).repeat(1, 3, 1)
    tar_tmp = torch.gather(gt_xyz, 2, inds)
    add_ij = torch.mean(torch.norm(pred - tar_tmp, dim=1), dim=1)  # [nv]

    return add_ij


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a pose estimation model")
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
    output_dir = f"6d_pose_estimation/results/train/{opts_fname}"
    os.makedirs(output_dir, exist_ok=True)
    with open(f"{output_dir}/opts.yaml", "w") as f:
        yaml.dump(opts, f)

    train(
        opts["dset_path"],
        opts["dset_type"],
        output_dir,
        epochs=opts["epochs"],
        batch_size=opts["batch_size"],
        save_model_interval=opts["save_model_interval"],
        rot_type=opts["rot_type"],
        obj_path=opts["obj_path"],
        check_real=opts["check_real"],
        real_path=opts["real_path"],
        include_hist_idxs=opts["include_hist_idxs"],
        loss_type=opts["loss_type"],
    )
