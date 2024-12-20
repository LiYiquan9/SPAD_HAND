import argparse
import json
import logging
import os
from datetime import datetime

import numpy as np
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader

from manotorch.manolayer import ManoLayer
from pose_estimation.data_loaders.real_data_loader import RealGTDataset
from pose_estimation.model import MANOEstimator
from pose_estimation.utils.eval_utils import compute_f_score, eval_pose
from pose_estimation.utils.utils import axis_angle_to_6d, rot_6d_to_axis_angle
from pose_estimation.vis_results import vis_results

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device: {device}")

# Load in mean and std for normalization
mean_std_data = np.load("data/mean_std_data.npz")
mean_loaded = mean_std_data["mean"]
std_loaded = mean_std_data["std"]
MEAN = torch.from_numpy(mean_loaded).cuda()
STD = torch.from_numpy(std_loaded).cuda()


def normalize_hists(hists):
    return (hists - MEAN) / (STD + 3e-9)


# Evaluation
def eval_real_data(
    checkpoint_path,
    eval_dataset_paths,
    output_dir,
    rot_type,
    vis_results=True,
):

    assert rot_type == "6d", "Only 6d rotation type is supported"

    logging.basicConfig(
        filename=f"{output_dir}/evaluation_log.log",
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        filemode="w",
    )

    # Set up model
    model = MANOEstimator(device=device, num_cameras=8, rot_type=rot_type)
    criterion = nn.SmoothL1Loss()
    l1_metrics = nn.L1Loss()
    mano_layer = ManoLayer(mano_assets_root="SPAD-Hand-Sim/data", flat_hand_mean=False).cuda()
    batch_size = 16
    model.load_state_dict(torch.load(checkpoint_path))
    model.to(device)
    model.eval()

    # Set up dataset

    # the below if is a temporary hack TODO support this through opt file
    if eval_dataset_paths == ["data/real_data/captures/2024-12-02_carter_50"]:
        trainset = RealGTDataset(["data/real_data/captures/2024-10-23_carter_250"], "train")
    else:
        trainset = RealGTDataset(eval_dataset_paths, "train")
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    testset = RealGTDataset(eval_dataset_paths, "test")
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False)

    # raw_data will hold the raw predictions and ground truth for each sample
    raw_data = []
    results = {}

    with torch.no_grad():
        for data_loader, split in zip([test_loader, train_loader], ["test", "train"]):

            # Initialize metrics
            mpvpe_sum, mpjpe_sum, pa_mpvpe_sum, pa_mpjpe_sum = 0.0, 0.0, 0.0, 0.0
            f5mm_sum, f15mm_sum = 0.0, 0.0
            total_count = 0

            for x, y in data_loader:

                batch_size = x.size(0)
                x = torch.tensor(x).float().to(device)
                x = normalize_hists(x).float()

                y = torch.tensor(y).float().to(device)
                y_pose = y[..., :45]
                y_shape = y[..., 45:55]
                y_trans = y[..., 55:58]
                y_rot = y[..., 58:61]

                # Forward pass
                outputs = model(x)

                y_pose_6d = axis_angle_to_6d(y_pose.reshape(-1, 3)).reshape(y.shape[0], 90)
                y_rot_6d = axis_angle_to_6d(y_rot.reshape(-1, 3)).reshape(y.shape[0], 6)
                outputs_pose_6d = outputs[..., :90]
                outputs_pose = rot_6d_to_axis_angle(outputs_pose_6d.reshape(-1, 6)).reshape(
                    outputs.shape[0], 45
                )
                outputs_shape = outputs[..., 90:100]
                outputs_trans = outputs[..., 100:103]
                outputs_rot_6d = outputs[..., 103:109]
                outputs_rot = rot_6d_to_axis_angle(outputs_rot_6d.reshape(-1, 6)).reshape(
                    outputs.shape[0], 3
                )

                # pose_pred = torch.cat([outputs_rot, outputs_pose], dim=1)
                # vertices_pred = (mano_layer(pose_pred, outputs_shape).verts + outputs_trans.unsqueeze(1)).reshape(outputs_pose.shape[0],-1)
                # joint_pred = (mano_layer(pose_pred, outputs_shape).joints + outputs_trans.unsqueeze(1)).reshape(outputs_pose.shape[0],-1)
                pose_pred = torch.cat([y_rot, outputs_pose], dim=1)
                vertices_pred = (
                    mano_layer(pose_pred, y_shape).verts + y_trans.unsqueeze(1)
                ).reshape(outputs_pose.shape[0], -1)
                joint_pred = (mano_layer(pose_pred, y_shape).joints + y_trans.unsqueeze(1)).reshape(
                    outputs_pose.shape[0], -1
                )

                pose_gt = torch.cat([y_rot, y_pose], dim=1)
                vertices_gt = (mano_layer(pose_gt, y_shape).verts + y_trans.unsqueeze(1)).reshape(
                    y_pose.shape[0], -1
                )
                joint_gt = (mano_layer(pose_gt, y_shape).joints + y_trans.unsqueeze(1)).reshape(
                    y_pose.shape[0], -1
                )

                pose_6d_loss = criterion(outputs_pose_6d, y_pose_6d)
                rot_6d_loss = criterion(outputs_rot_6d, y_rot_6d)

                # pose_loss = criterion(outputs_pose, y_pose)
                # shape_loss = criterion(outputs_shape, y_shape)
                # trans_loss = criterion(outputs_trans, y_trans)
                # rot_loss = criterion(outputs_rot, y_rot)
                # vertices_loss = criterion(vertices_pred*1000, vertices_gt*1000)
                # joint_loss = criterion(joint_gt*1000, joint_pred*1000)

                MPVPE, PA_MPVPE = eval_pose(
                    (vertices_pred).reshape(-1, 778, 3), (vertices_gt).reshape(-1, 778, 3)
                )
                MPJPE, PA_MPJPE = eval_pose(
                    (joint_pred).reshape(-1, 21, 3), (joint_gt).reshape(-1, 21, 3)
                )

                f_scores = compute_f_score(
                    vertices_pred.reshape(-1, 778, 3), vertices_gt.reshape(-1, 778, 3)
                )
                logging.info(
                    f"metrics: pose:{l1_metrics(outputs_pose, y_pose):.3f} shape:{l1_metrics(outputs_shape, y_shape):.3f} trans:{l1_metrics(outputs_trans, y_trans):.3f} rot:{l1_metrics(outputs_rot, y_rot):.3f} vertices:{torch.norm((vertices_pred*1000).reshape(-1, 778, 3) - (vertices_gt*1000).reshape(-1, 778, 3), dim=2).mean():.3f} joint:{torch.norm((joint_pred*1000).reshape(-1, 21, 3) - (joint_gt*1000).reshape(-1, 21, 3), dim=2).mean():.3f}"
                )

                logging.info(
                    f"formal metrics: MPVPE:{MPVPE.mean():.3f} MPJPE:{MPJPE.mean():.3f} PA_MPVPE:{PA_MPVPE.mean():.3f} PA_MPJPE:{PA_MPJPE.mean():.3f} F@5mm: {f_scores['F@5mm']:.3f}, F@15mm: {f_scores['F@15mm']:.3f}"
                )

                mpvpe_sum += MPVPE.sum()
                mpjpe_sum += MPJPE.sum()
                pa_mpvpe_sum += PA_MPVPE.sum()
                pa_mpjpe_sum += PA_MPJPE.sum()
                f5mm_sum += f_scores["F@5mm"] * batch_size
                f15mm_sum += f_scores["F@15mm"] * batch_size
                total_count += batch_size

                for k in range(outputs_pose.shape[0]):
                    data = {
                        "split": split,
                        "prediction_pose": outputs_pose[k].tolist(),
                        "target_pose": y_pose[k].tolist(),
                        "prediction_shape": outputs_shape[k].tolist(),
                        "target_shape": y_shape[k].tolist(),
                        "prediction_trans": outputs_trans[k].tolist(),
                        "target_trans": y_trans[k].tolist(),
                        "prediction_rot": outputs_rot[k].tolist(),
                        "target_rot": y_rot[k].tolist(),
                        "dataset": "eval",
                    }
                    raw_data.append(data)

            # if the split is empty, don't compute avg metrics
            if total_count > 0:
                avg_mpvpe = mpvpe_sum / total_count
                avg_mpjpe = mpjpe_sum / total_count
                avg_pa_mpvpe = pa_mpvpe_sum / total_count
                avg_pa_mpjpe = pa_mpjpe_sum / total_count
                avg_f5mm = f5mm_sum / total_count
                avg_f15mm = f15mm_sum / total_count

                # Log average metrics
                logging.info(
                    f"[{split}] Total metrics: MPVPE:{avg_mpvpe:.3f} MPJPE:{avg_mpjpe:.3f} PA_MPVPE:{avg_pa_mpvpe:.3f} PA_MPJPE:{avg_pa_mpjpe:.3f} F@5mm: {avg_f5mm:.3f}, F@15mm: {avg_f15mm:.3f}"
                )

                results[split] = {
                    "MPVPE": float(avg_mpvpe),
                    "MPJPE": float(avg_mpjpe),
                    "PA_MPVPE": float(avg_pa_mpvpe),
                    "PA_MPJPE": float(avg_pa_mpjpe),
                    "F@5mm": float(avg_f5mm),
                    "F@15mm": float(avg_f15mm),
                }

    # As a baseline, guess the average of the train set for each sample, and evaluate performance
    baseline_metrics, avg_output = guess_avg_baseline(trainset, testset)
    results["guess_avg_baseline_on_test"] = baseline_metrics

    # add the result of guessing the average of the training data to the raw model outputs
    for data in raw_data:
        for k, v in avg_output.items():
            data[k] = v

    # Save results to JSON
    with open(f"{output_dir}/results.json", "w") as f:
        json.dump(results, f, indent=4)

    with open(f"{output_dir}/model_output_data.json", "w") as f:
        json.dump(raw_data, f, indent=4)

    if vis_results:
        vis_results(output_dir, test_only=True)


def guess_avg_baseline(trainset: RealGTDataset, testset: RealGTDataset) -> dict:
    """
    Calculate baseline results (mpvpe, etc) by guessing the average of the train set for each sample
    in the test set

    Args:
        trainset (RealGTDataset): training dataset
        testset (RealGTDataset): test dataset

    Returns:
        dict: dictionary of metrics (mpvpe, mpjpe, pa_mpvpe, pa_mpjpe, f@5mm, f@15mm)
    """

    batch_size = 16
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False)
    mano_layer = ManoLayer(mano_assets_root="SPAD-Hand-Sim/data", flat_hand_mean=False).cuda()

    # calculate the average of
    avg_train_y = torch.tensor(trainset.labels).mean(dim=0).to(device)
    metrics = {
        "mpvpe": [],
        "mpjpe": [],
        "pa_mpvpe": [],
        "pa_mpjpe": [],
        "f@5mm": [],
        "f@15mm": [],
    }

    for _, y in test_loader:

        # in y, idxs 0-44 are pose_aa, 45-54 are shape, 55-57 are global translation,
        # 58-60 are global rotation
        # extract the various components from the ground truth labels
        y = torch.tensor(y).float().to(device)  # (batch_size, 61)
        y_pose = y[..., :45]
        y_shape = y[..., 45:55]
        y_trans = y[..., 55:58]
        y_rot = y[..., 58:61]

        # get vertices and joints of the ground truth
        pose_gt = torch.cat([y_rot, y_pose], dim=1)  # (batch_size, 48)
        vertices_gt = (mano_layer(pose_gt, y_shape).verts + y_trans.unsqueeze(1)).reshape(
            y_pose.shape[0], -1
        )  # (batch_size, 2334)
        joint_gt = (mano_layer(pose_gt, y_shape).joints + y_trans.unsqueeze(1)).reshape(
            y_pose.shape[0], -1
        )  # (batch_size, 63)

        # get vertices and joints from the average train set
        outputs_pose = avg_train_y[:45].unsqueeze(0).repeat(y_pose.shape[0], 1).float().to(device)
        pose_pred = torch.cat([y_rot, outputs_pose], dim=1)
        vertices_pred = (mano_layer(pose_pred, y_shape).verts + y_trans.unsqueeze(1)).reshape(
            outputs_pose.shape[0], -1
        )
        joint_pred = (mano_layer(pose_pred, y_shape).joints + y_trans.unsqueeze(1)).reshape(
            outputs_pose.shape[0], -1
        )

        mpvpe, pa_mpvpe = eval_pose(
            (vertices_pred).reshape(-1, 778, 3), (vertices_gt).reshape(-1, 778, 3)
        )
        mpjpe, pa_mpjpe = eval_pose((joint_pred).reshape(-1, 21, 3), (joint_gt).reshape(-1, 21, 3))
        f_scores = compute_f_score(
            vertices_pred.reshape(-1, 778, 3), vertices_gt.reshape(-1, 778, 3)
        )

        metrics["mpvpe"].append(mpvpe)
        metrics["mpjpe"].append(mpjpe)
        metrics["pa_mpvpe"].append(pa_mpvpe)
        metrics["pa_mpjpe"].append(pa_mpjpe)
        metrics["f@5mm"].append(f_scores["F@5mm"])
        metrics["f@15mm"].append(f_scores["F@15mm"])

        avg_output = {
            "avg_train_pose": outputs_pose[0].tolist(),
            "avg_train_shape": avg_train_y[45:55].tolist(),
            "avg_train_trans": avg_train_y[55:58].tolist(),
            "avg_train_rot": avg_train_y[58:61].tolist(),
        }

        avg_metrics = {}
        for key, value in metrics.items():
            avg_metrics[key] = float(np.mean(value))

        return avg_metrics, avg_output


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Evaluate a pose estimation model on real data")
    parser.add_argument(
        "-o",
        "--opts",
        type=str,
        help="Path to YAML file containing evaluation options",
        required=True,
    )
    args = parser.parse_args()

    # The opts filename will be used as the directory name for output
    opts_fname = os.path.basename(args.opts).split(".")[0]

    # Open and read in opts YAML file
    with open(args.opts, "r") as f:
        opts = yaml.safe_load(f)

    assert opts["test_data_type"] == "real", "Data type must be real for this script"

    # Create output directory and copy yaml tile to it
    output_dir = f"hand_pose_estimation/results/eval/{opts_fname}"
    os.makedirs(output_dir, exist_ok=True)
    with open(f"{output_dir}/opts.yaml", "w") as f:
        yaml.dump(opts, f)

    eval_real_data(opts["checkpoint_path"], opts["eval_dataset_paths"], output_dir, rot_type="6d")
