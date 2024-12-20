import argparse
import json
import logging
import os
from datetime import datetime

import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from manotorch.manolayer import ManoLayer
from pose_estimation.data_loaders.sim_data_loader import SimDataset
from pose_estimation.model import MANOEstimator
from pose_estimation.utils.utils import axis_angle_to_6d, rot_6d_to_axis_angle
from pose_estimation.utils.eval_utils import compute_f_score, eval_pose
from pose_estimation.vis_results import vis_results as vis_results_fn
from eval_real_data import guess_avg_baseline

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device: {device}")


# Evaluation
def eval(checkpoint_path, eval_dataset_path, output_dir, vis_results=True):

    logging.basicConfig(
        filename=f"{output_dir}/evalutating_log.log",
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        filemode="w",
    )

    batch_size = 8
    trainset = SimDataset(eval_dataset_path, "train")
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    testset = SimDataset(eval_dataset_path, "test")
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=True)

    # Set up model for evaluation
    model = MANOEstimator(device=device, num_cameras=8, rot_type="6d")
    criterion = nn.SmoothL1Loss()
    model.load_state_dict(torch.load(checkpoint_path))
    model.to(device)
    model.eval()

    mano_layer = ManoLayer(mano_assets_root="SPAD-Hand-Sim/data", flat_hand_mean=False).cuda()
    l1_metrics = nn.L1Loss()

    # raw_data will hold the raw predictions and ground truth for each sample
    raw_data = []
    results = {}

    with torch.no_grad():
        for data_loader, split in zip([test_loader], ["test"]): # TODO add train loader back

            # Initialize metrics
            mpvpe_sum, mpjpe_sum, pa_mpvpe_sum, pa_mpjpe_sum = 0.0, 0.0, 0.0, 0.0
            f5mm_sum, f15mm_sum = 0.0, 0.0
            total_count = 0

            for x, y in tqdm(data_loader, total=len(data_loader), desc=f"Evaluating {split}"):
                x = torch.tensor(x).float().to(device)
                y = torch.tensor(y).float().to(device)

                # use indices to downsaple cameras (optional)
                # indices = torch.tensor([0,2,4,6,8,10,12,14])
                # x, h = x[:,indices,:], h[:,indices,:]

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
        vis_results_fn(output_dir, test_only=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate a pose estimation model on simulated data"
    )
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

    assert opts["test_data_type"] == "sim", "Data type must be sim for this script"

    # Create output directory and copy yaml tile to it
    output_dir = f"hand_pose_estimation/results/eval/{opts_fname}"
    os.makedirs(output_dir, exist_ok=True)
    with open(f"{output_dir}/opts.yaml", "w") as f:
        yaml.dump(opts, f)

    eval(opts["checkpoint_path"], opts["eval_dataset_path"], output_dir)
