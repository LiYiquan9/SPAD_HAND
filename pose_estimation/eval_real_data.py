import json
import logging
import os
from datetime import datetime

import numpy as np
import torch
from data_loader_real_gt import RealGTDataset
from eval_utils import compute_f_score, eval_pose
from model import MANOEstimator
from torch import nn, optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, Dataset
from utils import axis_angle_to_6d, rot_6d_to_axis_angle

import wandb
from manotorch.manolayer import ManoLayer

wandb.init(project="spad_hand_pose_estimation", name="spad_hand_pose_estimator_training")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"device: {device}")

batch_size = 16

num_cameras = 8

trainset = RealGTDataset("train")

trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)

testset = RealGTDataset("test")

testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)

data = np.load("mean_std_data.npz")

mean_loaded = data["mean"]
std_loaded = data["std"]

MEAN = torch.from_numpy(mean_loaded).cuda()
STD = torch.from_numpy(std_loaded).cuda()


def normalize_hists(hists):

    return (hists - MEAN) / (STD + 3e-9)


# Model
mano_estimator = MANOEstimator(device=device, num_cameras=num_cameras, rot_type="6d")

optimizer = optim.Adam(mano_estimator.parameters(), lr=1e-4)

scheduler = lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)

criterion = nn.SmoothL1Loss()

l1_metrics = nn.L1Loss()

mano_layer = ManoLayer(mano_assets_root="SPAD-Hand-Sim/data", flat_hand_mean=False).cuda()

# checkpoint_path = "pose_estimation/results/train/2024-11-22_10-38-49/model_199.pth" # (carter) pre-trained on sim. data only
checkpoint_path = "pose_estimation/results/train/2024-11-22_15-30-12/model_499.pth" # (carter) pre-trained on sim. data, fine tuned on my hand training data


# Evaluation
def eval(
    model,
    trainloader,
    testloader,
    optimizer,
    criterion,
    scheduler,
    epochs,
    rot_type="aa",
    save_model_interval=25,
):

    now = datetime.now()
    utc_time = now.strftime("%Y-%m-%d_%H-%M-%S")
    os.makedirs(f"pose_estimation/results/eval/{utc_time}")

    logging.basicConfig(
        filename=f"pose_estimation/results/eval/{utc_time}/evaluation_log.log",
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        filemode="w",
    )

    model.load_state_dict(torch.load(checkpoint_path))

    model.to(device)

    epoch_data = []

    model.eval()

    mpvpe_sum, mpjpe_sum, pa_mpvpe_sum, pa_mpjpe_sum = 0.0, 0.0, 0.0, 0.0
    f5mm_sum, f15mm_sum = 0.0, 0.0
    total_count = 0

    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(testloader):

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

            if rot_type == "6d":
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
            else:
                raise Exception("suport only 6d totation type")

            # pose_pred = torch.cat([outputs_rot, outputs_pose], dim=1)
            # vertices_pred = (mano_layer(pose_pred, outputs_shape).verts + outputs_trans.unsqueeze(1)).reshape(outputs_pose.shape[0],-1)
            # joint_pred = (mano_layer(pose_pred, outputs_shape).joints + outputs_trans.unsqueeze(1)).reshape(outputs_pose.shape[0],-1)
            pose_pred = torch.cat([y_rot, outputs_pose], dim=1)
            vertices_pred = (mano_layer(pose_pred, y_shape).verts + y_trans.unsqueeze(1)).reshape(
                outputs_pose.shape[0], -1
            )
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

            if rot_type == "6d":
                pose_6d_loss = criterion(outputs_pose_6d, y_pose_6d)
                rot_6d_loss = criterion(outputs_rot_6d, y_rot_6d)
            else:
                raise Exception("suport only 6d totation type")

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
                epoch_data.append(data)

    avg_mpvpe = mpvpe_sum / total_count
    avg_mpjpe = mpjpe_sum / total_count
    avg_pa_mpvpe = pa_mpvpe_sum / total_count
    avg_pa_mpjpe = pa_mpjpe_sum / total_count
    avg_f5mm = f5mm_sum / total_count
    avg_f15mm = f15mm_sum / total_count

    # Log average metrics
    logging.info(
        f"Total metrics: MPVPE:{avg_mpvpe:.3f} MPJPE:{avg_mpjpe:.3f} PA_MPVPE:{avg_pa_mpvpe:.3f} PA_MPJPE:{avg_pa_mpjpe:.3f} F@5mm: {avg_f5mm:.3f}, F@15mm: {avg_f15mm:.3f}"
    )

    with open(f"pose_estimation/results/eval/{utc_time}/model_output_data.json", "w") as f:
        json.dump(epoch_data, f, indent=4)


if __name__ == "__main__":

    eval(
        mano_estimator,
        trainloader,
        testloader,
        optimizer,
        criterion,
        scheduler,
        epochs=200,
        rot_type="6d",
        save_model_interval=50,
    )
