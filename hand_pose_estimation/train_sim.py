import argparse
import datetime
import json
import logging
import os

import numpy as np
import torch
import wandb
import yaml
from torch import nn, optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader

from manotorch.manolayer import ManoLayer
from pose_estimation.data_loaders.sim_data_loader import SimDataset
from pose_estimation.model import MANOEstimator
from pose_estimation.utils.utils import axis_angle_to_6d, rot_6d_to_axis_angle

wandb.init(
    project="spad_hand_pose_estimation", name="spad_hand_pose_estimator_training", dir="data"
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device: {device}")


def train_sim(
    dset_path,
    output_dir,
    epochs,
    batch_size,
    num_cameras,
    save_model_interval,
    rot_type="aa",
):

    start_time = datetime.datetime.now()

    logging.basicConfig(
        filename=f"{output_dir}/training_log.log",
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        filemode="w",
    )

    # Set up dataset
    trainset = SimDataset(dset_path, "train")
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    testset = SimDataset(dset_path, "test")
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=True)
    all_hists_train = trainset.x_data

    # Set up model
    model = MANOEstimator(device=device, num_cameras=num_cameras, rot_type="6d")
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)
    criterion = nn.SmoothL1Loss()
    l1_metrics = nn.L1Loss()
    mano_layer = ManoLayer(mano_assets_root="SPAD-Hand-Sim/data/", flat_hand_mean=False).cuda()
    model.to(device)

    # get mean and std for normalization
    MEAN = torch.tensor([all_hists_train.mean()]).cuda()
    STD = torch.tensor([all_hists_train.std()]).cuda()
    mean_np = MEAN.cpu().numpy()
    std_np = STD.cpu().numpy()
    # Save MEAN and STD in an .npz file
    np.savez("data/mean_std_data.npz", mean=mean_np, std=std_np)

    def normalize_hists(hists):
        return (hists - MEAN) / (STD + 3e-9)

    epoch_data = []

    for epoch in range(epochs):
        model.train()
        for batch_idx, (x, y) in enumerate(trainloader):
            noise = torch.randn(x.size()).to(device) * 0.00

            x = torch.tensor(x).float().to(device)

            x = x + noise
            x = normalize_hists(x).float()

            y = torch.tensor(y).float().to(device)

            y_pose = y[..., :45]
            y_shape = y[..., 45:55]
            y_trans = y[..., 55:58]
            y_rot = y[..., 58:61]

            optimizer.zero_grad()

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

            pose_loss = criterion(outputs_pose, y_pose)
            shape_loss = criterion(outputs_shape, y_shape)
            trans_loss = criterion(outputs_trans, y_trans)
            rot_loss = criterion(outputs_rot, y_rot)
            vertices_loss = criterion(vertices_pred * 1000, vertices_gt * 1000)
            joint_loss = criterion(joint_gt * 1000, joint_pred * 1000)

            loss = (
                1.0 * pose_6d_loss
                + 0.01 * shape_loss
                + 0.01 * trans_loss
                + 0.01 * rot_loss
                + 0.01 * vertices_loss
                + 0.01 * joint_loss
            )

            wandb.log(
                {
                    "train_pose_6d_loss": pose_6d_loss.item(),
                    "train_shape_loss": shape_loss.item(),
                    "train_trans_loss": trans_loss.item(),
                    "train_rot_6d_loss": rot_6d_loss.item(),
                    "train_vertices_loss": vertices_loss.item(),
                    "train_joint_loss": joint_loss.item(),
                    "train_total_loss": loss.item(),
                }
            )

            logging.info(
                f"train epoch {epoch} pose:{pose_6d_loss:.3f} shape:{shape_loss:.3f} trans:{trans_loss:.3f} rot:{rot_6d_loss:.3f} vertices:{vertices_loss:.3f} joint:{joint_loss:.3f}"
            )

            logging.info(
                f"metrics: pose:{l1_metrics(outputs_pose, y_pose):.3f} shape:{l1_metrics(outputs_shape, y_shape):.3f} trans:{l1_metrics(outputs_trans, y_trans):.3f} rot:{l1_metrics(outputs_rot, y_rot):.3f} vertices:{torch.norm((vertices_pred*1000).reshape(-1, 778, 3) - (vertices_gt*1000).reshape(-1, 778, 3), dim=2).mean():.3f} joint:{torch.norm((joint_pred*1000).reshape(-1, 21, 3) - (joint_gt*1000).reshape(-1, 21, 3), dim=2).mean():.3f}"
            )

            loss.backward()
            optimizer.step()

            # running_loss += loss.item() * x.size(0)

            data = {
                "prediction_pose": outputs_pose[0].tolist(),
                "target_pose": y_pose[0].tolist(),
                "prediction_shape": outputs_shape[0].tolist(),
                "target_shape": y_shape[0].tolist(),
                "prediction_trans": outputs_trans[0].tolist(),
                "target_trans": y_trans[0].tolist(),
                "prediction_rot": outputs_rot[0].tolist(),
                "target_rot": y_rot[0].tolist(),
                "epoch": epoch,
                "batch": batch_idx,
                "dataset": "train",
            }
            epoch_data.append(data)

        # logging.info(f"Train Epoch: {epoch}, Loss: {running_loss*1.0 / (len(trainloader)):.4f}")

        # generate an ETA for all epochs to complete
        time_elapsed = datetime.datetime.now() - start_time
        avg_time_per_epoch = time_elapsed / (epoch + 1)
        eta = avg_time_per_epoch * (epochs - epoch)
        # print the time remaining in H:M:S format
        print(f"Epoch {epoch} complete.")
        print(f"Time Remaining: {eta} ETA: {datetime.datetime.now() + eta}")

        scheduler.step()

        if epoch % 1 == 0:
            model.eval()

            for batch_idx, (x, y) in enumerate(testloader):
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

                if rot_type == "6d":
                    pose_6d_loss = criterion(outputs_pose_6d, y_pose_6d)
                    rot_6d_loss = criterion(outputs_rot_6d, y_rot_6d)
                else:
                    raise Exception("suport only 6d totation type")

                pose_loss = criterion(outputs_pose, y_pose)
                shape_loss = criterion(outputs_shape, y_shape)
                trans_loss = criterion(outputs_trans, y_trans)
                rot_loss = criterion(outputs_rot, y_rot)
                vertices_loss = criterion(vertices_pred * 1000, vertices_gt * 1000)
                joint_loss = criterion(joint_gt * 1000, joint_pred * 1000)

                logging.info(
                    f"test epoch {epoch} pose:{pose_6d_loss:.3f} shape:{shape_loss:.3f} trans:{trans_loss:.3f} rot:{rot_6d_loss:.3f} vertices:{vertices_loss:.3f} joint:{joint_loss:.3f}"
                )

                logging.info(
                    f"metrics: pose:{l1_metrics(outputs_pose, y_pose):.3f} shape:{l1_metrics(outputs_shape, y_shape):.3f} trans:{l1_metrics(outputs_trans, y_trans):.3f} rot:{l1_metrics(outputs_rot, y_rot):.3f} vertices:{torch.norm((vertices_pred*1000).reshape(-1, 778, 3) - (vertices_gt*1000).reshape(-1, 778, 3), dim=2).mean():.3f} joint:{torch.norm((joint_pred*1000).reshape(-1, 21, 3) - (joint_gt*1000).reshape(-1, 21, 3), dim=2).mean():.3f}"
                )

                data = {
                    "prediction_pose": outputs_pose[0].tolist(),
                    "target_pose": y_pose[0].tolist(),
                    "prediction_shape": outputs_shape[0].tolist(),
                    "target_shape": y_shape[0].tolist(),
                    "prediction_trans": outputs_trans[0].tolist(),
                    "target_trans": y_trans[0].tolist(),
                    "prediction_rot": outputs_rot[0].tolist(),
                    "target_rot": y_rot[0].tolist(),
                    "epoch": epoch,
                    "batch": batch_idx,
                    "dataset": "test",
                }
                epoch_data.append(data)

        if (epoch + 1) % save_model_interval == 0:
            model_save_path = f"{output_dir}/model_{epoch}.pth"
            torch.save(model.state_dict(), model_save_path)
            logging.info(f"Model state dictionary saved to {model_save_path}")

            with open(f"{output_dir}/model_output_data_{epoch}.json", "w") as f:
                json.dump(epoch_data, f, indent=4)

    # save the final model as model_final.pth
    model_save_path = f"{output_dir}/model_final.pth"
    torch.save(model.state_dict(), model_save_path)
    logging.info(f"Final model state dictionary saved to {model_save_path}")
    with open(f"{output_dir}/model_output_data_final.json", "w") as f:
        json.dump(epoch_data, f, indent=4)


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
    output_dir = f"hand_pose_estimation/results/train/{opts_fname}"
    os.makedirs(output_dir, exist_ok=True)
    with open(f"{output_dir}/opts.yaml", "w") as f:
        yaml.dump(opts, f)

    train_sim(
        opts["dset_path"],
        output_dir,
        epochs=opts["epochs"],
        batch_size=opts["batch_size"],
        num_cameras=opts["num_cameras"],
        save_model_interval=opts["save_model_interval"],
        rot_type=opts["rot_type"],
    )
