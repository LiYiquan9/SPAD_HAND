"""
Use W&B Sweep to search for ideal training hyperparameters for transferring simulated data to
real world performance without fine-tuning.
"""

import argparse
import json
import os

import yaml
from eval import test
from train import train
import time

import wandb

START_TIME = time.strftime("%Y-%m-%d_%H-%M-%S")

instance_idx = 0


# objective function that we'd like to minimize
def objective(config):
    global instance_idx

    train_path = os.path.join(
        "6d_pose_estimation/results/train_hparam_search/train", START_TIME, f"{instance_idx:03d}"
    )
    os.makedirs(train_path, exist_ok=True)
    test_path = os.path.join(
        "6d_pose_estimation/results/train_hparam_search/eval", START_TIME, f"{instance_idx:03d}"
    )
    os.makedirs(test_path, exist_ok=True)

    train(
        config["dset_path"],
        config["dset_type"],
        train_path,
        config["epochs"],
        config["batch_size"],
        config["save_model_interval"],
        {
            "lr": config["lr"],
            "weight_decay": config["weight_decay"],
        },
        {
            "step_size": config["step_size"],
            "gamma": config["gamma"],
        },
        config["noise_level"],
        config["include_hist_idxs"],
        config["test_interval"],
        config["rot_type"],
        config["obj_path"],
        config["check_real"],
        config["real_path"],
        config["loss_type"],
        config["mesh_sample_count"],
        config["symmetric_object"],
        use_wandb=False,
    )

    test(
        # things taken from test opts
        dset_path=test_opts["dset_path"],
        dset_type=test_opts["dset_type"],
        batch_size=test_opts["batch_size"],
        obj_path=test_opts["obj_path"],
        test_on_split=test_opts["test_on_split"],
        include_hist_idxs=test_opts["include_hist_idxs"],
        sensor_plane_path=test_opts["sensor_plane_path"],
        # things that are fixed
        output_dir=test_path,
        training_path=os.path.join(train_path, "model_final.pth"),
        num_samples_to_vis=0,
        do_optimize=False,
        opt_params=test_opts["opt_params"],
        # subsample_n_test_samples=5, # just to speed up testing the script
    )

    instance_idx += 1

    with open(os.path.join(test_path, "summary_metrics.json"), "rb") as f:
        file_contents = json.load(f)
        return file_contents


def main():
    wandb.init(project="spad_6d_pose_estimation", dir="data")
    results = objective(wandb.config)
    if train_opts["symmetric_object"]:
        wandb.log(
            {
                "supervised_AUC-ADD-S": results["supervised_model"]["AUC-ADD-S"],
                "supervised_mean_ADD-S": results["supervised_model"]["ADD-S"]["mean"],
            }
        )
    else:
        wandb.log(
            {
                "supervised_AUC-ADD-S": results["supervised_model"]["AUC-ADD-S"],
                "supervised_mean_ADD-S": results["supervised_model"]["ADD-S"]["mean"],
                "supervised_AUC-ADD": results["supervised_model"]["AUC-ADD"],
                "supervised_mean_ADD": results["supervised_model"]["ADD"]["mean"],
            }
        )


parser = argparse.ArgumentParser()
parser.add_argument(
    "--train_opts",
    type=str,
    help="Path to YAML file containing training options",
    required=True,
)
parser.add_argument(
    "--test_opts",
    type=str,
    help="Path to YAML file containing testing options",
    required=True,
)
args = parser.parse_args()

# Open and read in opts YAML files
with open(args.train_opts, "r") as f:
    train_opts = yaml.safe_load(f)
with open(args.test_opts, "r") as f:
    test_opts = yaml.safe_load(f)

sweep_parameters = {
    # things that are taken from opts file
    "dset_path": {"value": train_opts["dset_path"]},
    "dset_type": {"value": train_opts["dset_type"]},
    "epochs": {"value": train_opts["epochs"]},
    "batch_size": {"value": train_opts["batch_size"]},
    "include_hist_idxs": {"value": train_opts["include_hist_idxs"]},
    "rot_type": {"value": train_opts["rot_type"]},
    "obj_path": {"value": train_opts["obj_path"]},
    "check_real": {"value": False},  # not needed
    "real_path": {"value": train_opts["real_path"]},  # not needed if check_real is false
    "loss_type": {"value": train_opts["loss_type"]},
    "mesh_sample_count": {"value": train_opts["mesh_sample_count"]},
    "symmetric_object": {"value": train_opts["symmetric_object"]},
    # things that are fixed
    "test_interval": {"value": 10000},  # no need to test during training
    "save_model_interval": {"value": 10000},  # no need to save intermediate models
    # things worth ablating
    "noise_level": {"values": [0.0, 0.000001, 0.00001, 0.0001]},
    "lr": {"values": [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]},
    "weight_decay": {"values": [1e-5, 1e-4, 1e-3, 1e-2]},
    "step_size": {"values": [10000, 5000, 1000]},
    "gamma": {"values": [0.9, 0.8, 0.6]},
}

sweep_configuration = {
    "method": "grid",
    "metric": {
        "goal": "maximize",
        "name": f"supervised_{'AUC-ADD-S' if train_opts['symmetric_object'] else 'AUC-ADD'}",
    },
    "parameters": sweep_parameters,
}

# 3: Start the sweep
sweep_id = wandb.sweep(sweep=sweep_configuration, project="spad_6d_pose_estimation")

wandb.agent(sweep_id, function=main, count=100)
