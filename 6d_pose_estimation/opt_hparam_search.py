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
    test_path = os.path.join(
        "6d_pose_estimation/results/opt_hparam_search/eval", START_TIME, f"{instance_idx:03d}"
    )
    os.makedirs(test_path, exist_ok=True)

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
        training_path=test_opts["training_path"],
        num_samples_to_vis=0,
        do_optimize=True,
        # hyperparameters to search over
        opt_params={
            "translation_lr": config["translation_lr"],
            "rotation_lr": config["rotation_lr"],
            "albedo_obj_lr": config["albedo_obj_lr"],
            "opt_steps": config["opt_steps"],
            "use_lowest": config["use_lowest"],
            "method": config["method"],
            "num_runs": config["num_runs"],
        }
        # subsample_n_test_samples=5, # just to speed up testing the script
    )

    instance_idx += 1

    with open(os.path.join(test_path, "summary_metrics.json"), "rb") as f:
        file_contents = json.load(f)
        return file_contents


def main():
    wandb.init(project="spad_6d_pose_estimation", dir="data")
    results = objective(wandb.config)
    if test_opts["symmetric_object"]:
        wandb.log(
            {
                "AUC-ADD-S": results["supervised_and_optimize"]["AUC-ADD-S"],
                "mean_ADD-S": results["supervised_and_optimize"]["ADD-S"]["mean"],
            }
        )
    else:
        wandb.log(
            {
                "AUC-ADD-S": results["supervised_and_optimize"]["AUC-ADD-S"],
                "mean_ADD-S": results["supervised_and_optimize"]["ADD-S"]["mean"],
                "AUC-ADD": results["supervised_and_optimize"]["AUC-ADD"],
                "mean_ADD": results["supervised_and_optimize"]["ADD"]["mean"],
            }
        )


parser = argparse.ArgumentParser()
parser.add_argument(
    "--test_opts",
    type=str,
    help="Path to YAML file containing testing options",
    required=True,
)
args = parser.parse_args()

# Open and read in opts YAML file
with open(args.test_opts, "r") as f:
    test_opts = yaml.safe_load(f)

sweep_parameters = {
    "translation_lr": {"values": [1e-4, 1e-3, 1e-2, 1e-1]},
    "rotation_lr": {"values": [1e-4, 1e-3, 1e-2, 1e-1]},
    "albedo_obj_lr": {"values": [1e-4, 1e-3, 1e-2, 1e-1]},
    "opt_steps": {"values": [50, 100, 200]},
    "use_lowest": {"values": [True, False]},
    "method": {"values": ['fixed', 'random_start', 'random_lr', 'random_start_random_lr', 'flips']},
    "num_runs": {"values": [1, 4, 8]}
}

sweep_configuration = {
    "method": "random",
    "metric": {
        "goal": "maximize",
        "name": "AUC-ADD-S" if test_opts["symmetric_object"] else "AUC-ADD",
    },
    "parameters": sweep_parameters,
}

# 3: Start the sweep
sweep_id = wandb.sweep(sweep=sweep_configuration, project="spad_6d_pose_estimation")

wandb.agent(sweep_id, function=main, count=1000)
