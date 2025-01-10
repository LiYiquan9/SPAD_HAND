"""
Use W&B Sweep to search for ideal training hyperparameters for transferring simulated data to
real world performance without fine-tuning.
"""

import json

import wandb
import yaml
from eval import test
from train import train
import numpy as np


# objective function that we'd like to minimize
def objective(config):

    # it doesn't make sense to normalize hists and normalize hist peaks at the same time, so if
    # we see this, just return an invalid value and move on
    if config["normalize_hists"] and config["normalize_hist_peaks"]:
        return np.nan

    train(
        config["dset_path"],
        config["dset_type"],
        config["output_dir"],
        config["epochs"],
        config["batch_size"],
        config["save_model_interval"],
        config["noise_level"],
        config["test_interval"],
        config["rot_type"],
        config["obj_path"],
        config["check_real"],
        config["real_path"],
        config["normalize_hists"],
        config["normalize_hist_peaks"],
        config["standardize_input"],
        config["trim_to_range"],
        config["subtract_dc_offset_from_real"],
    )

    # create a dummy opts file that contains a few variables needed by test, and place it in the
    # temp training directory
    opts_contents = {
        "standardize_input": config["standardize_input"],
        "trim_to_range": config["trim_to_range"],
        "subtract_dc_offset_from_real": config["subtract_dc_offset_from_real"],
        "normalize_hists": config["normalize_hists"],
        "normalize_hist_peaks": config["normalize_hist_peaks"],
    }
    with open(f"{config['output_dir']}/opts.yaml", "w") as f:
        yaml.dump(opts_contents, f)

    test(
        dset_path="data/real_data/6d_pose/two_16_poses",
        dset_type="real",
        output_dir="6d_pose_estimation/results/hparam_search/eval_tmp",
        batch_size=2048,
        training_path="6d_pose_estimation/results/hparam_search/train_tmp",
        obj_path="data/obj_meshes/two.stl",
        num_samples_to_vis=0,
        test_on_split="all",
    )

    with open("6d_pose_estimation/results/hparam_search/eval_tmp/summary_metrics.json", "rb") as f:
        real_data_ADD = json.load(f)["supervised_model"]["ADD"]["mean"]

    return real_data_ADD


def main():
    wandb.init(project="spad_6d_pose_estimation", dir="data")
    real_data_ADD = objective(wandb.config)
    wandb.log({"real_data_ADD": real_data_ADD})


sweep_configuration = {
    "method": "grid",
    "metric": {"goal": "minimize", "name": "real_data_ADD"},
    "parameters": {
        "dset_path": {"value": "data/sim_data/6d_pose/two_50000/simulated_data.npz"},
        "dset_type": {"value": "sim"},
        "output_dir": {"value": "6d_pose_estimation/results/hparam_search/train_tmp"},
        "epochs": {"value": 500},
        "batch_size": {"value": 2048},
        "save_model_interval": {"value": 10000},  # no need to save intermediate models
        "noise_level": {"value": 0.00001},
        "test_interval": {"value": 10000},  # no need to test during training
        "rot_type": {"value": "6d"},
        "obj_path": {"value": "data/obj_meshes/two.stl"},
        "check_real": {"value": False},  # not needed
        "real_path": {"value": None},  # not needed if check_real is false
        "normalize_hists": {"values": [True, False]},
        "normalize_hist_peaks": {"values": [True, False]},
        "standardize_input": {"values": [True, False]},
        "trim_to_range": {"value": [0, 128]},
        "subtract_dc_offset_from_real": {"values": [True, False]},
    },
}

# 3: Start the sweep
sweep_id = wandb.sweep(sweep=sweep_configuration, project="spad_6d_pose_estimation")

wandb.agent(sweep_id, function=main, count=10)
