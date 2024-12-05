"""
Ablate the size of the fine-tuning (real world) dataset, and look at the effect on the model's
performance.
"""

import json
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import yaml

from pose_estimation.eval_real_data import eval_real_data
from pose_estimation.finetune_real_data import finetune_real_data
from pose_estimation.model import MANOEstimator

PRETRAIN_MODEL_PATH = "pose_estimation/results/train/pt_sim/model_final.pth"
OUTPUT_DIR = "pose_estimation/results/ablate_ft_dset_size"
FT_DSET_PATH = "data/real_data/captures/2024-10-23_carter_250"
FT_DSET_SIZES = [10, 25, 50]


def ablate_ft_dset_size() -> None:

    # create a subdirectory in the output_dir to store the results of this run
    pt_opt_name = PRETRAIN_MODEL_PATH.split("/")[-2]
    ft_dset_name = FT_DSET_PATH.split("/")[-1]
    output_subdir = os.path.join(OUTPUT_DIR, f"{pt_opt_name}_{ft_dset_name}")
    os.makedirs(output_subdir, exist_ok=True)

    # generate all the fine-tuned models with various fine-tuning dataset sizes
    # for ft_dset_size in FT_DSET_SIZES:
    #     this_run_output_dir = os.path.join(output_subdir, "train", f"{ft_dset_size}")
    #     os.makedirs(this_run_output_dir, exist_ok=True)

    #     finetune_real_data(
    #         dset_paths=[FT_DSET_PATH],
    #         checkpoint_path=PRETRAIN_MODEL_PATH,
    #         output_dir=this_run_output_dir,
    #         epochs=500,
    #         batch_size=16,
    #         num_cameras=8,
    #         save_model_interval=10e10,  # only save the final model
    #         rot_type="6d",
    #     )

    # run evaluation on all the models
    for ft_dset_size in FT_DSET_SIZES:
        output_dir = os.path.join(output_subdir, "eval", f"{ft_dset_size}")
        os.makedirs(output_dir, exist_ok=True)
        eval_real_data(
            checkpoint_path=os.path.join(
                output_subdir, "train", f"{ft_dset_size}", "model_final.pth"
            ),
            eval_dataset_paths=[FT_DSET_PATH],
            output_dir=output_dir,
            rot_type="6d",
            vis_results=False,
        )

    # read in all the evaluation results and visualize them
    all_results = {
        "ft_dset_size": [],
        "MPVPE": [],
        "MPJPE": [],
        "PA_MPVPE": [],
        "PA_MPJPE": [],
        "F@5mm": [],
        "F@15mm": [],
    }
    for ft_dset_size in FT_DSET_SIZES:
        with open(os.path.join(output_subdir, "eval", f"{ft_dset_size}", "results.json"), "r") as f:
            results = json.load(f)

        all_results["ft_dset_size"].append(ft_dset_size)
        for k, v in results["test"].items():
            all_results[k].append(v)

    all_results = pd.DataFrame(all_results)
    sns.lineplot(data=all_results, x="ft_dset_size", y="PA_MPJPE")

    plt.show()

if __name__ == "__main__":
    ablate_ft_dset_size()
