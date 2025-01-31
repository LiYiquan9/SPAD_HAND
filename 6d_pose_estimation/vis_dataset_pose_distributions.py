"""
Visualize the distribution of poses between multiple 6D capture datasets

This is useful to check whether differing results on two similar datasets could be because one
is out of distribution with the training set, etc.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

DATASET_PATHS = [
    "data/real_data/6d_pose/colored_twos/matte_white_refined_gt",
    "data/real_data/6d_pose/two_16_poses_refined_gt",
]

SIM_DATASET_PATHS = [
    "data/sim_data/6d_pose/two_50000/simulated_data.npz"
]


def main():

    poses = []

    for dataset_path in DATASET_PATHS:
        for subdir in os.listdir(dataset_path):
            if not os.path.isdir(os.path.join(dataset_path, subdir)):
                continue
            pose = np.load(os.path.join(dataset_path, subdir, "gt", "gt_pose.npy"))
            poses.append(
                {
                    "simulated": False,
                    "dataset": dataset_path,
                    "x": pose[0, 3],
                    "y": pose[1, 3],
                    "z": pose[2, 3],
                }
            )

    for dataset_path in SIM_DATASET_PATHS:
        sim_poses = np.load(dataset_path)['object_poses']
        # subsample the poses and extract the x, y, z
        for pose in sim_poses[::100]:
            poses.append(
                {
                    "simulated": True,
                    "dataset": dataset_path,
                    "x": pose[0, 3],
                    "y": pose[1, 3],
                    "z": pose[2, 3],
                }
            )

    df = pd.DataFrame(poses)

    # Separate real and simulated datasets
    real_df = df[df['simulated'] != True]
    sim_df = df[df['simulated'] == True]

    fig, ax = plt.subplots()

    # Plot simulated dataset with reduced opacity
    sns.scatterplot(data=sim_df, x="x", y="y", ax=ax, alpha=0.3, color="green")

    # Plot real dataset
    sns.scatterplot(data=real_df, x="x", y="y", hue="dataset", ax=ax)

    # Set axes equal
    ax.set_aspect('equal', 'box')
    plt.show()


if __name__ == "__main__":
    main()
