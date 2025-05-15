"""
Visualize histograms and corresponding images of object poses for paper figure. Originally
created for rebuttal.
"""

import json
import os
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import PIL.Image as Image
import seaborn as sns


OUT_DIR = "pose_estimation/results/plots"

ROOT_DATA_DIR = "/home/carter/projects/SPAD_HAND/data/mustard_refined_gt"
OBJ_POSE_IDXS = [1, 14]
CAM_POSE_IDX = 10

def main():

    rgb_imgs = []
    histograms = []

    for obj_pose_idx in OBJ_POSE_IDXS:
        rgb_path = os.path.join(ROOT_DATA_DIR, f"{obj_pose_idx:03d}", "realsense", "rgb", f"{CAM_POSE_IDX:06d}.png")
        rgb_img = Image.open(rgb_path)
        # rotate image by 90 degrees
        rgb_img = rgb_img.rotate(90, expand=True)
        rgb_imgs.append(rgb_img)

        histogram_path = os.path.join(ROOT_DATA_DIR, f"{obj_pose_idx:03d}", "tmf.json")
        with open(histogram_path, "r") as f:
            hist_data = json.load(f)
        histograms.append(np.array(hist_data[CAM_POSE_IDX]["hists"]).sum(axis=0))

    """
    Create a figure for our visualization and to copy paste photos from into inkscape
    """
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].plot(histograms[0], label="Obj. Pose 1")
    ax[0].plot(histograms[1], label="Obj. Pose 2")
    ax[0].legend()
    ax[0].tick_params(axis="both", which="major")
    ax[0].tick_params(axis="both", which="minor")
    ax[0].set_yticks([])

    ax[0].set_xlabel("Bin Index")
    ax[0].set_ylabel("Count")
    ax[0].set_title("Captured Histograms from One Sensor Pose")

    ax[1].imshow(rgb_imgs[0])
    ax[2].imshow(rgb_imgs[1])
    # remove ax ticks and labels from ax[1] and ax[2]
    ax[1].set_xticks([])
    ax[1].set_yticks([])
    ax[2].set_xticks([])
    ax[2].set_yticks([])

    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "ref_hists_poses_for_figure.pdf"), bbox_inches="tight")

    """
    Set up matplotlib / seaborn so that everything looks nice
    """
    sns.set_theme(style="whitegrid")
    plt.rcParams["axes.prop_cycle"] = plt.cycler(color=plt.cm.Set2.colors)

    legend_fontsize = 6
    legend_title_fontsize = 8
    axis_tick_fontsize = 6
    axis_label_fontsize = 8
    tick_label_padding = 0
    border_thickness = 1
    linewidth = 1.2
    scatterplot_dotsize = 12
    bg_grid_linewidth = 0.8

    mpl.rcParams["font.family"] = "sans-serif"
    mpl.rcParams["font.sans-serif"] = ["CMU Sans Serif"]
    # this makes sure that we don't use type 3 fonts, which are forbidden by papercept
    mpl.rcParams["pdf.fonttype"] = 42
    mpl.rcParams["ps.fonttype"] = 42
    mpl.rcParams["axes.linewidth"] = border_thickness

    """
    Create a figure to piece together in inkscape
    """
    fig, ax = plt.subplots(1, 1, figsize=(1.8, 1.8))
    ax.plot(histograms[0], label="Obj. Pose 1", linewidth=linewidth)
    ax.plot(histograms[1], label="Obj. Pose 2", linewidth=linewidth)
    ax.legend(fontsize=legend_fontsize)
    ax.tick_params(axis="both", which="major", labelsize=axis_tick_fontsize)
    ax.tick_params(axis="both", which="minor", labelsize=axis_tick_fontsize)
    ax.set_yticks([])
    # move x axis ticks closer to the plot
    ax.tick_params(axis="x", which="major", pad=-2)

    ax.set_xlabel("Bin Index", fontsize=axis_label_fontsize)
    ax.set_ylabel("Count", fontsize=axis_label_fontsize)
    ax.set_title("Captured Histograms from One Sensor Pose", fontsize=axis_label_fontsize)

    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "hists_poses_for_figure.pdf"), bbox_inches="tight")

if __name__ == "__main__":
    main()