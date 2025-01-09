"""
Visualize simulated data compared to real data of a mesh in a given 6D pose.
"""

import argparse
import json
import os
import yaml

import matplotlib.pyplot as plt
import numpy as np
import trimesh
from util import convert_json_to_meshhist_pose_format, get_dc_offset
import torch

from spad_mesh.sim.model import MeshHist

TEMP_SCENE_MESH_PATH = "data/TEMP_scene_mesh.npz"
TEMP_SENSOR_POSES_PATH = "data/TEMP_sensor_poses.npz"


def vis_sim_data(real_data_path: str, show: bool = False, trained_model_path: str = None):
    # load scene mesh from obj file and save in the .npz format expected by MeshHist
    scene_mesh = trimesh.load(os.path.join(real_data_path, "gt", "plane_with_object.obj"))

    # load sensor positions from json file and save in the .npz format expected by MeshHist
    with open(os.path.join(real_data_path, "tmf.json")) as f:
        tmf_data = json.load(f)

    poses_homog = np.array([measurement["pose"] for measurement in tmf_data])

    cam_rotations, cam_translations = convert_json_to_meshhist_pose_format(poses_homog)

    # create forward model for scene mesh + camera positions
    forward_model = MeshHist(
        camera_config={
            "rotations": cam_rotations,
            "translations": cam_translations,
            "camera_ids": np.arange(len(poses_homog)),
        },
        mesh_info={
            "vertices": scene_mesh.vertices,
            "faces": scene_mesh.faces,
            "face_normals": scene_mesh.face_normals,
            "vert_normals": scene_mesh.vertex_normals,
        },
        with_bin_scaling=False,
    )

    rendered_hists = (
        forward_model(None, None, "6d_pose_estimation/results/test.png").detach().cpu().numpy()
    )

    # get real histograms from real data
    real_hists = np.array([np.array(measurement["hists"]) for measurement in tmf_data])
    # pool zones in real data
    real_hists = real_hists.sum(axis=1)
    # TODO this scaling factor was found by trial and error just to visualize the plots in a similar
    # scale. We should find the true scaling factor or have a more systematic way to do this
    real_hists_scaled = real_hists / 4000000

    plot_and_save_hists(
        real_hists_scaled,
        rendered_hists,
        os.path.join(real_data_path, "sim_real_comparison.png"),
        forward_model.camera_ids,
        show=show,
    )

    if trained_model_path is not None:
        results = np.load(f"{trained_model_path}/mean_std_data.npz")
        hist_means = np.array(results["mean"])
        hist_stds = np.array(results["std"])

        with open(f"{trained_model_path}/opts.yaml", "r") as f:
            opts = yaml.safe_load(f)
        noise_level = opts["noise_level"]
        noise_level = 0.00001

        # add some random noise, like would be done before feeding the training hists to the network
        noise = torch.randn(rendered_hists.shape).numpy() * noise_level
        rendered_hists = rendered_hists + noise

        def apply_standardization(hists):
            output = (hists - hist_means[None, :, :]) / hist_stds[None, :, :]
            output[np.isnan(output)] = 0
            return output

        # real_hists_scaled[real_hists_scaled < get_dc_offset(real_hists_scaled) + 0.0005] = 0

        real_hists_scaled = real_hists_scaled - get_dc_offset(real_hists_scaled)

        # set the peak of both hists to 1
        real_hists_scaled = real_hists_scaled / real_hists_scaled.max(axis=1)[:, None]
        rendered_hists = rendered_hists / rendered_hists.max(axis=1)[:, None]

        plot_and_save_hists(
            real_hists_scaled,
            rendered_hists,
            os.path.join(real_data_path, "sim_real_coimparison_no_standardization.png"),
            forward_model.camera_ids,
            show=show,
        )

        real_hists_scaled = apply_standardization(real_hists_scaled)[0]

        rendered_hists = apply_standardization(rendered_hists)[0]

        plot_and_save_hists(
            real_hists_scaled,
            rendered_hists,
            os.path.join(real_data_path, "sim_real_comparison_fully_preprocessed.png"),
            forward_model.camera_ids,
            show=show,
        )


def plot_and_save_hists(real_hists, rendered_hists, save_path, camera_ids, show=False):
    num_cameras = real_hists.shape[0]
    subplot_rows = int(num_cameras / 2 + 0.5)
    subplot_cols = 2

    fig, ax = plt.subplots(subplot_rows, 2, figsize=(subplot_cols * 4, subplot_rows * 1.5))
    ax = ax.flatten()
    for i in range(num_cameras):
        ax[i].plot(rendered_hists[i], label="sim_" + str(camera_ids[i]))
        ax[i].plot(real_hists[i], label="real_" + str(camera_ids[i]))
        ax[i].set_title(f"View {camera_ids[i]}")
        plt.legend()

    fig.tight_layout()

    if show:
        plt.show()

    fig.savefig(save_path)


def homog_inv(tf: np.ndarray) -> np.ndarray:
    """
    Given some 4x4 homogenous transform, return its inverse
    """
    rot_inv = np.linalg.inv(tf[:3, :3])
    new_t = (-rot_inv) @ np.array([*tf[:3, 3]])
    return np.array(
        [[*rot_inv[0], new_t[0]], [*rot_inv[1], new_t[1]], [*rot_inv[2], new_t[2]], [0, 0, 0, 1]]
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-rd",
        "--real_data",
        type=str,
        required=True,
        help="Path to real data. Should be folder with 'gt' and 'realsense' subfolders.",
    )
    parser.add_argument(
        "-s",
        "--show",
        action="store_true",
        help="If included, show the interactive plots",
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        required=False,
        help="Path to folder containing a trained model. If passed, the parameters of the trained model are used to standardize and normalize the data",
    )

    args = parser.parse_args()
    vis_sim_data(args.real_data, args.show, trained_model_path=args.model)
