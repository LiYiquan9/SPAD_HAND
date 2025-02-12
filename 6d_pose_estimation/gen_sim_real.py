"""
Visualize simulated data compared to real data of a mesh in a given 6D pose.
"""

import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import trimesh

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from spad_mesh.sim.model import MeshHist
from util import convert_json_to_meshhist_pose_format

TEMP_SCENE_MESH_PATH = "data/TEMP_scene_mesh.npz"
TEMP_SENSOR_POSES_PATH = "data/TEMP_sensor_poses.npz"


def vis_sim_data(real_data_path):
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
        forward_model(None, None, "results/verify_data/test_cam_view.png").detach().cpu().numpy()
    )

    rendered_hists = np.roll(rendered_hists, shift=1, axis=1)
    hists_noise = np.random.randn(*rendered_hists.shape) * 0.00001
    rendered_hists += hists_noise
    
    # get real histograms from real data
    real_hists = np.array([np.array(measurement["hists"]) for measurement in tmf_data])
    # pool zones in real data
    real_hists = real_hists.sum(axis=1)
    real_hists = real_hists * 0.00000035
    real_hists = real_hists - 0.0006 # background light
    
    np.save(os.path.join(real_data_path, "tmf_sim.json"), rendered_hists)
    
    # fig, ax = plt.subplots(int(forward_model.num_cameras / 2 + 0.5), 2)
    # ax = ax.flatten()
    # for i in range(forward_model.num_cameras):
    #     ax[i].plot(rendered_hists[i], label="sim_" + str(forward_model.camera_ids[i]))
    #     ax[i].plot(real_hists[i], label="real_" + str(forward_model.camera_ids[i]))
    #     plt.legend()

    # # plt.show()
    # plt.savefig("results/verify_data/test_hists_gen.png", dpi=300, bbox_inches="tight")


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

    args = parser.parse_args()
    for i in range(1,26):
        real_path = os.path.join(args.real_data, f"{i:03d}")
        vis_sim_data(real_path)
