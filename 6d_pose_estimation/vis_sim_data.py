"""
Visualize simulated data compared to real data of a mesh in a given 6D pose.
"""

import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import trimesh

from spad_mesh.sim.model import MeshHist

TEMP_SCENE_MESH_PATH = "data/TEMP_scene_mesh.npz"
TEMP_SENSOR_POSES_PATH = "data/TEMP_sensor_poses.npz"
CAMERA_TF = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]])


def vis_sim_data(real_data_path):
    # load scene mesh from obj file and save in the .npz format expected by MeshHist
    scene_mesh = trimesh.load(os.path.join(real_data_path, "gt", "plane_with_object.obj"))
    np.savez(
        "data/TEMP_scene_mesh.npz",
        vertices=scene_mesh.vertices,
        faces=scene_mesh.faces,
        face_normals=scene_mesh.face_normals,
        vert_normals=scene_mesh.vertex_normals,
    )

    # load sensor positions from json file and save in the .npz format expected by MeshHist
    with open(os.path.join(real_data_path, "tmf.json")) as f:
        tmf_data = json.load(f)

    poses_homog = np.array([measurement["pose"] for measurement in tmf_data])

    rotations = []
    translations = []
    for pose in poses_homog:
        rotation = pose[:3, :3] @ CAMERA_TF
        rotations.append(rotation)
        translations.append(-pose[:3, 3] @ rotation)

    rotations = np.array(rotations)
    translations = np.array(translations)

    np.savez(
        TEMP_SENSOR_POSES_PATH,
        rotations=rotations,
        translations=translations,
        camera_ids=np.arange(len(poses_homog)),
    )

    # create forward model for scene mesh + camera positions
    forward_model = MeshHist(
        camera_config=TEMP_SENSOR_POSES_PATH,
        mesh_path=TEMP_SCENE_MESH_PATH,
        with_bin_scaling=False,
    )

    rendered_hists = (
        forward_model(None, None, "6d_pose_estimation/results/test.png").detach().cpu().numpy()
    )

    fig, ax = plt.subplots(int(forward_model.num_cameras / 2 + 0.5), 2)
    ax = ax.flatten()
    for i in range(forward_model.num_cameras):
        ax[i].plot(rendered_hists[i], label="sim_" + str(forward_model.camera_ids[i]))
        # plt.plot(
        #     real_hists[i].detach().cpu().numpy(), label="real_" + str(layer.camera_ids[i])
        # )
        plt.legend()

    plt.show()


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
    vis_sim_data(args.real_data)
