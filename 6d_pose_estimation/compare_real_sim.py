"""
Visualize simulated data compared to real data of a mesh in a given 6D pose.
"""

import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import trimesh
from util import convert_json_to_meshhist_pose_format
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from spad_mesh.sim.model import MeshHist
from gen_sim_dataset import sim_data_adjustment
from scipy.spatial import Delaunay

TEMP_SCENE_MESH_PATH = "data/TEMP_scene_mesh.npz"
TEMP_SENSOR_POSES_PATH = "data/TEMP_sensor_poses.npz"


def vis_sim_data(real_data_path: str, show: bool = False):
    # load scene mesh from obj file and save in the .npz format expected by MeshHist
    # scene_mesh = trimesh.load(os.path.join(real_data_path, "gt", "plane_with_object.obj"))
    
    plane_mesh = trimesh.load(os.path.join(real_data_path, "gt", "plane.obj"))
    
    # plane_mesh = plane_mesh.subdivide()
    plane_mesh = plane_mesh.subdivide_loop(10)

    scene_mesh = plane_mesh
    
    # plane_mesh = plane_mesh.subdivide()
    
    # new_plane_points = trimesh.sample.sample_surface_even(plane_mesh, 50000)[0]
    # tri = Delaunay(new_plane_points[:, :2])
    # plane_mesh = trimesh.Trimesh(vertices=new_plane_points, faces=tri.simplices) 

    # object_mesh = trimesh.load(os.path.join(real_data_path, "gt", "object.obj"))

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
        # mesh_info={
        #     "vertices": object_mesh.vertices,
        #     "faces": object_mesh.faces,
        #     "face_normals": object_mesh.face_normals,
        #     "vert_normals": object_mesh.vertex_normals,
        # },
        # background_mesh={
        #     "vertices": plane_mesh.vertices,
        #     "faces": plane_mesh.faces,
        #     "face_normals": plane_mesh.face_normals,
        #     "vert_normals": plane_mesh.vertex_normals,
        # },
        with_bin_scaling=False,
        albedo_obj=1.15,
    )

    rendered_hists = (
        forward_model(None, None, "6d_pose_estimation/results/test.png").detach().cpu().numpy()
    )
    rendered_hists = sim_data_adjustment(rendered_hists)
    
    # get real histograms from real data
    real_hists = np.array([np.array(measurement["hists"]) for measurement in tmf_data])
    # pool zones in real data
    real_hists = real_hists.sum(axis=1)
    real_hists = real_hists * 0.00000035 - 0.0006

    subplot_rows = int(forward_model.num_cameras / 2 + 0.5)
    subplot_cols = 2

    fig, ax = plt.subplots(subplot_rows, 2, figsize=(subplot_cols*4, subplot_rows*1.5))
    ax = ax.flatten()
    for i in range(forward_model.num_cameras):
        ax[i].plot(rendered_hists[i], label="sim_" + str(forward_model.camera_ids[i]))
        ax[i].plot(real_hists[i], label="real_" + str(forward_model.camera_ids[i]))
        ax[i].set_title(f"View {i:06d}")
        plt.legend()

    fig.tight_layout()

    if show:
        plt.show()

    fig.savefig(os.path.join(real_data_path, "sim_real_comparison.png"))


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

    args = parser.parse_args()
    vis_sim_data(args.real_data, args.show)
