"""
Optimize 6D pose from a random noised pose
"""

import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import trimesh
import torch
import torch.nn.functional as F
from tqdm import trange

import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from spad_mesh.sim.model import MeshHist
from util import convert_json_to_meshhist_pose_format
from hand_pose_estimation.utils.utils import matrix_to_rotation_6d, rotation_6d_to_matrix


def optimize_pose(
    real_data_path: str,
    obj_path: str,
    rot_noise: float = 0.3,
    trans_noise: float = 0.05,
    opt_steps: int = 200,
) -> None:

    scene_mesh = trimesh.load(os.path.join(real_data_path, "gt", "plane_with_object.obj"))

    plane_mesh = trimesh.load(os.path.join(real_data_path, "gt", "plane.obj"))
    
    plane_mesh = plane_mesh.subdivide_loop(10)
    
    transformed_object_mesh = trimesh.load(os.path.join(real_data_path, "gt", "object.obj"))

    albedo_obj = torch.tensor([1.0]).float().cuda()
    albedo_bg = torch.tensor([1.15]).float().cuda()
    
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
        # mesh_info={
        #     "vertices": scene_mesh.vertices,
        #     "faces": scene_mesh.faces,
        #     "face_normals": scene_mesh.face_normals,
        #     "vert_normals": scene_mesh.vertex_normals,
        # },
        mesh_info={
            "vertices": transformed_object_mesh.vertices,
            "faces": transformed_object_mesh.faces,
            "face_normals": transformed_object_mesh.face_normals,
            "vert_normals": transformed_object_mesh.vertex_normals,
        },
        background_mesh={
            "vertices": plane_mesh.vertices,
            "faces": plane_mesh.faces,
            "face_normals": plane_mesh.face_normals,
            "vert_normals": plane_mesh.vertex_normals,
        },
        with_bin_scaling=False,
    )

    rendered_hists = (
        forward_model(None, None, "results/verify_optimization/gt_sim.png", albedo_obj, albedo_bg).detach().cpu().numpy()
    )

    # get real histograms from real data
    real_hists = np.array([np.array(measurement["hists"]) for measurement in tmf_data])
    # pool zones in real data
    real_hists = real_hists.sum(axis=1)
    real_hists = real_hists * 0.00000035 - 0.0006  # background light subtraction
    real_hists = torch.from_numpy(real_hists).float().cuda()

    rendered_hists = np.roll(rendered_hists, shift=1, axis=1)
    rendered_hists = torch.from_numpy(rendered_hists).float().cuda()

    # generate sim data to be optimized
    # plane_mesh = trimesh.load(f"{real_data_path}/gt/plane.obj")

    object_mesh = trimesh.load(obj_path)

    gt_pose = np.load(os.path.join(real_data_path, "gt", "gt_pose.npy"))
    gt_translation = torch.from_numpy(gt_pose[:3, 3]).float().cuda()

    gt_rotation = (
        matrix_to_rotation_6d(torch.from_numpy(gt_pose[:3, :3])).resize(2, 3).float().cuda()
    )
    rotation = gt_rotation + torch.randn_like(gt_rotation) * rot_noise 
    rotation = F.normalize(rotation.view(-1), dim=0).view(2, 3)

    translation = (
        torch.from_numpy(gt_pose[:3, 3] + np.random.randn(*gt_pose[:3, 3].shape) * trans_noise)
        .float()
        .cuda()
    )

    init_rotation = rotation.clone()
    init_translation = translation.clone()


    layer = MeshHist(
        camera_config={
            "rotations": cam_rotations,
            "translations": cam_translations,
            "camera_ids": np.arange(len(poses_homog)),
        },
        mesh_info={
            "vertices": object_mesh.vertices,
            "faces": object_mesh.faces,
            "face_normals": object_mesh.face_normals,
            "vert_normals": object_mesh.vertex_normals,
        },
        background_mesh={
            "vertices": plane_mesh.vertices,
            "faces": plane_mesh.faces,
            "face_normals": plane_mesh.face_normals,
            "vert_normals": plane_mesh.vertex_normals,
        },
        with_bin_scaling=False,
    )

    
    hists_before = (
        layer(rotation, translation, "results/verify_optimization/before.png", albedo_obj, albedo_bg)
        .detach()
        .cpu()
        .numpy()
    )
    hists_before = np.roll(hists_before, shift=1, axis=1)

    translation.requires_grad = True
    rotation.requires_grad = True
    albedo_obj.requires_grad = True
    # albedo_bg.requires_grad = True
    
    optimizer = torch.optim.Adam(
        [
            {"params": translation, "lr": 5e-1},
            {"params": rotation, "lr": 5e-1},
            {"params": albedo_obj, "lr": 1e-1},
        ]
    )
    losses = []
    include_hist_idxs = [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15]
    
    for i in trange(opt_steps):
        hists = layer(rotation, translation, None, albedo_obj, albedo_bg)
        hists = torch.roll(hists, shifts=1, dims=1)
        
        hists_loss = hists[include_hist_idxs, :]
        real_hists_loss = real_hists[include_hist_idxs, :]

        loss = torch.nn.MSELoss()(hists_loss, real_hists_loss)
        print(loss.item())
        losses.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    hists_after = (
        layer(rotation, translation, "results/verify_optimization/after.png",albedo_obj,albedo_bg).detach().cpu().numpy()
    )
    hists_after = np.roll(hists_after, shift=1, axis=1)

    print("opt rotation: ", rotation)
    print("gt rotation: ", gt_rotation)
    print("init rotation: ", init_rotation)
    print("opt translation: ", translation)
    print("gt translation: ", gt_translation)
    print("init translation: ", init_translation)
    print("gt albedo: ", albedo_obj)
    print("init albedo: ", 1.0)

    # Helper function to apply transformations
    def apply_transformation(mesh, rotation, translation):
        rotation = rotation_6d_to_matrix(rotation.resize(6))
        rotation_np = rotation.detach().cpu().numpy()
        translation_np = translation.detach().cpu().numpy()
        transformed_vertices = np.dot(mesh.vertices, rotation_np.T) + translation_np
        transformed_mesh = mesh.copy()
        transformed_mesh.vertices = transformed_vertices
        return transformed_mesh

    # Apply transformations
    opt_mesh = apply_transformation(object_mesh, rotation, translation)
    gt_mesh = apply_transformation(object_mesh, gt_rotation, gt_translation)
    init_mesh = apply_transformation(object_mesh, init_rotation, init_translation)

    # Assign colors to the meshes
    init_mesh.visual.vertex_colors = [255, 0, 0, 255]  # Red
    opt_mesh.visual.vertex_colors = [0, 0, 255, 255]  # Blue
    gt_mesh.visual.vertex_colors = [0, 255, 0, 255]  # Green

    # Combine the meshes
    combined_mesh = trimesh.util.concatenate([init_mesh, opt_mesh, gt_mesh])

    # Save the combined mesh as an OBJ file
    combined_mesh.export("results/verify_optimization/3_mesh.obj")


    # print(loss.item())
    # plt.plot(losses)
    plt.savefig("results/verify_optimization/losses.png")
    plt.cla()

    fig, ax = plt.subplots(int(layer.num_cameras / 2 + 0.5), 2)
    ax = ax.flatten()
    for i in range(layer.num_cameras):
        ax[i].plot(hists_before[i], label="before")
        # ax[i].plot(
        #     rendered_hists[i].detach().cpu().numpy(), label="gt_sim"
        # )
        ax[i].plot(real_hists[i].detach().cpu().numpy(), label="gt_real")
        ax[i].plot(hists_after[i], label="after")
        plt.legend()

    # plt.show()
    plt.savefig("results/verify_optimization/optimize_comparison.png", dpi=900, bbox_inches="tight")


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
        "-op",
        "--opject_path",
        type=str,
        required=True,
        help="Path to object file.",
    )

    args = parser.parse_args()
    optimize_pose(args.real_data, args.opject_path)
