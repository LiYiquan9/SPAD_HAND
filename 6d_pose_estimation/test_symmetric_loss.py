"""
Test manually defined symmetric loss function
"""

import copy

import numpy as np
import open3d as o3d
import torch
import yaml
from gen_sim_dataset import generate_random_poses
from symmetric_loss import SYMMETRIES, symmetric_loss
from hand_pose_estimation.utils.utils import (matrix_to_rotation_6d,
                                              rotation_6d_to_matrix)

opt_file = "opts/6d_pose_estimation/train/pt_sim_mustard_50000_15cam.yaml"

def main():

    # read in opt file
    with open(opt_file, "r") as f:
        opts = yaml.safe_load(f)

    """
    Visualize object symmetries
    """
    # open mesh with open3d
    mesh = o3d.io.read_triangle_mesh(opts["obj_path"])
    mesh.compute_vertex_normals()

    # display mesh at ground truth pose along with coordinate frame
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
    vis.add_geometry(mesh)
    vis.add_geometry(coord_frame)
    vis.run()

    # display original mesh and symmetric meshes all overlaid on one another
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    mesh.paint_uniform_color([0, 1, 0]) # make original mesh green
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
    vis.add_geometry(mesh)
    vis.add_geometry(coord_frame)
    for sym_tf in SYMMETRIES[opts["mesh_name"]]:
        mesh_copy = copy.deepcopy(mesh)
        mesh_copy.transform(sym_tf)
        mesh_copy.paint_uniform_color([1, 0, 0]) # make symmetric mesh red
        vis.add_geometry(mesh_copy)
    vis.run()

    """
    Test symmetric loss function by taking a random transform and testing the loss of it and the
    symmetric transform - both should be 0
    """
    # generate a random pose that we'll use as if it's the ground truth label
    gt_pose = torch.from_numpy(generate_random_poses(
        np.array([0.0, 0.0, 0.0]),
        1,
        {
            "x": [-1, 1],
            "y": [-1, 1],
            "z": [-1, 1],
        },
    )[0]).float()

    # generate all the symmetric poses that should equivalently have 0 loss
    sym_poses = []
    for sym_tf in SYMMETRIES[opts["mesh_name"]]:
        sym_pose = gt_pose @ sym_tf
        sym_poses.append(sym_pose)

    # for each symmetric pose, visualize along with g.t. pose and calculate the loss
    for sym_pose in sym_poses:

        # for debugging
        print(f"GT pose: {gt_pose}")
        print(f"Symmetric pose: {sym_pose}")

        # calculate loss
        gt_6d_pose = matrix_to_rotation_6d(gt_pose[:3, :3])
        gt_trans = gt_pose[:3, 3]

        sym_6d_pose = matrix_to_rotation_6d(sym_pose[:3, :3])
        sym_trans = sym_pose[:3, 3]

        gt_loss = symmetric_loss(
            gt_6d_pose[None, :],
            gt_6d_pose[None, :],
            gt_trans[None, :],
            gt_trans[None, :],
            opts["mesh_name"]
        )

        sym_loss = symmetric_loss(
            sym_6d_pose[None, :],
            gt_6d_pose[None, :],
            sym_trans[None, :],
            gt_trans[None, :],
            opts["mesh_name"]
        )

        print(f"Loss for GT pose: {gt_loss}")
        print(f"Loss for symmetric pose: {sym_loss}")
        
        # visualize two poses to make sure they overlap
        vis = o3d.visualization.Visualizer()
        vis.create_window()
        gt_mesh = copy.deepcopy(mesh)
        gt_mesh.transform(gt_pose)
        sym_mesh = copy.deepcopy(mesh)
        sym_mesh.transform(sym_pose)
        gt_mesh.paint_uniform_color([0, 1, 0])
        sym_mesh.paint_uniform_color([1, 0, 0])
        vis.add_geometry(gt_mesh)
        vis.add_geometry(sym_mesh)
        vis.add_geometry(coord_frame)
        vis.run()


    """
    Test symmetric loss by overlaying two random rotations and printing the loss
    """

if __name__ == "__main__":
    main()