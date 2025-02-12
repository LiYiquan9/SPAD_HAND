"""
Evaluate a 6D pose estimation model (works on real or simulated data).
"""

import argparse
import copy
import datetime
import json
import os
import sys
from typing import List, Tuple

import matplotlib.pyplot as plt
import open3d as o3d
import yaml
from data_loader import PoseEstimation6DDataset
from model import PoseEstimation6DModel
from PIL import Image
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm
from util import (convert_json_to_meshhist_pose_format, create_plane_mesh,
                  homog_inv, vis_hists)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import time
from random import gauss

import numpy as np
import torch
import trimesh
from scipy.spatial import cKDTree
from torch.utils.data import DataLoader

from hand_pose_estimation.utils.utils import (matrix_to_rotation_6d,
                                              rotation_6d_to_matrix)
from spad_mesh.sim.model import MeshHist

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

start_time = datetime.datetime.now()


def self_norm(hists):
    per_hist_mean = hists.mean(dim=-1, keepdim=True)
    per_hist_std = hists.std(dim=-1, keepdim=True)
    return (hists - per_hist_mean) / (per_hist_std + 3e-9)


def test(
    dset_path: str,
    dset_type: str,
    output_dir: str,
    batch_size: int,
    training_path: str,
    opt_params: dict,
    noise_level: float = 0.00,
    rot_type: str = "6d",
    obj_path: str = "",
    num_samples_to_vis: int = 10,
    test_on_split: bool = "test",
    sensor_plane_path: str = "",
    subsample_n_test_samples: int = 0,
    include_hist_idxs: list | str = "all",
    symmetric_object: bool = False,
    do_optimize: bool = True,
) -> None:
    """
    Test a 6D pose estimation model (works on real or simulated data)

    Args:
        dset_path (str): Path to the dataset
        dset_type (str): Type of dataset, must be 'sim' or 'real'
        output_dir (str): Directory to save model checkpoints and logs
        batch_size (int): Batch size
        training_path (str): Path to the directory containing the trained model
        opt_params (dict): Dictionary of hyperparameters for the optimization (render and compare)
        noise_level (float): Noise level to add to the input data
        rot_type (str): Type of rotation representation to use
        obj_path (str): Path to the object mesh file
        test_on_split (str): If 'test', only test on the test split of the dataset. If 'train',
            test only on the train split. If 'all', test on the entire dataset.
        sensor_plane_path (str): Path to the sensor plane mesh file
        subsample_n_test_samples (int): If > 0, only use this many test samples, the rest will be
            discarded. Useful for faster eval.
        include_hist_idxs (list, optional): If a list, only include the histograms at these indices.
        symmetric_object (bool): If True, the object is symmetric. Changes what metrics are used
            for eval.
        do_optimize (bool): If True, optimize the model output using MeshHist and report those
            results as well

    Returns:
        None
    """

    if dset_type == "sim":
        full_dset_path = os.path.join(dset_path, "simulated_data.npz")
    else:
        full_dset_path = dset_path

    # load obj points
    obj_mesh = trimesh.load(obj_path)
    obj_points = obj_mesh.sample(1000)
    obj_points = torch.tensor(obj_points, dtype=torch.float32).to(device)

    # load dataset
    test_dataset = PoseEstimation6DDataset(
        full_dset_path,
        dset_type=dset_type,
        split=test_on_split,
        subsample_n_test_samples=subsample_n_test_samples,
        include_hist_idxs=include_hist_idxs,
    )
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # calculate the average of the test dataset labels - these will be used as a baseline for
    # comparison (test_set_avg inference mode)
    test_labels = []
    for loaded_data in test_dataset:
        if dset_type == "sim":
            _, label, albedos,albedos_bg = loaded_data
        elif dset_type == "real":
            _, label, albedos,albedos_bg, _ = loaded_data

        label_ortho6d = matrix_to_rotation_6d(torch.from_numpy(label[:3, :3])[None, :])[0]
        label_translation = label[:3, 3]
        flat_label = np.concatenate([label_ortho6d, label_translation])
        test_labels.append(torch.from_numpy(flat_label))
    test_labels = torch.stack(test_labels).to(device)
    test_labels_avg = torch.mean(test_labels, dim=0)

    print(f"Testing dataset has {len(test_dataset)} samples")

    if do_optimize:
        inference_modes = ["supervised_model", "supervised_and_optimize", "test_set_avg"]
    else:
        inference_modes = ["supervised_model", "test_set_avg"]
    all_results = {inference_type: [] for inference_type in inference_modes}

    # load model
    model = PoseEstimation6DModel(device=device, num_cameras=test_dataset.histograms.shape[1]).to(
        device
    )
    model.load_state_dict(torch.load(training_path, weights_only=False))

    # evaluate model
    model.eval()

    for inference_mode in inference_modes:
        for batch_idx, loaded_data in tqdm(
            enumerate(test_loader),
            total=len(test_loader),
            desc=f"Running test data through model ({inference_mode})",
        ):
            if dset_type == "sim":
                raw_input_hists, labels, albedos = loaded_data
            elif dset_type == "real":
                raw_input_hists, labels, albedos, albedos_bg, filenames = loaded_data

            raw_input_hists = raw_input_hists.float().to(device)
            self_norm_hists = self_norm(raw_input_hists)
            norm_input_hists = self_norm_hists
            labels = labels.float().to(device)

            if inference_mode == "supervised_model":
                outputs, a_logists, b_logists = model(norm_input_hists)
            elif inference_mode == "test_set_avg":
                outputs = test_labels_avg.repeat(norm_input_hists.shape[0], 1).to(device).float()
            elif inference_mode == "supervised_and_optimize":

                assert batch_size == 1, "batch size must be 1 for supervised_and_optimize"
                assert (
                    sensor_plane_path != ""
                ), "sensor_plane_path must be provided for supervised_and_optimize"

                outputs_before_opt, a_logists, b_logists = model(norm_input_hists)

                # run optimization on the outputs - depending on opt_params["method"], multiple
                # optimizations may be run and the best chosen.
                all_outputs = []
                losses = []
                for _ in range(opt_params["num_runs"]):
                    if opt_params["method"] not in ["random_lr", "random_start", "fixed"]:
                        raise Exception(f"invalid optimization method name ({opt_params['method']})")

                    tweaked_opt_params = opt_params
                    tweaked_outputs_before_opt = outputs_before_opt

                    if "random_lr" in opt_params["method"]:
                        tweaked_opt_params = {
                            "translation_lr": opt_params["translation_lr"]
                            * np.random.uniform(0.5, 1.5),
                            "rotation_lr": opt_params["rotation_lr"] * np.random.uniform(0.5, 1.5),
                            "albedo_obj_lr": opt_params["albedo_obj_lr"]
                            * np.random.uniform(0.5, 1.5),
                            "opt_steps": opt_params["opt_steps"],
                            "use_lowest": opt_params["use_lowest"],
                        }
                    if "random_start" in opt_params["method"]:
                        tweaked_outputs_before_opt = perturb_outputs(
                            outputs_before_opt, rotation_range=np.deg2rad(10), translation_range=0.1
                        )

                    outputs, loss = optimize(
                        tweaked_outputs_before_opt,
                        raw_input_hists,
                        tweaked_opt_params,
                        sensor_plane_path,
                        obj_path,
                        include_hist_idxs,
                    )

                    all_outputs.append(outputs)
                    losses.append(loss)

                best_idx = np.argmin(losses)
                outputs = all_outputs[best_idx]

            if rot_type == "6d":
                gt_rot_6d = matrix_to_rotation_6d(labels[:, :3, :3])
                gt_translation = labels[:, :3, 3].reshape(labels.size(0), 3)

                pred_rot_6d = outputs[:, :6]
                pred_translation = outputs[:, 6:9]

                gt_rot_matrix = labels[:, :3, :3]
                pred_rot_matrix = rotation_6d_to_matrix(outputs[:, :6])

                gt_obj_pcd = torch.matmul(obj_points, gt_rot_matrix.transpose(1, 2))
                pred_obj_pcd = torch.matmul(obj_points, pred_rot_matrix.transpose(1, 2))
                
                gt_obj_pcd = gt_obj_pcd + gt_translation.unsqueeze(1)
                pred_obj_pcd = pred_obj_pcd + pred_translation.unsqueeze(1)
                

            else:
                raise Exception("support only 6d rotation type")

            for sample_idx in range(pred_rot_6d.shape[0]):
                results = {
                    "pred_rot_6d": pred_rot_6d[sample_idx].tolist(),
                    "gt_rot_6d": gt_rot_6d[sample_idx].tolist(),
                    "pred_translation": pred_translation[sample_idx].tolist(),
                    "gt_translation": gt_translation[sample_idx].tolist(),
                    "filename": filenames[sample_idx] if dset_type == "real" else "",
                    "batch": batch_idx,
                    "dataset": "test",
                    "dataset_type": dset_type,
                    # "hists": raw_input_hists[sample_idx].tolist(),
                }

                pred_metrics = get_pred_metrics(
                    pred_rot_6d[sample_idx],
                    gt_rot_6d[sample_idx],
                    pred_translation[sample_idx],
                    gt_translation[sample_idx],
                    gt_obj_pcd[sample_idx],
                    pred_obj_pcd[sample_idx],
                )

                results.update(pred_metrics)  # add metrics to data

                all_results[inference_mode].append(results)

    # calculate average of metrics and save to file
    summary_metrics = {}
    for inference_mode in inference_modes:
        summary_metrics[inference_mode] = {}
        for metric in pred_metrics.keys():
            summary_metrics[inference_mode][metric] = {}
            all_datapoints = [data[metric] for data in all_results[inference_mode]]
            summary_metrics[inference_mode][metric]["mean"] = np.mean(all_datapoints)
            summary_metrics[inference_mode][metric]["90_pct"] = np.percentile(all_datapoints, 90)
        summary_metrics[inference_mode]["AUC-ADD"], summary_metrics[inference_mode]["AUC-ADD-S"] = (
            compute_auc_add(all_results[inference_mode])
        )

    # if the object is symmetric, some of the metrics are not valid - remove them to avoid confusion
    if symmetric_object:
        for inference_mode in inference_modes:
            del summary_metrics[inference_mode]["ADD"]
            del summary_metrics[inference_mode]["rotation_error"]

    with open(f"{output_dir}/summary_metrics.json", "w") as f:
        json.dump(summary_metrics, f, indent=4)

    # save model output data
    print("Saving model predictions to json...")
    with open(f"{output_dir}/model_predictions.json", "w") as f:
        json.dump(all_results, f, indent=4)

    # generate plots and visualizations for results
    plot_metrics(
        all_results,
        os.path.join(output_dir, "plots"),
        pred_metrics.keys(),
        symmetric_object=symmetric_object,
    )
    visualize_results(
        all_results,
        os.path.join(output_dir, "visualizations"),
        obj_path,
        dset_path,
        dset_type,
        num_samples_to_vis,
    )

def perturb_outputs(outputs, rotation_range, translation_range):
    """
    Randomly perturb NN outputs - to be used before feeding into optimization.

    Args:
        outputs (torch.Tensor): Outputs from the neural network
        rotation_range (float): Maximum rotation in radians to perturb the rotation by
        translation_range (float): Maximum translation in meters to perturb the translation by

    Returns:
        torch.Tensor: Perturbed outputs
    """
    pred_rot_6d = outputs[:, :6]
    pred_translation = outputs[:, 6:9]

    # perturb rotation by choosing a random axis and angle in the rotation_range to apply to the 
    # rotation matrix
    pred_rot_matrix = rotation_6d_to_matrix(pred_rot_6d)
    axis = make_rand_vector(3)
    angle = np.random.uniform(-rotation_range, rotation_range)
    perturbation_rot_matrix = R.from_rotvec(axis * angle).as_matrix()
    perturbed_rot_matrix = pred_rot_matrix @ torch.from_numpy(perturbation_rot_matrix).float().to(device)
    perturbed_rot_6d = matrix_to_rotation_6d(perturbed_rot_matrix)

    # perturb translation by adding a random vector in the translation_range
    perturbation_translation = torch.tensor(
        [np.random.uniform(-translation_range, translation_range) for _ in range(3)]
    ).to(outputs.device)
    perturbed_translation = pred_translation + perturbation_translation

    return torch.cat([perturbed_rot_6d, perturbed_translation], dim=-1)
    

def make_rand_vector(dims):
    """
    Create a random dims-dimensional unit vector
    """
    vec = [gauss(0, 1) for i in range(dims)]
    mag = sum(x**2 for x in vec) ** .5
    return np.array([x/mag for x in vec])


def optimize(
    outputs_supervised: torch.Tensor,
    raw_input_hists: torch.Tensor,
    opt_params: dict,
    sensor_plane_path: str = None,
    obj_path: str = None,
    include_hist_idxs: list | str = "all",
):

    # load plane and object meshes
    plane_mesh = trimesh.load(f"{sensor_plane_path}/gt/plane.obj")
    object_mesh = trimesh.load(obj_path)

    # plane_mesh.vertices = [ # setting this fixes it??
    #     [-5.,        -5.,        -0.0973211,],
    #     [-5.,         5.,        -0.207898 ,],
    #     [ 5.,         5.,        -0.234209 ,],
    #     [ 5.,        -5.,        -0.123633 ,],
    # ]

    plane_mesh.vertices[0][2] += 1e-7  # this fixes it????

    # print("plane_mesh vertices", plane_mesh.vertices)
    # print("plane mesh faces", plane_mesh.faces)
    # print("plane mesh normals", plane_mesh.face_normals)
    # print("plane mesh vertex normals", plane_mesh.vertex_normals)

    """
    mattewhite2 (broken)
    plane_mesh vertices
    [[-5.       -5.       -0.131366]
    [-5.        5.       -0.210161]
    [ 5.        5.       -0.199364]
    [ 5.       -5.       -0.120569]]
    plane mesh faces
    [[0 2 1]
    [0 3 2]]
    plane mesh normals
    [[-0.00107967  0.00787925  0.99996838]
    [-0.00107967  0.00787925  0.99996838]]
    plane mesh vertex normals
    [[-0.00107967  0.00787925  0.99996838]
    [-0.00107967  0.00787925  0.99996838]
    [-0.00107967  0.00787925  0.99996838]
    [-0.00107967  0.00787925  0.99996838]]

    real (works)
    [[-5.        -5.        -0.0973211]
    [-5.         5.        -0.207898 ]
    [ 5.         5.        -0.234209 ]
    [ 5.        -5.        -0.123633 ]]
    plane mesh faces
    [[0 2 1]
    [0 3 2]]
    plane mesh normals
    [[0.00263093 0.01105698 0.99993541]
    [0.00263102 0.01105689 0.99993541]]
    plane mesh vertex normals
    [[0.00263098 0.01105693 0.99993541]
    [0.00263093 0.01105698 0.99993541]
    [0.00263098 0.01105693 0.99993541]
    [0.00263102 0.01105689 0.99993541]]
    """

    # load rotation and translation from supervised model
    rotation = outputs_supervised[0, :6].reshape(2, 3).detach().requires_grad_()
    translation = outputs_supervised[0, 6:9].detach().requires_grad_()
    # albedo_obj = outputs_supervised[0, 9].detach().requires_grad_()
    albedo_obj = 0.9
    albedo_bg = 1.15
    
    translation.requires_grad = True
    rotation.requires_grad = True
    # albedo_obj.requires_grad = True

    # load sensor positions from json file and save in the .npz format expected by MeshHist
    with open(os.path.join(sensor_plane_path, "tmf.json")) as f:
        tmf_data = json.load(f)

    poses_homog = np.array([measurement["pose"] for measurement in tmf_data])

    cam_rotations, cam_translations = convert_json_to_meshhist_pose_format(poses_homog)

    # init optimization layer
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

    optimizer = torch.optim.Adam(
        [
            # {"params": translation, "lr": 1e-3}, # 1e-3
            # {"params": rotation, "lr": 1e-2}, # 1e-2
            # {"params": albedo_obj, "lr": 5e-9}, # 5e-2

            {"params": translation, "lr": opt_params["translation_lr"]},
            {"params": rotation, "lr": opt_params["rotation_lr"]},
            # {"params": albedo_obj, "lr": opt_params["albedo_obj_lr"]},
        ]
    )
    # store losses and parameters for each iteration. If use_lowest is true, the parameters which
    # led to the lowest loss are returned. Otherwise, the final parameters are returned
    losses = []

    params = []
    for i in range(opt_params["opt_steps"]):
        rendered_hists = layer(rotation, translation, None, albedo_obj, albedo_bg)

        if include_hist_idxs != "all":
            rendered_hists = rendered_hists[include_hist_idxs, :]

        # vis_hists(
        #     rendered_hists, raw_input_hists[0, :, :], "rendered (blue) and raw input hists (orange)"
        # )

        rendered_hists = torch.roll(rendered_hists, shifts=1, dims=1)
        rendered_hists = self_norm(rendered_hists)
        norm_raw_input_hists = self_norm(raw_input_hists)
        loss = torch.nn.MSELoss()(rendered_hists, norm_raw_input_hists[0, :, :])
        # if loss.item() > 0.5:
        #     break
        losses.append(loss.item())
        params.append(torch.cat([rotation.reshape(6), translation], dim=-1)[None,].detach())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if opt_params["use_lowest"]:
        print(f"idx of lowest loss: {np.argmin(losses)}")
        return params[np.argmin(losses)], losses[np.argmin(losses)]
    else:
        return params[-1], losses[-1]


def visualize_results(
    all_results: dict,
    output_dir: str,
    obj_mesh_path: str,
    dset_path: str,
    dset_type: str,
    num_samples_to_vis: int,
) -> None:
    """
    For each inference mode, visualize the predicted and ground truth poses for each sample.

    Args:
        all_results (dict): Dictionary containing the results of the model predictions
        output_dir (str): Directory to save the visualizations
        obj_mesh (str): Path to the target mesh obj file
        dset_path (str): Path to the dataset
        dset_type (str): Type of dataset, must be 'sim' or 'real'
        num_samples_to_vis (int): Number of samples to visualize. n samples will be randomly
            selected from the dataset.
    """

    # this camera transform can be applied on the right side to convert a camera pose from the
    # tmf coordinate system to the o3d coordinate system.
    # for example: o3d_frame_pose = tmf_frame_pose @ tmf_to_o3d_cam_tf
    tmf_to_o3d_cam_tf = np.array(
        [
            [-1.0, 0.0, 0.0, 0.0],
            [0.0, -1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )

    os.makedirs(output_dir, exist_ok=True)

    inference_modes = list(all_results.keys())
    mesh = o3d.io.read_triangle_mesh(obj_mesh_path)

    # load in plane mesh and camera poses
    # how this is done depends on the dataset type
    # camera poses should be the same for all samples
    if dset_type == "sim":
        with open(os.path.join(dset_path, "metadata.json"), "r") as f:
            metadata = json.load(f)
        plane_params = metadata["plane_params"]
        plane_mesh = create_plane_mesh(**plane_params)

        camera_poses = [np.array(pose) @ tmf_to_o3d_cam_tf for pose in metadata["tmf_poses"]]

    elif dset_type == "real":
        with open(os.path.join(dset_path, "plane_registration.json"), "r") as f:
            plane_registration = json.load(f)
        plane_mesh = create_plane_mesh(
            plane_registration["plane_a"][0],
            plane_registration["plane_a"][1],
            plane_registration["plane_a"][2],
            -plane_registration["plane_d"],
        )

        with open(os.path.join(dset_path, "001", "tmf.json"), "r") as f:
            tmf_data = json.load(f)

        camera_poses = [np.array(datapoint["pose"]) @ tmf_to_o3d_cam_tf for datapoint in tmf_data]

    plane_mesh.compute_vertex_normals()

    # select sample_idxs for visualization
    if num_samples_to_vis > len(all_results[inference_modes[0]]):
        num_samples_to_vis = len(all_results[inference_modes[0]])
        print(
            f"WARNING: num_samples_to_vis ({num_samples_to_vis}) is greater than the number of samples in the test dataset ({len(all_results[inference_modes[0]])}). Visualizing all samples."
        )

    np.random.seed(0)  # for reproducibility
    sample_idxs = np.random.choice(
        len(all_results[inference_modes[0]]), num_samples_to_vis, replace=False
    )

    for inference_mode in tqdm(inference_modes, desc="Visualizing results", unit="inference mode"):
        for sample_idx in tqdm(sample_idxs, desc="Visualizing results", unit="sample", leave=False):
            result = all_results[inference_mode][sample_idx]

            pred_tf_matrix = np.eye(4)
            pred_tf_matrix[:3, :3] = rotation_6d_to_matrix(torch.tensor(result["pred_rot_6d"]))
            pred_tf_matrix[:3, 3] = torch.tensor(result["pred_translation"])

            gt_tf_matrix = np.eye(4)
            gt_tf_matrix[:3, :3] = rotation_6d_to_matrix(torch.tensor(result["gt_rot_6d"]))
            gt_tf_matrix[:3, 3] = torch.tensor(result["gt_translation"])

            # transform object mesh by predicted pose
            predicted_obj_mesh = copy.deepcopy(mesh)
            predicted_obj_mesh.transform(pred_tf_matrix)

            # transform object mesh by ground truth pose
            gt_obj_mesh = copy.deepcopy(mesh)
            gt_obj_mesh.transform(gt_tf_matrix)

            predicted_obj_mesh.compute_vertex_normals()
            gt_obj_mesh.compute_vertex_normals()

            # due to some weirdness with open3d, this render_mesh_from_viewpoints function MUST
            # be in a different function, or rendering will fail when the object or plane
            # mesh changes
            render_mesh_from_viewpoints(
                plane_mesh,
                predicted_obj_mesh,
                gt_obj_mesh,
                camera_poses,
                dset_type,
                dset_path,
                result["filename"],
                os.path.join(output_dir, inference_mode, f"{sample_idx:06d}.stl"),
            )


def render_mesh_from_viewpoints(
    plane_mesh: o3d.geometry.TriangleMesh,
    predicted_obj_mesh: o3d.geometry.TriangleMesh,
    gt_obj_mesh: o3d.geometry.TriangleMesh,
    camera_poses: list,
    dset_type: str,
    dset_path: str,
    original_filename: int,
    output_path: str,
):
    """
    Render the predicted and ground truth poses for a single sample from the dataset at a number
    of camera viewpoints. Viewpoints are combined to one .png image and saved to file.

    Args:
        plane_mesh (o3d.cpu.pybind.geometry.TriangleMesh): Open3D mesh object representing the
            plane
        predicted_obj_mesh (o3d.cpu.pybind.geometry.TriangleMesh): Open3D mesh object representing
            the predicted object pose
        camera_poses (list): List of camera poses to render the scene from. Should be in o3d
            camera convention - if they come from TMF poses, they must be converted to o3d first.
        dset_type (str): Type of dataset, must be 'sim' or 'real'
        dset_path (str): Path to the dataset
        sample_idx (int): Index of the sample, used to retrieve the corresponding real rgb image
        output_path (str): Path to save the rendered image. Should end in '.png'
    """

    plane_color = [0.3, 0.3, 0.3]
    predicted_mesh_color = [0.8, 0.1, 0.1]
    gt_mesh_color = [0.1, 0.8, 0.1]

    plane_mesh.paint_uniform_color(plane_color)
    predicted_obj_mesh.paint_uniform_color(predicted_mesh_color)
    gt_obj_mesh.paint_uniform_color(gt_mesh_color)

    # save as mesh
    # TODO: enable open3d visualization on remote server
    combined_mesh = plane_mesh + predicted_obj_mesh + gt_obj_mesh
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    combined_mesh.compute_vertex_normals()
    o3d.io.write_triangle_mesh(output_path, combined_mesh)
    print("output_path ",output_path)
    return

    vis = o3d.visualization.Visualizer()
    # window size needs to stay this - otherwise
    # ctr.convert_from_pinhole_camera_parameters does not work due to an o3d limitation
    window_size = {"width": 848, "height": 480}
    vis.create_window(**window_size)
    vis.add_geometry(plane_mesh)
    vis.add_geometry(predicted_obj_mesh)
    vis.add_geometry(gt_obj_mesh)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    full_image = Image.new(
        "RGB",
        (
            window_size["width"] if dset_type == "sim" else window_size["width"] * 2,
            window_size["height"] * len(camera_poses),
        ),
    )

    for frame_idx, cam_pose in enumerate(camera_poses):
        vis.reset_view_point(True)
        vis.update_geometry(plane_mesh)
        vis.update_geometry(predicted_obj_mesh)
        vis.update_geometry(gt_obj_mesh)

        # set the camera pose
        ctr = vis.get_view_control()
        parameters = ctr.convert_to_pinhole_camera_parameters()
        parameters.extrinsic = homog_inv(cam_pose)

        # https://github.com/isl-org/Open3D/issues/1164#issuecomment-2474064640
        ctr.convert_from_pinhole_camera_parameters(parameters, allow_arbitrary=True)

        vis.poll_events()
        vis.update_renderer()

        # capture the screen image as a numpy array
        screen_image = vis.capture_screen_float_buffer(do_render=True)
        screen_image = (np.asarray(screen_image) * 255).astype(np.uint8)
        screen_image = Image.fromarray(screen_image)

        # paste the captured image into the full image
        full_image.paste(screen_image, (0, frame_idx * window_size["height"]))

        # if using real data, also paste in the corresponding real rgb image
        if dset_type == "real":

            rgb_image_path = os.path.join(
                dset_path, original_filename, "realsense", "rgb", f"{frame_idx + 1:06d}.png"
            )
            rgb_image = Image.open(rgb_image_path)
            rgb_image = rgb_image.resize(
                (window_size["width"], window_size["height"]), Image.Resampling.LANCZOS
            )
            full_image.paste(rgb_image, (window_size["width"], frame_idx * window_size["height"]))

    full_image.save(output_path)

    vis.destroy_window()


def create_plane_mesh(
    a: float,
    b: float,
    c: float,
    d: float,
    x_bounds: Tuple[float, float] = (-5, 5),
    y_bounds: Tuple[float, float] = (-5, 5),
# ) -> o3d.cpu.pybind.geometry.TriangleMesh:
):
    """
    Create a triangle mesh representing the plane defined by the equation a*x + b*y + c*z + d = 0.

    Assumes the mesh is not (very near to) vertical.
    """

    def get_z(x, y):
        return (-a * x - b * y - d) / c

    # create a point at each corner of the plane
    p1 = np.array([x_bounds[0], y_bounds[0], get_z(x_bounds[0], y_bounds[0])])
    p2 = np.array([x_bounds[0], y_bounds[1], get_z(x_bounds[0], y_bounds[1])])
    p3 = np.array([x_bounds[1], y_bounds[1], get_z(x_bounds[1], y_bounds[1])])
    p4 = np.array([x_bounds[1], y_bounds[0], get_z(x_bounds[1], y_bounds[0])])

    # create a mesh from the points with two triangles
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector([p1, p2, p3, p4])
    # the below vertex windings create a plane normal pointing in the positive z direction,
    # so it is visible from above
    mesh.triangles = o3d.utility.Vector3iVector([[0, 2, 1], [0, 3, 2]])

    return mesh


def plot_metrics(
    all_results: dict, output_dir: str, metrics: list, symmetric_object: bool = False
) -> None:
    """
    For each metric and each inference mode, plot the distribution of the metric as a histogram.

    Args:
        all_results (dict): Dictionary containing the results of the model predictions
        output_dir (str): Directory to save the plots
        metrics (list): List of metrics to plot
        symmetric_object (bool): If True, the object is symmetric. Changes what metrics are used.
    """
    inference_modes = list(all_results.keys())

    os.makedirs(output_dir, exist_ok=True)

    if symmetric_object:
        metrics = [metric for metric in metrics if metric not in ["ADD", "rotation_error"]]

    for metric in metrics:
        fig, ax = plt.subplots(1, len(inference_modes), figsize=(5 * len(inference_modes), 5))
        subplot_idx = 0
        for inference_mode in inference_modes:
            metric_values = [
                all_results[inference_mode][i][metric]
                for i in range(len(all_results[inference_mode]))
            ]
            ax[subplot_idx].hist(metric_values, bins=50)
            ax[subplot_idx].set_title(inference_mode)
            subplot_idx += 1
        # set all plot x limits to be the same
        x_max = max([ax[i].get_xlim()[1] for i in range(len(inference_modes))])
        for i in range(len(inference_modes)):
            ax[i].set_xlim(0, x_max)
        fig.suptitle(metric)
        fig.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{metric}.png"))

    # Plot AUC-ADD curves
    if symmetric_object:
        auc_metrics = ["ADD-S"]
    else:
        auc_metrics = ["ADD", "ADD-S"]
    for add in auc_metrics:
        auc_add_thresholds = np.linspace(0, 0.1, 100)
        plt.figure(figsize=(8, 6))
        for inference_mode in all_results.keys():
            add_values = [
                all_results[inference_mode][i][f"{add}"]
                for i in range(len(all_results[inference_mode]))
            ]
            add_values = np.array(add_values)
            accuracies = [(add_values <= threshold).mean() for threshold in auc_add_thresholds]
            plt.plot(
                auc_add_thresholds,
                accuracies,
                label=f"{inference_mode} (AUC = {np.trapz(accuracies, auc_add_thresholds)*10.0:.4f})",
                linewidth=2,
            )
        plt.xlabel("Threshold (meters)", fontsize=12)
        plt.ylabel("Accuracy", fontsize=12)
        plt.title(f"AUC-{add} Curves", fontsize=14)
        plt.grid(alpha=0.3)
        plt.legend(fontsize=10, loc="lower right")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"AUC_{add}.png"))
        plt.close()


def compute_auc_add(all_results: List[dict]) -> float:
    """
    Compute the AUC-ADD metric for a list of results

    Args:
        all_results (List[dict]): List of results, each containing the ADD metric

    Returns:
        float: AUC-ADD metric
    """

    ret_dict = {}
    for add in ["ADD", "ADD-S"]:
        add_values = np.array([result[add] for result in all_results])
        thresholds = np.linspace(0, 0.1, 100)
        accuracies = [(add_values <= threshold).mean() for threshold in thresholds]
        auc_add = np.trapz(accuracies, thresholds) * 10.0
        ret_dict[add] = auc_add

    return ret_dict["ADD"], ret_dict["ADD-S"]


def get_pred_metrics(
    pred_rot_6d: torch.Tensor,
    gt_rot_6d: torch.Tensor,
    pred_translation: torch.Tensor,
    gt_translation: torch.Tensor,
    gt_obj_pcd: torch.Tensor,
    pred_obj_pcd: torch.Tensor,
):
    """
    Get prediction metrics (e.g. rotation error, translation error, ADD) for a single sample

    Args:
        pred_rot_6d (torch.Tensor): Predicted 6D rotation
        gt_rot_6d (torch.Tensor): Ground truth 6D rotation
        pred_translation (torch.Tensor): Predicted translation
        gt_translation (torch.Tensor): Ground truth translation
        gt_obj_pcd (torch.Tensor): Ground truth object point cloud
        pred_obj_pcd (torch.Tensor): Predicted object point cloud

    Returns:
        dict: Dictionary containing the prediction metrics:
            ADD: Average distance between paired points on the ground truth and predicted model,
                as used in PoseCNN, FoundationPose, etc. Not valid for symmetric objects.
            ADD-S: Symmetric version of ADD which uses the closest point distance rather than
                matching pairs of points. Valid for symmetric (and non-) objects.
            translation_error: Euclidean distance between the predicted and ground truth
                translations
            rotation_error: Angular distance between the predicted and ground truth rotations.
                Find the axis-angle representation of the rotation difference and take the angle.
    """

    ADD = torch.mean(torch.norm(gt_obj_pcd - pred_obj_pcd, dim=1)).item()

    # TODO the vectorized verison of ADD-S uses a ton of memory but this looped version is really slow
    # time: 0.055s
    # point_distances = []
    # for point_idx in range(gt_obj_pcd.shape[0]):
    #     point_distances.append(
    #         torch.norm(gt_obj_pcd[point_idx] - pred_obj_pcd, dim=1).min().item()
    #     )
    # ADD_S = np.mean(point_distances)

    # time: 0.0018s
    # code from https://github.com/NVlabs/FoundationPose/blob/fbbbe456c6f841d844025b5e493db1f731164e3d/Utils.py#L242
    nn_index = cKDTree(pred_obj_pcd.cpu().detach().numpy())
    nn_dists, _ = nn_index.query(gt_obj_pcd.cpu().detach().numpy(), k=1, workers=-1)
    ADD_S = nn_dists.mean()

    translation_error = torch.norm(gt_translation - pred_translation).item()

    # rotation error
    # https://math.stackexchange.com/q/2113634
    def get_angle(mat_1, mat_2):
        rot = np.dot(mat_1, mat_2.T)
        cos_theta = (np.trace(rot) - 1) / 2
        return np.rad2deg(np.arccos(cos_theta))

    pred_rot_matrix = rotation_6d_to_matrix(pred_rot_6d).detach().cpu().numpy()
    gt_rot_matrix = rotation_6d_to_matrix(gt_rot_6d).detach().cpu().numpy()
    rotation_error = float(get_angle(pred_rot_matrix, gt_rot_matrix))

    return {
        "ADD": ADD,
        "ADD-S": ADD_S,
        "translation_error": translation_error,
        "rotation_error": rotation_error,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test a pose estimation model")
    parser.add_argument(
        "-o",
        "--opts",
        type=str,
        help="Path to YAML file containing training options",
        required=True,
    )
    args = parser.parse_args()

    # The opts filename will be used as the directory name for output
    opts_fname = os.path.basename(args.opts).split(".")[0]

    # Open and read in opts YAML file
    with open(args.opts, "r") as f:
        opts = yaml.safe_load(f)

    # Create output directory and copy yaml tile to it
    output_dir = f"6d_pose_estimation/results/eval/{opts_fname}"
    os.makedirs(output_dir, exist_ok=True)
    with open(f"{output_dir}/opts.yaml", "w") as f:
        yaml.dump(opts, f)

    test(
        opts["dset_path"],
        opts["dset_type"],
        output_dir,
        training_path=opts["training_path"],
        batch_size=opts["batch_size"],
        rot_type=opts["rot_type"],
        obj_path=opts["obj_path"],
        num_samples_to_vis=opts["num_samples_to_vis"],
        test_on_split=opts["test_on_split"],
        sensor_plane_path=opts["sensor_plane_path"],
        subsample_n_test_samples=opts["subsample_n_test_samples"],
        include_hist_idxs=opts["include_hist_idxs"],
        symmetric_object=opts["symmetric_object"],
        opt_params=opts["opt_params"],
    )
