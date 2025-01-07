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
from tqdm import tqdm
from util import homog_inv, create_plane_mesh

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import torch
import trimesh
from torch.utils.data import DataLoader

from hand_pose_estimation.utils.utils import matrix_to_rotation_6d, rotation_6d_to_matrix

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

start_time = datetime.datetime.now()


def test(
    dset_path: str,
    dset_type: str,
    output_dir: str,
    batch_size: int,
    training_path: str,
    noise_level: float = 0.00,
    rot_type: str = "6d",
    obj_path: str = "",
    num_samples_to_vis: int = 10,
) -> None:
    """
    Test a 6D pose estimation model (works on real or simulated data)

    Args:
        dset_path (str): Path to the dataset
        dset_type (str): Type of dataset, must be 'sim' or 'real'
        output_dir (str): Directory to save model checkpoints and logs
        batch_size (int): Batch size
        training_path (str): Path to the directory containing the trained model
        noise_level (float): Noise level to add to the input data
        rot_type (str): Type of rotation representation to use
        obj_path (str): Path to the object mesh file

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
    test_dataset = PoseEstimation6DDataset(full_dset_path, dset_type=dset_type, split="test")
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    # calculate the average of the test dataset labels - these will be used as a baseline for
    # comparison (test_set_avg inference mode)
    test_labels = []
    for loaded_data in test_dataset:
        if dset_type == "sim":
            _, label = loaded_data
        elif dset_type == "real":
            _, label, _ = loaded_data
        label_ortho6d = matrix_to_rotation_6d(torch.from_numpy(label[:3, :3])[None, :])[0]
        label_translation = label[:3, 3]
        flat_label = np.concatenate([label_ortho6d, label_translation])
        # print(flat_label)
        test_labels.append(torch.from_numpy(flat_label))
    test_labels = torch.stack(test_labels).to(device)
    test_labels_avg = torch.mean(test_labels, dim=0)

    print(f"Testing dataset has {len(test_dataset)} samples")

    results = np.load(f"{training_path}/mean_std_data.npz")
    MEAN = torch.tensor(results["mean"]).to(device)
    STD = torch.tensor(results["std"]).to(device)

    def normalize_hists(hists):
        return (hists - MEAN) / (STD + 3e-9)

    inference_modes = ["supervised_model", "test_set_avg"]
    all_results = {inference_type: [] for inference_type in inference_modes}

    # load model
    model = PoseEstimation6DModel(device=device).to(device)
    model.load_state_dict(torch.load(f"{training_path}/model_final.pth", weights_only=False))

    # evaluate model
    model.eval()

    for inference_mode in inference_modes:
        for batch_idx, loaded_data in tqdm(
            enumerate(test_loader),
            total=len(test_loader),
            desc=f"Running test data through model ({inference_mode})",
        ):
            if dset_type == "sim":
                raw_input_hists, labels = loaded_data
            elif dset_type == "real":
                raw_input_hists, labels, filenames = loaded_data

            raw_input_hists = raw_input_hists.float().to(device)
            norm_input_hists = normalize_hists(raw_input_hists).float()
            labels = labels.float().to(device)

            if inference_mode == "supervised_model":
                outputs = model(norm_input_hists)
            elif inference_mode == "test_set_avg":
                outputs = test_labels_avg.repeat(norm_input_hists.shape[0], 1).to(device).float()
            elif inference_mode == "supervised_and_optimize":
                raise NotImplementedError("supervised_model_and_optimize not implemented yet")

            if rot_type == "6d":
                gt_rot_6d = matrix_to_rotation_6d(labels[:, :3, :3])
                gt_translation = labels[:, :3, 3].reshape(labels.size(0), 3)

                pred_rot_6d = outputs[:, :6]
                pred_translation = outputs[:, 6:9]

                gt_rot_matrix = labels[:, :3, :3]
                pred_rot_matrix = rotation_6d_to_matrix(outputs[:, :6])

                gt_obj_pcd = torch.matmul(obj_points, gt_rot_matrix.transpose(1, 2))
                pred_obj_pcd = torch.matmul(obj_points, pred_rot_matrix.transpose(1, 2))
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

    # generate plots and visualizations for results
    visualize_results(
        all_results,
        os.path.join(output_dir, "visualizations"),
        obj_path,
        dset_path,
        dset_type,
        num_samples_to_vis,
    )
    plot_metrics(all_results, os.path.join(output_dir, "plots"), pred_metrics.keys())

    # calculate average of metrics and save to file
    avg_metrics = {}
    for inference_mode in inference_modes:
        avg_metrics[inference_mode] = {}
        for metric in pred_metrics.keys():
            avg_metrics[inference_mode][metric] = np.mean(
                [data[metric] for data in all_results[inference_mode]]
            )
    with open(f"{output_dir}/avg_metrics.json", "w") as f:
        json.dump(avg_metrics, f, indent=4)

    # save model output data
    print("Saving model predictions to json...")
    with open(f"{output_dir}/model_predictions.json", "w") as f:
        json.dump(all_results, f, indent=4)


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
                os.path.join(output_dir, inference_mode, f"{sample_idx:06d}.png"),
            )


def render_mesh_from_viewpoints(
    plane_mesh: o3d.cpu.pybind.geometry.TriangleMesh,
    predicted_obj_mesh: o3d.cpu.pybind.geometry.TriangleMesh,
    gt_obj_mesh: o3d.cpu.pybind.geometry.TriangleMesh,
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


def plot_metrics(all_results: dict, output_dir: str, metrics: list) -> None:
    """
    For each metric and each inference mode, plot the distribution of the metric as a histogram.

    Args:
        all_results (dict): Dictionary containing the results of the model predictions
        output_dir (str): Directory to save the plots
        metrics (list): List of metrics to plot
    """
    inference_modes = list(all_results.keys())

    os.makedirs(output_dir, exist_ok=True)

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

    # TODO the vectorized verison of ADD-S uses a ton of memory but this looped version is really
    # slow
    # point_distances = []
    # for point_idx in range(gt_obj_pcd.shape[0]):
    #     point_distances.append(
    #         torch.norm(gt_obj_pcd[point_idx] - pred_obj_pcd, dim=1).min().item()
    #     )
    # ADD_S = np.mean(point_distances)

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
        # "ADD-S": ADD_S,
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
    )
