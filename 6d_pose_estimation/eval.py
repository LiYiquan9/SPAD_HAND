"""
Evaluate a 6D pose estimation model (works on real or simulated data).
"""

import argparse
import datetime
import json
import os
import sys

import yaml
from data_loader import PoseEstimation6DDataset
from model import PoseEstimation6DModel
from tqdm import tqdm

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

    # load obj points
    obj_mesh = trimesh.load(obj_path)
    obj_points = obj_mesh.sample(1000)
    obj_points = torch.tensor(obj_points, dtype=torch.float32).to(device)

    # load dataset
    test_dataset = PoseEstimation6DDataset(dset_path, dset_type=dset_type, split="test")
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
    model.load_state_dict(torch.load(f"{training_path}/model_final.pth"))

    # evaluate model
    model.eval()

    for inference_mode in inference_modes:
        for batch_idx, loaded_data in tqdm(
            enumerate(test_loader), total=len(test_loader), desc=f"Running test data through model ({inference_mode})"
        ):
            if dset_type == "sim":
                raw_input_hists, labels = loaded_data
            elif dset_type == "real":
                raw_input_hists, labels, filenames = loaded_data

            raw_input_hists = torch.tensor(raw_input_hists).float().to(device)
            norm_input_hists = normalize_hists(raw_input_hists).float()
            labels = torch.tensor(labels).float().to(device)

            if inference_mode == "supervised_model":
                outputs = model(norm_input_hists)
            elif inference_mode == "test_set_avg":
                outputs = test_labels_avg.repeat(norm_input_hists.shape[0], 1).to(device).float()
            elif inference_mode == "supervised_and_optimize":
                raise NotImplementedError("supervised_model_and_optimize not implemented yet")
            
            print(outputs.shape)

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
                    "hists": raw_input_hists[sample_idx].tolist(),
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

    print(all_results.keys())

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
    )
