import json
import numpy as np
import os
import trimesh
from typing import List
import argparse
from scipy.spatial import cKDTree

# --------- Convert 6D rotation to 3x3 matrix ---------
def rot6d_to_matrix(rot_6d):
    a1 = np.array(rot_6d[:3])
    a2 = np.array(rot_6d[3:])
    b1 = a1 / np.linalg.norm(a1)
    b2 = a2 - np.dot(b1, a2) * b1
    b2 /= np.linalg.norm(b2)
    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], axis=1)

# --------- AUC calculation ---------
def compute_auc_add(all_results: List[dict]) -> float:
    ret_dict = {}
    for add in ["ADD", "ADD-S"]:
        add_values = np.array([result[add] for result in all_results])
        thresholds = np.linspace(0, 0.1, 100)
        accuracies = [(add_values <= threshold).mean() for threshold in thresholds]
        auc_add = np.trapz(accuracies, thresholds) * 10.0  # Scale to 0–10 cm
        ret_dict[add] = auc_add
    return ret_dict["ADD"], ret_dict["ADD-S"]

# --------- Main Pipeline ---------
def compute(json_path, gt_dir, mesh_path):
    # Load predictions
    with open(json_path, "r") as f:
        pred_data = json.load(f)

    sorted_keys = sorted(pred_data.keys())
    pred_poses = []
    
    for k in sorted_keys:
        entry = pred_data[k]
        R = rot6d_to_matrix(entry['pred_rot_6d'])
        t = np.array(entry['pred_translation']).reshape(3, 1)
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = t.flatten()
        pred_poses.append(T)

    # Load GT poses
    gt_poses = []
    for i in range(1, 26):
        pose_path = os.path.join(gt_dir, f"{i:03d}/gt/gt_pose.npy")
        gt_pose_data = np.load(pose_path, allow_pickle=True)
        gt_poses.append(gt_pose_data)  # assumes key is T_cam_obj

    assert len(pred_poses) == len(gt_poses), "Prediction and GT count mismatch"
    # Load mesh and sample points
    mesh = trimesh.load(mesh_path, process=False)
    model_points = mesh.sample(1000)

    output_meshes = []
    # Compute ADD / ADD-S
    all_results = []
    for i in range(len(pred_poses)):
        T_gt = gt_poses[i]
        T_pred = pred_poses[i]

        pts_gt = (T_gt[:3, :3] @ model_points.T + T_gt[:3, 3:]).T
        pts_pred = (T_pred[:3, :3] @ model_points.T + T_pred[:3, 3:]).T
        

        add = np.linalg.norm(pts_gt - pts_pred, axis=1).mean()
        tree = cKDTree(pts_gt)
        dist, _ = tree.query(pts_pred, k=1)
        add_s = dist.mean()

        gt_mesh = mesh.copy()
        gt_mesh.apply_transform(T_gt)
        gt_mesh.visual.vertex_colors = [0, 255, 0, 255]  # Green for GT

        pred_mesh = mesh.copy()
        pred_mesh.apply_transform(T_pred)
        pred_mesh.visual.vertex_colors = [255, 0, 0, 255]  # Red for prediction

        green = [0, 255, 0, 255]  # RGBA
        red = [255, 0, 0, 255]

        gt_mesh.visual.vertex_colors = np.tile(green, (len(gt_mesh.vertices), 1))
        pred_mesh.visual.vertex_colors = np.tile(red, (len(pred_mesh.vertices), 1))

        # output_meshes.append(gt_mesh) 
        # output_meshes.append(pred_mesh)
        combined_mesh = trimesh.util.concatenate([gt_mesh, pred_mesh])
        combined_mesh.export(f"/home/yiquan/spad_hand_carter/SPAD_HAND__Carter/6d_pose_estimation/mesh_output/gt_vs_pred_mesh_{i}.ply")
        all_results.append({"ADD": add, "ADD-S": add_s})

    # Compute AUC
    auc_add, auc_adds = compute_auc_add(all_results)
    print(f"✓ AUC-ADD:   {auc_add:.5f}")
    print(f"✓ AUC-ADD-S: {auc_adds:.5f}")


# --------- Entry Point ---------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate 6D pose predictions using ADD/ADD-S AUC")
    parser.add_argument(
        "--pred_json",
        type=str,
        default="",
        help="Path to prediction JSON file"
    )
    parser.add_argument(
        "--gt_dir",
        type=str,
        default="",
        help="Path to directory containing GT gt_pose.npy files (25 folders)"
    )
    parser.add_argument(
        "--mesh",
        type=str,
        default="",
        help="Path to object mesh file (.stl)"
    )
    args = parser.parse_args()
    compute(args.pred_json, args.gt_dir, args.mesh)

