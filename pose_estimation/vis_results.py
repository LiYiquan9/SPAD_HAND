"""
Visualize hand pose estimation results. Given the path to some eval results, e.g. 
"pose_estimation/results/eval/pt_sim_test_carter", this script creates a folder called "vis" in
the provided directory and saves a visualization of the hand pose estimation results for each
image in the eval set.

Visualization code is based on demo.ipynb
"""

import argparse
import io
import json
import os

import mano
import numpy as np
import torch
import trimesh
from mano.utils import Mesh
from PIL import Image, ImageDraw, ImageFont

MODEL_PATH = "MANO/man_v1_2/models/MANO_RIGHT.pkl"
N_COMPS = 45
BATCH_SIZE = 1
IMG_SAVE_RESOLUTION = (720, 720)
SHOW_AXES = False
VIEWPOINTS = [  # viewpoints to use for visualization
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.4],
        [0.0, 0.0, 0.0, 1.0],
    ],
    [
        [0.0, 0.0, 1.0, 0.3],
        [0.0, 1.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0, 0.1],
        [0.0, 0.0, 0.0, 1.0],
    ],
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.3],
        [0.0, -1.0, 0.0, 0.1],
        [0.0, 0.0, 0.0, 1.0],
    ],
]


def vis_results(eval_results_dir: str) -> None:

    # load in the output of the model, which contains both model predictions and ground truth
    with open(os.path.join(eval_results_dir, "model_output_data.json"), "r") as f:
        poses = json.load(f)

    # create vis directory to save visualizations
    output_dir = os.path.join(eval_results_dir, "vis")
    os.makedirs(output_dir, exist_ok=True)
    train_output_dir = os.path.join(output_dir, "train")
    test_output_dir = os.path.join(output_dir, "test")
    os.makedirs(train_output_dir, exist_ok=True)
    os.makedirs(test_output_dir, exist_ok=True)

    # load mano model
    mano_model = mano.load(
        model_path=MODEL_PATH,
        is_rhand=True,
        num_pca_comps=N_COMPS,
        batch_size=BATCH_SIZE,
        flat_hand_mean=False,
    )

    for sample_idx, sample in enumerate(poses):
        """
        Generate and create visualization for target mesh
        """
        target_shape = sample["target_shape"]
        target_pose = sample["target_pose"]

        target_pose = np.reshape(target_pose, (45,)).astype(np.float32)
        betas = torch.tensor(target_shape).unsqueeze(0)
        target_pose = torch.tensor(target_pose).unsqueeze(0)
        global_orient = torch.tensor([0.0, 0.0, 0.0]).unsqueeze(0)
        global_orient[0][1] += np.pi / 2
        transl = torch.tensor([-0.1, -0.01, 0.0]).unsqueeze(0)

        target_output = mano_model(
            betas=betas,
            global_orient=global_orient,
            hand_pose=target_pose,
            transl=transl,
            return_verts=True,
            return_tips=True,
        )

        target_mesh = mano_model.hand_meshes(target_output)[0]

        target_scene = trimesh.Scene(target_mesh)

        if SHOW_AXES:
            # Add the XYZ axis to the scene for reference
            axis = trimesh.creation.axis(origin_size=0.01, transform=None)
            target_scene.add_geometry(axis)

        """
        Generate and create visualization for predicted mesh
        """
        pred_shape = sample["prediction_shape"]
        pred_pose = sample["prediction_pose"]

        pred_pose = np.reshape(pred_pose, (45,)).astype(np.float32)
        betas = torch.tensor(pred_shape).unsqueeze(0)
        pred_pose = torch.tensor(pred_pose).unsqueeze(0)
        global_orient = torch.tensor([0.0, 0.0, 0.0]).unsqueeze(0)
        global_orient[0][1] += np.pi / 2
        transl = torch.tensor([-0.1, -0.01, 0.0]).unsqueeze(0)

        pred_output = mano_model(
            betas=betas,
            global_orient=global_orient,
            hand_pose=pred_pose,
            transl=transl,
            return_verts=True,
            return_tips=True,
        )

        pred_mesh = mano_model.hand_meshes(pred_output)[0]

        pred_scene = trimesh.Scene(pred_mesh)

        if SHOW_AXES:
            # Add the XYZ axis to the scene for reference
            axis = trimesh.creation.axis(origin_size=0.01, transform=None)
            pred_scene.add_geometry(axis)

        """
        Generate and create visualization for avg. ground truth baseline mesh
        """
        avg_train_shape = sample["avg_train_shape"]
        avg_train_pose = sample["avg_train_pose"]

        avg_train_pose = np.reshape(avg_train_pose, (45,)).astype(np.float32)
        betas = torch.tensor(avg_train_shape).unsqueeze(0)
        avg_train_pose = torch.tensor(avg_train_pose).unsqueeze(0)
        global_orient = torch.tensor([0.0, 0.0, 0.0]).unsqueeze(0)
        global_orient[0][1] += np.pi / 2
        transl = torch.tensor([-0.1, -0.01, 0.0]).unsqueeze(0)

        avg_train_output = mano_model(
            betas=betas,
            global_orient=global_orient,
            hand_pose=avg_train_pose,
            transl=transl,
            return_verts=True,
            return_tips=True,
        )

        avg_train_mesh = mano_model.hand_meshes(avg_train_output)[0]

        avg_train_scene = trimesh.Scene(avg_train_mesh)

        if SHOW_AXES:
            # Add the XYZ axis to the scene for reference
            axis = trimesh.creation.axis(origin_size=0.01, transform=None)
            avg_train_scene.add_geometry(axis)

        """
        Render the scenes from each viewpoint and create an image with all of them
        """
        final_img = np.zeros((IMG_SAVE_RESOLUTION[0] * 3, 0, 4), dtype=np.uint8)
        for viewpoint in VIEWPOINTS:
            target_scene.camera_transform = viewpoint
            target_img = target_scene.save_image(resolution=IMG_SAVE_RESOLUTION)

            pred_scene.camera_transform = viewpoint
            pred_img = pred_scene.save_image(resolution=IMG_SAVE_RESOLUTION)

            avg_train_scene.camera_transform = viewpoint
            avg_train_img = avg_train_scene.save_image(resolution=IMG_SAVE_RESOLUTION)

            # save pred_img and target_img side-by-side in one image file
            combined_img = np.concatenate(
                [
                    np.array(Image.open(io.BytesIO(target_img))),
                    np.array(Image.open(io.BytesIO(pred_img))),
                    np.array(Image.open(io.BytesIO(avg_train_img))),
                ],
                axis=0,
            )
            final_img = np.concatenate([final_img, combined_img], axis=1)

        pil_img = Image.fromarray(final_img)

        # add text to the combined img to label gt and pred
        draw = ImageDraw.Draw(pil_img)
        font = ImageFont.truetype("DejaVuSans.ttf", 70)
        draw.text((10, 10), "Ground Truth", font=font, fill=(0, 0, 0, 255))
        draw.text(
            (10, IMG_SAVE_RESOLUTION[0] + 10),
            "Prediction",
            font=font,
            fill=(0, 0, 0, 255),
        )
        draw.text(
            (10, IMG_SAVE_RESOLUTION[0] * 2 + 10),
            "Avg. of Training Set",
            font=font,
            fill=(0, 0, 0, 255),
        )

        if sample["split"] == "train":
            save_path = os.path.join(train_output_dir, f"{sample_idx:06d}.png")
        elif sample["split"] == "test":
            save_path = os.path.join(test_output_dir, f"{sample_idx:06d}.png")

        pil_img.save(save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize hand pose estimation results")
    parser.add_argument(
        "--dir", "-d", type=str, required=True, help="Path to eval results directory"
    )
    args = parser.parse_args()

    vis_results(args.dir)
