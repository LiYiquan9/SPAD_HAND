"""
Generate a simulated dataset for 6D pose recognition.
"""

import argparse
import datetime
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import trimesh
import yaml
from tqdm import tqdm
from util import convert_json_to_meshhist_pose_format, get_random_rot_matrix

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from spad_mesh.sim.model import MeshHist

BASE_OUTPUT_DIR = "data/sim_data/6d_pose"
LAUNCH_TIME = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def sim_data_adjustment(histograms: np.ndarray) -> np.ndarray:
    """
    Adjust the histograms from sim data to make them more similar to the real data.

    Args:
        histograms: The histograms from sim data.

    Returns:
        The adjusted histograms.
    """
    histograms = np.roll(histograms, shift=1, axis=1)

    return histograms


def gen_sim_dataset(
    mesh_path: str,
    real_dataset_path: str,
    output_dir: str,
    num_hist_bins: int,
    translation_ranges: dict,
    n_obj_poses: int,
    gen_previews: bool = False,
    constraint: str = None,
    t_noise_level: float = 0.0,
) -> None:
    """
    Generate a simulated dataset for 6D pose recognition of a known mesh.

    Args:
        mesh_path: Path to the object mesh.
        real_dataset_path: Path to the real dataset from which to steal camera and plane positions.
        output_dir: Path to the directory where the simulated data will be saved.
        num_hist_bins: The number of bins in the simulated histograms.
        translation_ranges: The x, y,z ranges of the random translation. Should be a dictionary with
            key for each axis, and 2-element list for the range for each dictionary key.
        n_obj_poses: The number of random object poses to generate.
        gen_previews: Whether to generate preview images of the simulated data. Not reocmmended on
            full-size datasets.
        constraint: The constraint for the object poses. Can be "none", "object_on_surface", or
            "object_above_surface".
        t_noise_level: The standard deviation of the translation noise to add to the camera
            positions.
    """

    os.makedirs(os.path.join(output_dir), exist_ok=True)
    if gen_previews:
        os.makedirs(os.path.join(output_dir, "previews"), exist_ok=True)

    # load plane model, mesh, and object mesh
    plane_mesh = trimesh.load(os.path.join(real_dataset_path, "001", "gt", "plane.obj"))
    with open(os.path.join(real_dataset_path, "001", "gt", "plane_model.json")) as f:
        plane_params = json.load(f)
    object_mesh = trimesh.load(mesh_path)

    # read in camera positions and convert to MeshHist format
    with open(os.path.join(real_dataset_path, "001", "tmf.json")) as f:
        tmf_data = json.load(f)
    poses_homog = np.array([measurement["pose"] for measurement in tmf_data])
    cam_rotations, cam_translations = convert_json_to_meshhist_pose_format(poses_homog)

    avg_real_data_obj_translation = get_avg_obj_translation(real_dataset_path)

    object_poses = generate_random_poses(
        center_t=avg_real_data_obj_translation,
        n_poses=n_obj_poses,
        t_ranges=translation_ranges,
        constraint=constraint,
        plane_params=plane_params,
        obj_mesh=object_mesh,
    )

    # (n_object_poses, n_cameras, n_bins)
    all_rendered_hists = np.zeros((len(object_poses), len(poses_homog), num_hist_bins))

    for object_pose_idx, object_pose in tqdm(
        enumerate(object_poses), total=len(object_poses), desc="Generating simulated data"
    ):
        # transform object mesh by the object pose
        transformed_object_mesh = object_mesh.copy()
        transformed_object_mesh.apply_transform(object_pose)

        scene_mesh = trimesh.util.concatenate([plane_mesh, transformed_object_mesh])

        # create forward model for scene mesh + camera positions
        t_noise = np.random.randn(*cam_translations.shape) * t_noise_level
        forward_model = MeshHist(
            camera_config={
                "rotations": cam_rotations,
                "translations": cam_translations + t_noise,
                "camera_ids": np.arange(len(poses_homog)),
            },
            mesh_info={
                "vertices": scene_mesh.vertices,
                "faces": scene_mesh.faces,
                "face_normals": scene_mesh.face_normals,
                "vert_normals": scene_mesh.vertex_normals,
            },
            with_bin_scaling=False,
            num_bins=num_hist_bins,
        )

        # render histograms
        if gen_previews:
            image_output_path = os.path.join(
                output_dir, "previews", f"{object_pose_idx:08d}_render.png"
            )
        else:
            image_output_path = None

        # modify hists to make it close to real hists
        rendered_hist = forward_model(None, None, image_output_path).detach().cpu().numpy()
        # rendered_hist = np.roll(rendered_hist, shift=1, axis=1)
        rendered_hist = sim_data_adjustment(rendered_hist)

        all_rendered_hists[object_pose_idx, :, :] = rendered_hist

        if gen_previews:
            # plot histograms and save it
            fig, ax = plt.subplots(int(len(poses_homog) / 2 + 0.5), 2, figsize=(10, 20))
            ax = ax.flatten()
            for subplot_idx in range(forward_model.num_cameras):
                ax[subplot_idx].plot(
                    all_rendered_hists[object_pose_idx, subplot_idx, :],
                    label="sim_" + str(forward_model.camera_ids[subplot_idx]),
                )
                ax[subplot_idx].legend()
            fig.tight_layout()
            fig.savefig(os.path.join(output_dir, "previews", f"{object_pose_idx:08d}_hists.png"))
            plt.close(fig)

            # save the scene mesh
            scene_mesh.export(
                os.path.join(output_dir, "previews", f"{object_pose_idx:08d}_scene.obj")
            )

    # save simulated data
    np.savez(
        os.path.join(output_dir, "simulated_data.npz"),
        histograms=all_rendered_hists,
        object_poses=object_poses,
    )

    # save metadata about the simulation
    with open(os.path.join(output_dir, "metadata.json"), "w") as f:
        json.dump(
            {
                "object_mesh": mesh_path,
                "real_data_path": real_dataset_path,
                "avg_real_data_obj_translation": list(avg_real_data_obj_translation),
                "num_hist_bins": num_hist_bins,
                "launch_time": LAUNCH_TIME,
                "n_object_poses": len(object_poses),
                "n_camera_poses": len(poses_homog),
                "translation_ranges": translation_ranges,
                "constraint": constraint,
                "plane_params": plane_params,
            },
            f,
            indent=4,
        )


def get_avg_obj_translation(dataset_path: str) -> np.ndarray:
    """
    Get the average transformation from the plane to the object in the dataset.

    Args:
        dataset_path: Path to the dataset.

    Returns:
        The average transformation from the plane to the object.
    """

    obj_translations = []
    for subdir in os.listdir(dataset_path):
        if os.path.isdir(os.path.join(dataset_path, subdir)):
            obj_pose = np.load(os.path.join(dataset_path, subdir, "gt", "gt_pose.npy"))
            obj_translations.append(obj_pose[:3, 3])

    return np.mean(obj_translations, axis=0)


def generate_random_poses(
    center_t: np.ndarray,
    n_poses: int,
    t_ranges: dict,
    constraint: str = "none",
    plane_params: dict = None,
    obj_mesh: trimesh.Trimesh = None,
) -> np.ndarray:
    """
    Generate random object poses around the center translation.

    Args:
        center_t: The center translation around which to generate random poses.
        n_poses: The number of random poses to generate.
        t_ranges: The ranges for the random translations. If constraint is specified, z range may
            be ignored or limited to satisfy constraints.
        constraint: The constraint for the object poses. Can be "none", "object_on_surface", or
            "object_above_surface".
        plane_params: The plane model to use for object pose constraints. Not required if constraint
            is "none".
        obj_mesh: The object mesh to use for object pose constraints. Not required if constraint is
            "none".

    Returns:
        A numpy array of shape (n_poses, 4, 4) containing the random object poses.
    """
    if constraint not in ["none", "object_on_surface", "object_above_surface"]:
        raise ValueError("Invalid constraint specified.")

    if constraint != "none":
        if plane_params is None:
            raise ValueError(
                "Plane model must be provided for object pose constraints if a constraint is specified."
            )

    object_poses = []
    pose_idx = 0
    rejected_samples = 0
    while pose_idx < n_poses:
        translation = [
            center_t[0] + np.random.uniform(t_ranges["x"][0], t_ranges["x"][1]),
            center_t[1] + np.random.uniform(t_ranges["y"][0], t_ranges["y"][1]),
            center_t[2] + np.random.uniform(t_ranges["z"][0], t_ranges["z"][1]),
        ]
        rotation = get_random_rot_matrix(1)[0]

        tf_homog = np.array(
            [
                [rotation[0, 0], rotation[0, 1], rotation[0, 2], translation[0]],
                [rotation[1, 0], rotation[1, 1], rotation[1, 2], translation[1]],
                [rotation[2, 0], rotation[2, 1], rotation[2, 2], translation[2]],
                [0, 0, 0, 1],
            ]
        )

        if constraint != "none":
            # transform the object mesh by the object pose and find the lowest point on the
            # transformed mesh
            transformed_obj_mesh = obj_mesh.copy()
            transformed_obj_mesh.apply_transform(tf_homog)
            lowest_vertex_idx = transformed_obj_mesh.vertices[:, 2].argmin()
            lowest_obj_vertex = transformed_obj_mesh.vertices[lowest_vertex_idx]

            # find the z height of the plane under the lowest vertex
            # this is found by solving ax + by + cz + d = 0 for z
            plane_z = (
                -plane_params["a"] * lowest_obj_vertex[0]
                - plane_params["b"] * lowest_obj_vertex[1]
                - plane_params["d"]
            ) / plane_params["c"]

            if constraint == "object_on_surface":
                # modify the z translation so that the lowest object vertex is on the plane
                tf_homog[2, 3] += plane_z - lowest_obj_vertex[2]
            elif constraint == "object_above_surface":
                # if any point on the object is below the plane, reject the sample and try again
                if plane_z > lowest_obj_vertex[2]:
                    rejected_samples += 1
                    continue

        object_poses.append(tf_homog)
        pose_idx += 1

    if rejected_samples > 0:
        print(f"Rejected {rejected_samples} samples due to constraint violations.")

    return np.array(object_poses)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate a simulated dataset for 6D pose recognition."
    )
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

    output_dir = os.path.join(BASE_OUTPUT_DIR, f"{LAUNCH_TIME}_{opts_fname}")

    gen_sim_dataset(
        mesh_path=opts["mesh_path"],
        real_dataset_path=opts["real_dataset_path"],
        output_dir=output_dir,
        num_hist_bins=opts["num_hist_bins"],
        translation_ranges=opts["translation_ranges"],
        n_obj_poses=opts["n_obj_poses"],
        gen_previews=opts["gen_previews"],
        constraint=opts["constraint"],
        t_noise_level=opts["t_noise_level"],
    )
