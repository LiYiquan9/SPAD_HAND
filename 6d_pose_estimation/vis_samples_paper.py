"""
Visualize samples (e.g. 90th, 50th, 10th percentile) visually, for use in the paper
"""

import argparse
import json
import os
from typing import List, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import torch
import torch.nn.functional as F
import yaml
from PIL import Image, ImageEnhance
from util import homog_inv

TMF_TO_REALSENSE_TF = np.array(
    [
        [-1.0, 0.0, 0.0, 0.06452],
        [0.0, -1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.021676],
        [0.0, 0.0, 0.0, 1.0],
    ]
)

# params for which samples to visualize
PTILES_TO_VIS = [0, 50, 99]
METRIC = "ADD-S"
METHODS = ["supervised_model", "supervised_and_optimize"] # ICP is included if icp_results_path is provided
# METHODS = ["supervised_and_optimize"]
VIEWPOINT_IDXS = range(16)
# VIEWPOINT_IDXS = [15]

# grid composition params
IMG_RES = 512  # resolution of a single square image in the grid
GAP_SIZE = 20  # size of gap between images in the grid, in pixels
GRID_BG_COLOR = (255, 255, 255)
CROP_TYPE = "right"  # can be left, right, or center

# visualization rendering params
MESH_ALPHA = 1.0
MESH_COLOR = (0.5, 0.6, 0.9)
STROKE_THICKNESS = 6
STROKE_COLOR = (0, 255, 0)
STROKE_ALPHA = 0.0

# rgb image adjustments
BRIGHTNESS = 1.4
CONTRAST = 1.2


def vis_samples_paper(results_path, icp_results_path):

    with open(os.path.join(results_path, "sorted_results.json"), "r") as f:
        sorted_results = json.load(f)

    with open(os.path.join(results_path, "model_predictions.json"), "r") as f:
        model_predictions = json.load(f)

    with open(os.path.join(results_path, "opts.yaml"), "r") as f:
        eval_opts = yaml.safe_load(f)

    # if icp_results_path is provided, laod the icp results and merge them in with model_predictions
    # and sorted_results
    if icp_results_path is not None:
        with open(os.path.join(icp_results_path, "sorted_results.json"), "r") as f:
            icp_sorted_results = json.load(f)
        with open(os.path.join(icp_results_path, "model_predictions.json"), "r") as f:
            icp_model_predictions = json.load(f)
        model_predictions["ICP"] = icp_model_predictions["ICP"]
        sorted_results["ICP"] = icp_sorted_results["ICP"]
        METHODS.append("ICP")

    os.makedirs(os.path.join(results_path, "paper_vis"), exist_ok=True)

    # restructure model_predictions so that you can find things by the filename (key)
    # so now it will have structure like
    # {"method": {"filename": {}, "filename": {}...}, "method": ...}
    restructured_model_predictions = {}
    for method in METHODS:
        restructured_model_predictions[method] = {}
        for pred in model_predictions[method]:
            restructured_model_predictions[method][pred["filename"]] = pred
    model_predictions = restructured_model_predictions

    for method in METHODS:
        os.makedirs(os.path.join(results_path, "paper_vis", method, "raw"), exist_ok=True)
        os.makedirs(os.path.join(results_path, "paper_vis", method, "grid"), exist_ok=True)
        for viewpoint_idx in VIEWPOINT_IDXS:
            num_samples = len(sorted_results[method][METRIC])
            ptile_ranks = [int(num_samples * ptile / 100) for ptile in PTILES_TO_VIS]
            ptile_keys = [list(sorted_results[method][METRIC].keys())[rank] for rank in ptile_ranks]

            grid_img_res = (
                IMG_RES * len(PTILES_TO_VIS) + GAP_SIZE * (len(PTILES_TO_VIS) - 1),
                IMG_RES,
            )
            grid_img = Image.new("RGB", grid_img_res, GRID_BG_COLOR)

            for ptile, ptile_key, ptile_idx in zip(
                PTILES_TO_VIS, ptile_keys, range(len(PTILES_TO_VIS))
            ):
                pred = model_predictions[method][ptile_key]

                composite_img, rgb_img = vis_prediction(
                    pred,
                    ptile_key,
                    eval_opts["dset_path"],
                    eval_opts["obj_path"],
                    viewpoint_idx,
                    mesh_alpha=MESH_ALPHA,
                )

                # resize image so that the smallest dimension is IMG_RES
                if composite_img.size[0] < composite_img.size[1]:
                    composite_img = composite_img.resize((IMG_RES, int(composite_img.size[1] * IMG_RES / composite_img.size[0])))
                else:
                    composite_img = composite_img.resize((int(composite_img.size[0] * IMG_RES / composite_img.size[1]), IMG_RES))

                # save raw image
                composite_img.save(os.path.join(results_path, "paper_vis", method, "raw", f"{viewpoint_idx}_{ptile}.png"))
                rgb_img.save(os.path.join(results_path, "paper_vis", method, "raw", f"{viewpoint_idx}_{ptile}_rgb.png"))

                composite_img = crop_img_to_square(composite_img, CROP_TYPE)

                grid_img.paste(composite_img, (ptile_idx * (IMG_RES + GAP_SIZE), 0))

            grid_img.save(os.path.join(results_path, "paper_vis", method, "grid", f"{viewpoint_idx}.png"))


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


def rotation_6d_to_matrix(d6: torch.Tensor) -> torch.Tensor:
    """
    Converts 6D rotation representation by Zhou et al. [1] to rotation matrix
    using Gram--Schmidt orthogonalization per Section B of [1].
    Args:
        d6: 6D rotation representation, of size (*, 6)

    Returns:
        batch of rotation matrices of size (*, 3, 3)

    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks.
    IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    Retrieved from http://arxiv.org/abs/1812.07035
    """

    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-2)


def vis_prediction(
    prediction: dict,
    filename: str,
    dset_path: str,
    mesh_path: str,
    viewpoint_idx: int,
    mesh_alpha: float = 0.5,
) -> Image:
    """
    Load in object mesh, plane, camera pose, and RGB image
    """
    obj_mesh = o3d.io.read_triangle_mesh(mesh_path)
    obj_mesh.compute_vertex_normals()
    obj_mesh.transform(
        rot_mat_from_6d_trans(prediction["pred_rot_6d"], prediction["pred_translation"])
    )
    # for debugging
    # obj_mesh.transform(rot_mat_from_6d_trans(prediction["gt_rot_6d"], prediction["gt_translation"]))
    obj_mesh.paint_uniform_color(MESH_COLOR)

    # plane mesh is included in case it's needed in the future, but not currently used
    with open(os.path.join(dset_path, "plane_registration.json"), "r") as f:
        plane_registration = json.load(f)
    plane_mesh = create_plane_mesh(
        plane_registration["plane_a"][0],
        plane_registration["plane_a"][1],
        plane_registration["plane_a"][2],
        -plane_registration["plane_d"],
    )
    plane_mesh.compute_vertex_normals()

    with open(os.path.join(dset_path, filename, "tmf.json"), "r") as f:
        tmf_data = json.load(f)

    cam_pose = np.array(tmf_data[viewpoint_idx]["pose"])
    cam_pose = cam_pose @ TMF_TO_REALSENSE_TF

    rgb_path = os.path.join(dset_path, filename, "realsense", "rgb", f"{viewpoint_idx+1:06d}.png")
    rgb_img = Image.open(rgb_path)
    rgb_img = rgb_img.convert("RGBA")

    # apply adjustments to RGB image (brightness, contrast)
    enhancer = ImageEnhance.Brightness(rgb_img)
    rgb_img = enhancer.enhance(BRIGHTNESS)
    enhancer = ImageEnhance.Contrast(rgb_img)
    rgb_img = enhancer.enhance(CONTRAST)

    """
    Create scene and render
    """
    vis = o3d.visualization.Visualizer()
    # window size needs to stay this - otherwise
    # ctr.convert_from_pinhole_camera_parameters does not work due to an o3d limitation
    window_size = {"width": 848, "height": 480}
    vis.create_window(**window_size)
    # vis.add_geometry(plane_mesh)
    vis.add_geometry(obj_mesh)

    # set the camera pose
    ctr = vis.get_view_control()
    parameters = ctr.convert_to_pinhole_camera_parameters()
    parameters.extrinsic = homog_inv(cam_pose)
    focal_length = 430  # found by trial and error
    parameters.intrinsic.set_intrinsics(
        window_size["width"],
        window_size["height"],
        focal_length,  # x focal length (in pixels)
        focal_length,  # y focal length (in pixels)
        window_size["width"] / 2.0,  # x coord of principal point
        window_size["height"] / 2.0,  # y coord of principal point
    )

    # https://github.com/isl-org/Open3D/issues/1164#issuecomment-2474064640
    ctr.convert_from_pinhole_camera_parameters(parameters, allow_arbitrary=True)

    vis.poll_events()
    vis.update_renderer()

    # capture the screen image as a numpy array
    screen_img = vis.capture_screen_float_buffer(do_render=True)
    screen_img = (np.asarray(screen_img) * 255).astype(np.uint8)
    screen_img = Image.fromarray(screen_img)

    # remove the parts of screen_img that are not the object - they will be white. Replace them
    # with transparent pixels
    screen_img = screen_img.convert("RGBA")
    screen_data = np.array(screen_img)
    white_mask = (
        (screen_data[:, :, 0] == 255)
        & (screen_data[:, :, 1] == 255)
        & (screen_data[:, :, 2] == 255)
    )
    screen_data[white_mask] = [255, 255, 255, 0]
    screen_img = Image.fromarray(screen_data)

    outline_img = get_outline_img(screen_img, STROKE_THICKNESS, STROKE_COLOR)
    # set alpha of non-transparent outline image pixels to STROKE_ALPHA
    outline_data = np.array(outline_img)
    outline_data[outline_data[:, :, 3] != 0, 3] = STROKE_ALPHA * 255
    outline_img = Image.fromarray(outline_data)

    # set alpha of non-transparent screen image pixels to 0.5
    screen_data = np.array(screen_img)
    screen_data[screen_data[:, :, 3] != 0, 3] = mesh_alpha * 255
    screen_img = Image.fromarray(screen_data)

    # overlay the RGB image on top of the screen image
    # composite_img = Image.alpha_composite(rgb_img, screen_img)
    composite_img = Image.alpha_composite(rgb_img, screen_img)
    composite_img = Image.alpha_composite(composite_img, outline_img)

    # fig, ax = plt.subplots(4, 1, figsize=(10, 20))
    # ax[0].imshow(rgb_img)
    # ax[1].imshow(screen_img)
    # ax[2].imshow(composite_img)
    # ax[3].imshow(outline_img)
    # fig.tight_layout()
    # plt.show()

    return composite_img, rgb_img


def get_outline_img(
    img: Image, outline_thickness: int = 5, stroke_color: tuple = (0, 0, 0)
) -> Image:
    """
    Given an image with transparent pixels, create an outline of the object
    in the image, similar to the offset path feature in illustrator.

    Return only the outline, not the original image. Pixels not belonging to the
    outline are fully transparent.
    """
    cv2_img = np.array(img)
    cv2_img = cv2.cvtColor(cv2_img, cv2.COLOR_RGBA2BGRA)

    alpha = cv2_img[:, :, 3]
    # Create a binary mask where object pixels are white (255) and background is black (0)
    _, binary_mask = cv2.threshold(alpha, 1, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    outline_img = np.zeros_like(binary_mask)
    cv2.drawContours(outline_img, contours, -1, 255, thickness=outline_thickness)
    # Smooth the stroke using Gaussian blur
    outline_img = cv2.GaussianBlur(outline_img, (5, 5), 0)

    # Convert stroke to an RGBA image (white stroke on transparent background)
    stroke_rgba = np.zeros_like(cv2_img)
    # instead, make the stroke a certain color
    stroke_rgba[:, :, 0] = stroke_color[2]  # Blue channel
    stroke_rgba[:, :, 1] = stroke_color[1]  # Green channel
    stroke_rgba[:, :, 2] = stroke_color[0]  # Red channel
    stroke_rgba[:, :, 3] = outline_img  # Apply the stroke only to the alpha channel

    return Image.fromarray(cv2.cvtColor(stroke_rgba, cv2.COLOR_BGRA2RGBA))


def rot_mat_from_6d_trans(rot_6d: np.array, trans: np.array) -> np.array:
    """
    Convert ortho6D representation + a translation to a 4x4 transformation matrix

    Args:
        rot_6d: (6,) array of 6D rotation representation
        trans: (3,) array of translation
    Returns:
        (4, 4) array of transformation matrix
    """

    rot_6d = np.array(rot_6d)
    trans = np.array(trans)

    rot_matrix = rotation_6d_to_matrix(torch.from_numpy(rot_6d)).numpy()

    # Create the 4x4 transformation matrix
    transformation_matrix = np.eye(4)
    transformation_matrix[:3, :3] = rot_matrix
    transformation_matrix[:3, 3] = trans

    return transformation_matrix


def crop_img_to_square(img, crop_type):
    if crop_type == "center":
        img = img.crop(
            (
                (img.size[0] - IMG_RES) // 2,
                (img.size[1] - IMG_RES) // 2,
                (img.size[0] + IMG_RES) // 2,
                (img.size[1] + IMG_RES) // 2,
            )
        )
    elif crop_type == "left":
        img = img.crop(
            (
                0,
                (img.size[1] - IMG_RES) // 2,
                IMG_RES,
                (img.size[1] + IMG_RES) // 2,
            )
        )
    elif crop_type == "right":
        img = img.crop(
            (
                img.size[0] - IMG_RES,
                (img.size[1] - IMG_RES) // 2,
                img.size[0],
                (img.size[1] + IMG_RES) // 2,
            )
        )

    return img


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test a pose estimation model")
    parser.add_argument(
        "-r",
        "--results_path",
        type=str,
        help="Path to results folder generated by eval.py. Should contain a 'summary_metrics.json', etc.",
        required=True,
    )
    parser.add_argument(
        "-i",
        "--icp_results_path",
        type=str,
        help="Path to results folder generated by icp baseline. Should contain a 'summary_metrics.json', etc.",
        required=False,
    )
    args = parser.parse_args()

    vis_samples_paper(args.results_path, args.icp_results_path)
