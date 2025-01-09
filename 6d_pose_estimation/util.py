from typing import List, Tuple

import numpy as np
import open3d as o3d

JSON_TO_MESHHIST_CAM_TF = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]])


def convert_json_to_meshhist_pose_format(
    poses_homog: List[np.ndarray],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert from the camera pose format in real-world json files to the format used by MeshHist.

    Args:
        poses_homog: A list of 4x4 homogenous camera poses, from the real-world json file.

    Returns:
        Tuple of two numpy arrays:
        - cam_rotations: A numpy array of shape (num_cameras, 3, 3) containing the rotation matrices
            for each camera.
        - cam_translations: A numpy array of shape (num_cameras, 3) containing the translation
            vectors for each camera.
    """
    cam_rotations = []
    cam_translations = []
    for pose in poses_homog:
        rotation = pose[:3, :3] @ JSON_TO_MESHHIST_CAM_TF
        cam_rotations.append(rotation)
        cam_translations.append(-pose[:3, 3] @ rotation)

    cam_rotations = np.array(cam_rotations)
    cam_translations = np.array(cam_translations)

    return cam_rotations, cam_translations


def get_random_rot_matrix(num_samples) -> np.ndarray:
    """
    Calculate a random 3x3 rotation matrix, which uniformly samples so(3).

    Adapted from: https://math.stackexchange.com/a/4832876

    Other resources:
    https://math.stackexchange.com/questions/442418/random-generation-of-rotation-matrices/4394036#4394036
    https://math.stackexchange.com/questions/44689/how-to-find-a-random-axis-or-unit-vector-in-3d/44701#44701

    Args:
        num_samples: The number of samples to generate.

    Returns:
        A numpy array of shape (num_samples, 3, 3) containing the rotation matrices.
    """
    u = np.random.uniform(0, 1, size=num_samples)
    z = np.random.randn(num_samples, 1, 3)
    z = z / np.linalg.norm(z, axis=-1, keepdims=True)

    t = np.linspace(0, np.pi, 1024)
    cdf_psi = (t - np.sin(t)) / np.pi
    psi = np.interp(u, cdf_psi, t, left=0, right=np.pi)

    return rot3x3_from_axis_angle(z, psi)


def rot3x3_from_axis_angle(axis_vector: np.ndarray, angle: np.ndarray) -> np.ndarray:
    """
    Calculate a 3x3 rotation matrix from an axis vector and an angle.

    From: https://math.stackexchange.com/a/4832876

    Args:
        axis_vector: A numpy array of shape (n, 3) containing the axis vectors.
        angle: A numpy array of shape (n,) containing the angles.
    """
    angle = np.atleast_1d(angle)[..., None, None]
    K = np.cross(np.eye(3), axis_vector)
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)


def homog_inv(tf: np.ndarray) -> np.ndarray:
    """
    Given some 4x4 homogenous transform, return its inverse.

    Args:
        tf: 4x4 homogenous transform

    Returns:
        4x4 homogenous transform inverse
    """
    rot_inv = np.linalg.inv(tf[:3, :3])
    new_t = (-rot_inv) @ np.array([*tf[:3, 3]])
    return np.array(
        [[*rot_inv[0], new_t[0]], [*rot_inv[1], new_t[1]], [*rot_inv[2], new_t[2]], [0, 0, 0, 1]]
    )


def create_plane_mesh(
    a: float,
    b: float,
    c: float,
    d: float,
    x_bounds: Tuple[float, float] = (-5, 5),
    y_bounds: Tuple[float, float] = (-5, 5),
) -> o3d.cpu.pybind.geometry.TriangleMesh:
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

def get_dc_offset(hist, clip=0.01, bandwidth=0.0001, step_size=0.00001):
    """
    Find the DC offset of a histogram

    Args:
        hist (np.array): histogram
        clip (int): value above the minimum to clip the histogram at, e.g. if the minimum is 80 and
            clip is 500, all values above 580 will be removed from the histogram
    Returns:
        float: DC offset of hist
    """

    # if there's a really high DC offset
    clip = hist.min() + clip

    # remove all values above clip from hist
    hist = hist[hist <= clip]

    # remove all zero values from hist
    hist = hist[hist != 0]
    
    possible_offsets = np.arange(0, clip, step_size)
    # densities is a (n_possible_offsets, n_bins) array
    densities = np.sum(
        (1 / (np.sqrt(2 * np.pi * bandwidth**2)))
        * np.exp(-(((possible_offsets[:, None] - hist[None, :]) ** 2) / (2 * bandwidth**2)))
    , axis=1)

    offset = possible_offsets[np.argmax(densities)]

    return offset