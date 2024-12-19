from typing import List, Tuple
import numpy as np

JSON_TO_MESHHIST_CAM_TF = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]])

def convert_json_to_meshhist_pose_format(poses_homog: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
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