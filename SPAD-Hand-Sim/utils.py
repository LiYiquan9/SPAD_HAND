import torch
from pytorch3d.transforms.rotation_conversions import rotation_6d_to_matrix, matrix_to_axis_angle

def rotation_6d_axis_angle(d6: torch.Tensor) -> torch.Tensor:

    return matrix_to_axis_angle(rotation_6d_to_matrix(d6))




