import torch
from torch import nn
from hand_pose_estimation.utils.utils import (matrix_to_rotation_6d,
                                              rotation_6d_to_matrix)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SYMMETRIES = {
    "two": [],
    "cheezits": [], # TODO
    "mustard": [
        torch.tensor([
            [-1, 0, 0, -0.0295],
            [0, -1, 0, -0.046],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
    ]
}

def symmetric_loss(
    outputs_rot_6d: torch.Tensor,
    labels_rot_6d: torch.Tensor,
    outputs_trans: torch.Tensor,
    labels_trans: torch.Tensor,
    mesh_name: str,
    rot_weight: float = 1.0,
    trans_weight: float = 0.5
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Loss function which returns the loss to the closest symmetry transform

    Args:
        outputs_rot_6d (torch.Tensor): Predicted rotation in 6D format (batch_size, 6)
        labels_rot_6d (torch.Tensor): Ground truth rotation in 6D format (batch_size, 6)
        outputs_trans (torch.Tensor): Predicted translation (batch_size, 3)
        labels_trans (torch.Tensor): Ground truth translation (batch_size, 3)
        mesh_name (str): Name of the mesh (e.g. "mustard", "cheezits", "two")

    Returns:
        torch.Tensor: Combined (trans + rot) loss to the closest symmetry transform
    """
    
    l1_loss = nn.L1Loss()
    batch_size = outputs_rot_6d.shape[0]

    # go through every symmetry transform (+ the identity) and calculate the loss
    total_losses = torch.zeros(batch_size, len(SYMMETRIES[mesh_name]) + 1)
    tf_mats = [torch.eye(4), *SYMMETRIES[mesh_name]]
    for tf_mat_idx, tf_mat in enumerate(tf_mats):

        # apply tf_mat to rotations and translations
        outputs_rot_matrix = rotation_6d_to_matrix(outputs_rot_6d[:, :6])
        outputs_homog_matrix = torch.eye(4).repeat(batch_size, 1, 1).to(device) #(batch_size, 4, 4)
        outputs_homog_matrix[:, :3, :3] = outputs_rot_matrix
        outputs_homog_matrix[:, :3, 3] = outputs_trans

        tf_outputs_homog_matrix = outputs_homog_matrix @ tf_mat[None, :, :].to(device)
        
        tf_outputs_rot_6d = matrix_to_rotation_6d(tf_outputs_homog_matrix[:, :3, :3])
        tf_outputs_trans = tf_outputs_homog_matrix[:, :3, 3]

        rot_loss = l1_loss(tf_outputs_rot_6d, labels_rot_6d)
        trans_loss = l1_loss(tf_outputs_trans, labels_trans)

        total_losses[:, tf_mat_idx] = rot_weight * rot_loss + trans_weight * trans_loss

    return torch.min(total_losses, dim=1).values.mean()
