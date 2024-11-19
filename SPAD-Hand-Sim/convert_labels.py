import pickle as pkl
import numpy as np
import json
import torch
import pytorch3d
import matplotlib.pyplot as plt

from model import SPADHistNV
from meshes import Mesh, colors, hand_meshes
from manotorch.manolayer import ManoLayer

spad_layer = SPADHistNV(
    resolution=256,
    fov=33,
    degree=75,
    cube_mode=False,
)

# mano_layer = ManoLayer(mano_assets_root="data/", flat_hand_mean=False).cuda()
mano_layer  = spad_layer.mano_layer

results = []
for i in range(250):
    label = json.load(open("/real_data/2024-10-30_casey_250/gt/labels.json"))[
        "{:06d}".format(i + 1)
    ]
    pose_l = torch.tensor(label["mano_pose"]).float().cuda().unsqueeze(0)
    shape_l = torch.tensor(label["mano_shape"]).float().cuda().unsqueeze(0)
    transl_l = torch.tensor(label["mano_translation"]).float().cuda().unsqueeze(0)
    rot_l = torch.tensor(label["mano_rotation_matrix"]).float().cuda()

    co = torch.tensor([[0, 0, -1], [0, 1, 0], [1, 0, 0]]).float().cuda()
    global_rot_l = co @ rot_l @ pose_l[0, 0]
    global_rot_aa = pytorch3d.transforms.matrix_to_axis_angle(global_rot_l).unsqueeze(0)
    pose = (
        pytorch3d.transforms.matrix_to_axis_angle(pose_l).flatten()[3:].unsqueeze(0)
        - mano_layer.th_hands_mean
    )

    faces = mano_layer.th_faces.int()
    vertices = mano_layer(torch.cat([global_rot_aa, pose], dim=1), shape_l).verts

    vertices_new = (
        mano_layer(
            torch.cat(
                [
                    pytorch3d.transforms.matrix_to_axis_angle(pose_l[0, 0]).unsqueeze(
                        0
                    ),
                    pose,
                ],
                dim=1,
            ),
            shape_l,
        ).verts
        @ rot_l.T
        @ co.T
    )
    transl = (vertices_new - vertices)[0, 0] + transl_l @ co.T
    results.append(
        {
            "pose_aa": pose[0].detach().cpu().numpy().tolist(),
            "wrist_aa": global_rot_aa[0].detach().cpu().numpy().tolist(),
            "hand_transl": transl[0].detach().cpu().numpy().tolist(),
            "shape": shape_l[0].detach().cpu().numpy().tolist(),
        }
    )
    # hand_mesh = hand_meshes(vertices + transl, faces)[0]
    # hj_meshes = Mesh.concatenate_meshes([hand_mesh, spad_layer.camera_meshes])
    # hj_meshes.show()

open("/real_data/2024-10-30_casey_250/250_mano.json", "w").write(json.dumps(results))
