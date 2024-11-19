import torch
from tqdm import trange
import json
import matplotlib.pyplot as plt
import pickle
import numpy as np
import os

from model import SPADHistNV

torch.manual_seed(7)

num_bins = 128
hand_transl_4 = torch.tensor([[-1.04, -0.5, -0.5]]).cuda()
hand_transl_12 = torch.tensor([[-1.12, -0.5, -0.5]]).cuda()
global_rot_4 = torch.tensor([[1, 0, 0], [0, 1, 0]]).float().cuda()
global_rot_12 = torch.tensor([[1, 0, 0], [0, 1, 0]]).float().cuda()


spad_layer = SPADHistNV(
    num_cameras=8,
    num_bins=num_bins,
    resolution=256,
    fov=33,
    degree=60,
    cube_mode=True,
)
hist_before = (
    spad_layer(
        shape=None,
        pose=None,
        hand_transl=hand_transl_4,
        global_rot=global_rot_4,
        image_path="results/before.png",
        random_sample=False,
    )
    .detach()
    .cpu()
    .numpy()
)

root = "data/real/2024-10-03_calibration/tof/"
hists_target = []
for file in ["000001.json", "000024.json"]:
    hists = json.load(open(root + file))
    label = hists["1"]["capturetime_label"]
    for i in range(1, 9):
        response = np.array(hists[str(i)]["hists"][0][1:]).astype(np.float64)
        response = np.sum(response, axis=0)  # combine 9 sub-cameras
        # response -= np.min(response, axis=0)
        print(response.sum())
        response /= spad_layer.num_cycles
        plt.subplot(4, 2, i)
        plt.plot(response, label=label)
        hists_target.append(torch.tensor(response))
hists_target = torch.stack(hists_target).float().cuda()

# dist = torch.tensor(0.04, requires_grad=True, dtype=torch.float32, device="cuda:0")
spad_layer.albedo.requires_grad = True
spad_layer.noise.requires_grad = True
hand_transl_4.requires_grad = True
hand_transl_12.requires_grad = True
global_rot_4.requires_grad = True
global_rot_12.requires_grad = True
optimizer = torch.optim.Adam(
    [
        hand_transl_4,
        hand_transl_12,
        spad_layer.albedo,
        spad_layer.noise,
        global_rot_4,
        global_rot_12,
    ],
    lr=1e-4,
)
for _ in trange(100):
    optimizer.zero_grad()
    hist_new_4 = spad_layer(None, None, hand_transl_4, global_rot_4)
    hist_new_12 = spad_layer(None, None, hand_transl_12, global_rot_12)
    # rel = torch.cat([hist_new_4, hist_new_12]) / hists_target
    # loss = torch.nn.MSELoss()(rel, torch.ones_like(rel))
    loss = torch.nn.MSELoss()(torch.cat([hist_new_4, hist_new_12]), hists_target)
    loss.backward()
    optimizer.step()
    print(
        # spad_layer.hand_transl,
        hand_transl_4,
        global_rot_4,
        hand_transl_12,
        global_rot_12,
        spad_layer.albedo,
        spad_layer.noise,
        loss.item(),
    )


hist_after = (
    spad_layer(None, None, hand_transl_4, global_rot_4, image_path="results/after.png")
    .detach()
    .cpu()
    .numpy()
)
for i in range(8):
    plt.subplot(4, 2, i + 1)
    # plt.plot(hist_before[i], label="before")
    plt.plot(hist_after[i], label="4cm_after")
    plt.legend()

hist_after = (
    spad_layer(
        None, None, hand_transl_12, global_rot_12, image_path="results/after.png"
    )
    .detach()
    .cpu()
    .numpy()
)
for i in range(8):
    plt.subplot(4, 2, i + 1)
    # plt.plot(hist_before[i], label="before")
    plt.plot(hist_after[i], label="12cm_after")
    plt.legend()

plt.show()
