import torch
from tqdm import trange
import json
import matplotlib.pyplot as plt
import pickle
import numpy as np
import trimesh
from pytorch3d.transforms.rotation_conversions import matrix_to_axis_angle, matrix_to_euler_angles, axis_angle_to_matrix
from model import SPADHistNV
from scipy.interpolate import make_interp_spline

torch.manual_seed(7)

def save_hist_img(data, filename="hist.png"):
    # Assuming the data is your numpy array
    data = data[:,:64]
    # Create a figure with a 4x2 layout
    fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(12, 6), sharex=True)

    # Flatten the axes for easier iteration
    axes = axes.flatten()

    # Set a common y-axis limit (for example, from 0 to 300)
    common_ylim = (0, 0.5)

    # Plot each row of the data as a line plot
    for i, ax in enumerate(axes):
        # Original data points
        x = np.arange(len(data[i]))
        y = data[i]

        # Spline interpolation for smoother curve
        x_new = np.linspace(x.min(), x.max(), 300)  # Increase the number of points
        spline = make_interp_spline(x, y, k=3)  # Cubic spline interpolation
        y_smooth = spline(x_new)

        # Plot the smooth line with smaller markers
        ax.plot(x_new, y_smooth, linestyle='-', color='blue')  # Smooth curve
        ax.plot(x, y, marker='o', linestyle='', color='blue', markersize=1)  # Small markers
        
        # Set the common y-axis limits
        ax.set_ylim(common_ylim)

    # Add common x-label
    plt.xlabel('Bin')
    plt.tight_layout()
    plt.savefig(filename)
    
    
    
with open("data/training_mano.json", "r") as json_file:
    mano_data = torch.tensor(json.load(json_file)).cuda()

# f = pickle.load(open("data/part_0/part_0.pkl", "rb"))

EXP_NUM = 23

# batch_idx = [38]
batch_idx = [27]
mano_vec = mano_data[batch_idx, 0]
shape = mano_vec[:, 48:58].clone().detach()
pose = mano_vec[:, 3:48].clone().detach()

hamer_output = np.load(f"real_exp/2024-10-18_yiquan_carter_hands/hamer_output/0000{EXP_NUM}a_hamer_output.npz")
pose = torch.from_numpy(hamer_output["hand_pose"]).cuda()
pose = matrix_to_axis_angle(pose).reshape([1,45])
breakpoint()
rotation_matrix = torch.tensor([
    [0, 1, 0],
    [0, 0, 1],
    [-1, 0, 0]
], dtype=torch.float32)
vertices = (torch.from_numpy(hamer_output["pred_verts"]).float() + torch.tensor([[-0.108, -0.012, -0.014]]))
vertices = torch.matmul(vertices, rotation_matrix.T).cuda()
vertices[:,2] = -vertices[:,2] + 0.01
vertices[:,0] = vertices[:,0] - 0.02
vertices[:,1] = vertices[:,1] - 0.000

hand_transl = torch.tensor([[-0.108, -0.012, -0.014]]).cuda()
num_bins = 128
bin_size = 0.0136
spad_layer = SPADHistNV(
    num_cameras=8,
    num_bins=num_bins,
    bin_size=bin_size,
    resolution=256,
    fov=33,
    degree=75,
    cube_mode=False,
)
hist_gt = spad_layer(
    shape, pose, hand_transl, image_path="results/gt.png", random_sample=False, given_vertices=vertices
).detach()

# fft_gt = torch.fft.fft(hist_gt, dim=1)

# pose_gt = pose.clone()
# pose += 0.3 * torch.rand_like(pose)
# # pose = mano_data[[2], 0][:, 3:48]
# hist_before = (
#     spad_layer(shape, pose, image_path="results/before.png").detach().cpu().numpy()
# )

# hists = json.load(open("/home/yiquan/spad_hand/real_data/2024-09-03_testing/tof/000001.json"))


hists = json.load(open(f"real_exp/2024-10-18_yiquan_carter_hands/tof/0000{EXP_NUM}.json"))
hists_target = []
for i in range(1, 9):
    jitter = np.array(hists[str(i)]["hists"][0][0])
    # print(jitter.sum())
    # plt.plot(jitter / jitter.sum())
    response = np.array(hists[str(i)]["hists"][0][1:]).astype(np.float64)
    response = np.sum(response, axis=0)  # combine 9 sub-cameras
    # print(response.sum())
    print(spad_layer.num_cycles)
    response /= spad_layer.num_cycles
    plt.subplot(4, 2, i)
    plt.plot(response)
    # hists_target.append(torch.cat([torch.tensor(response[11:]), torch.zeros(11)]))
    hists_target.append(torch.tensor(response))
hists_target = torch.stack(hists_target).cuda()

# spad_layer.hand_transl.requires_grad = True
# spad_layer.hand_rot.requires_grad = True
# # pose.requires_grad = True
# # shape.requires_grad = True
# optimizer = torch.optim.Adam(
#     [spad_layer.hand_transl, spad_layer.hand_rot], lr=1e-3
# )
# # optimizer = torch.optim.Adam([pose], lr=1e-2)
# # torch.autograd.set_detect_anomaly(True)
# for _ in trange(200):
#     optimizer.zero_grad()
#     hist_new = spad_layer(shape, pose)
#     loss = torch.nn.L1Loss()(hists_target, hist_new)
#     # fft_new = torch.fft.fft(hist_new, dim=1)
#     # loss = (torch.linalg.norm(fft_new - fft_gt, dim=1) ** 2).mean()
#     loss.backward()
#     optimizer.step()
#     print(
#         # torch.nn.MSELoss()(pose, pose_gt).item(),
#         spad_layer.hand_transl,
#         loss.item(),
#     )

# hist_after = (
#     spad_layer(shape, pose, image_path="results/after.png").detach().cpu().numpy()
# )
for i in range(8):
    plt.subplot(4, 2, i + 1)
    plt.ylim((0, 0.5))
    plt.plot(hist_gt[i].cpu().numpy(), label="sim")
    # plt.plot(hist_before[i], label="before")
    # plt.plot(hist_after[i], label="after")
    plt.legend()

# plt.show()
plt.savefig(f'output_plot_{EXP_NUM}.png')

save_hist_img(hist_gt.cpu().numpy(), "results/hist_sim.png")
save_hist_img(hists_target.cpu().numpy(), "results/hist_real.png")