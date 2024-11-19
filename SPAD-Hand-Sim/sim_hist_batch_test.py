import torch
from tqdm import trange
import json
import matplotlib.pyplot as plt
import pickle
import numpy as np
from scipy.interpolate import make_interp_spline

from model import SPADHistNV
import time
from scipy import interpolate

np.set_printoptions(threshold=np.inf)
# torch.manual_seed(7)

hand_batch_size = 1


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

def transform_hist(hist):
    # hist: [8,128]
    # hist = np.roll(hist, -15, axis=1)
    
    def rescale_vector(vector, new_length):
        x_old = np.linspace(0, 1, len(vector))
        x_new = np.linspace(0, 1, new_length)
        f = interpolate.interp1d(x_old, vector, kind='linear')
        return f(x_new)
    
    hist = np.array([rescale_vector(vec, 200) for vec in hist])
    hist = np.roll(hist, 12, axis=1)
    return hist[:,:128]

def save_hist_img(data):
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
    plt.savefig('SPAD-Hand-Sim/hist0.png')
    
data = np.load("DART/DART_pose_data.npz")
mano_data =  torch.from_numpy(data['pose']).cuda() 

mano_data = mano_data[:100000,...]
num_batches = mano_data.shape[0] // hand_batch_size
mano_data = mano_data[:num_batches * hand_batch_size]  
mano_data = mano_data.view(num_batches, hand_batch_size,1, 48)
         
shape = torch.tensor([[
            -0.31101471185684204,
            -0.12113985419273376,
            -0.09649839997291565,
            -0.0431426465511322,
            -0.0910995602607727,
            -0.22325965762138367,
            -0.20525974035263062,
            0.03374302387237549,
            -0.005946546792984009,
            -0.046449512243270874
        ]]).cuda()

hand_rot = torch.tensor([[-0.1, 0.002, -0.05]]).cuda()
hand_transl = torch.tensor([[-0.12, -0.008, -0.010]]).cuda()
for batch_idx in trange(num_batches):
    batch_idx = 20
    mano_vec = mano_data[batch_idx, :, 0]  
    pose = mano_vec[:, 3:48]    
    shape_noise = (torch.rand(1, 10).cuda() * 1.0 - 0.5)*0.0
    # Get SPAD histogram output for the current batch
    
    extra_trans_noise = (torch.rand(1, 3).cuda() * 0.01 - 0.005)*0.0
    extra_rot_noise = (torch.rand(1, 3).cuda() * 0.02 - 0.01)*0.0
    
    pose = torch.tensor([[
            -0.026395872235298157,
            0.005158312618732452,
            -0.16215792298316956,
            -0.05817986652255058,
            0.06351165473461151,
            -0.5596176981925964,
            0.10054996609687805,
            -0.11684125661849976,
            -0.03167705237865448,
            0.16231843829154968,
            0.10647106170654297,
            -0.2508366107940674,
            -0.06485822796821594,
            0.07206913828849792,
            -0.5181739330291748,
            0.04983578622341156,
            -0.10207875072956085,
            -0.06942430138587952,
            0.4004977345466614,
            0.002179056406021118,
            -0.39787912368774414,
            0.15997305512428284,
            -0.18000522255897522,
            -0.3806765377521515,
            0.10676360130310059,
            -0.015932902693748474,
            -0.08025500178337097,
            0.18428704142570496,
            -0.06745775043964386,
            -0.36655905842781067,
            0.004903912544250488,
            -0.04178883880376816,
            -0.5484133362770081,
            0.011072218418121338,
            -0.12634240090847015,
            -0.05681878328323364,
            -0.22201883792877197,
            0.06658808887004852,
            -0.28276538848876953,
            0.007707744836807251,
            0.1595315933227539,
            0.1857874095439911,
            -0.19585362076759338,
            -0.07300959527492523,
            -0.23190554976463318
        ]]).cuda()
    hand_transl = torch.tensor([[
            -0.0981966033577919,
            0.00511957798153162,
            -0.020203225314617157
        ]]).cuda()
    hand_rot = torch.tensor([[
            0.12175261229276657,
            0.08035773038864136,
            -0.0655851885676384
        ]]).cuda()
    
    
    hist_gt = spad_layer(
        shape+shape_noise, pose, hand_transl=hand_transl+extra_trans_noise,hand_rot=hand_rot+extra_rot_noise, image_path="SPAD-Hand-Sim/results/camera_view.png", random_sample=False
    ).detach()
    # hist_gt = transform_hist(hist_gt.cpu().numpy())
    save_hist_img(hist_gt.cpu().numpy())
    # print(hist_gt.shape)
    exit(0)

    # print(f"Batch {batch_idx+1}/{num_batches} processed")
    # print(f"SPAD histogram shape: {hist_gt.shape}")
    # print(f"Time taken for this batch: {end_time - start_time} seconds")
    
    
# shape_arr = torch.stack(shape_arr) # [10000,10]
# print(shape_arr.shape)
# means = torch.mean(shape_arr, dim=0)

# stds = torch.std(shape_arr, dim=0)
# num_samples = 100
# new_samples_indep = torch.normal(means.expand(num_samples, -1), stds.expand(num_samples, -1))
# # print("new_samples_indep:", new_samples_indep)

# print(means)
# print(stds)

# tensor([-0.1073, -0.0667, -0.3026, -0.1289, -0.3277, -0.1917, -0.1815,  0.1711,
#         -0.0206,  0.0829], device='cuda:0')
# tensor([1.3268, 0.8880, 1.1057, 0.5041, 0.8932, 0.6403, 0.6777, 1.1113, 0.4500,
#         0.6528], device='cuda:0')