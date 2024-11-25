import torch
from tqdm import trange
import json
import matplotlib.pyplot as plt
import pickle
import numpy as np
from scipy.interpolate import make_interp_spline

from model import SPADHistNV
import time
np.set_printoptions(threshold=np.inf)
torch.manual_seed(7)
from scipy import interpolate

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

hand_batch_size = 1

hand_transl = torch.tensor([[-0.112, -0.012, -0.010]]).cuda()
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

data = np.load("data/DART/DART_pose_data.npz")
mano_data =  torch.from_numpy(data['pose']).cuda() 

mano_data = mano_data[:500000,...]
num_batches = mano_data.shape[0] // hand_batch_size
mano_data = mano_data[:num_batches * hand_batch_size]  
mano_data = mano_data.view(num_batches, hand_batch_size,1, 48)

shape = torch.tensor([[-0.1073, -0.0667, -0.3026, -0.1289, -0.3277, -0.1917, -0.1815,  0.1711,
         -0.0206,  0.0829]]).cuda() # average hand shape

pose_arr = []
hist_arr = []
shape_arr = []
trans_arr = []
rot_arr = []
hand_transl = torch.tensor([[-0.12, -0.008, -0.010]]).cuda()
hand_rot = torch.tensor([[-0.1, 0.002, -0.05]]).cuda()
for batch_idx in trange(num_batches):
    for k in range(1):
        mano_vec = mano_data[batch_idx, :, 0]  
        pose = mano_vec[:, 3:48]    
        shape_noise = (torch.randn(1, 10).cuda() * 0.2 - 0.1) 
        # Get SPAD histogram output for the current batch
        
        extra_trans_noise = (torch.rand(1, 3).cuda() * 0.005 - 0.0025)
        extra_rot_noise = (torch.rand(1, 3).cuda() * 0.01 - 0.005 )
        
        hist_gt = spad_layer(
            shape+shape_noise, pose, hand_transl=hand_transl+extra_trans_noise,hand_rot=hand_rot+extra_rot_noise, image_path=None, random_sample=False
        ).detach()

        pose_arr.append(pose.cpu().numpy())
        hist_arr.append(hist_gt.cpu().numpy())
        shape_arr.append((shape+shape_noise).cpu().numpy())
        trans_arr.append((extra_trans_noise+hand_transl).cpu().numpy())
        rot_arr.append((hand_rot+extra_rot_noise).cpu().numpy())

pose_arr = np.stack(pose_arr)
hist_arr = np.stack(hist_arr)
shape_arr = np.stack(shape_arr)
trans_arr = np.stack(trans_arr)

np.savez('data/DART/DART_hist_data_1000000_shape_rot_trans_sampling_11_10.npz', pose=pose_arr, hist=hist_arr, shape=shape_arr, hand_trans=trans_arr, hand_rot=rot_arr)


