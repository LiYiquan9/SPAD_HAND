import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import json


class CustomDataset(Dataset):
    def __init__(self, set="test",num_cameras=8):
        self.set = set
        self.x_data = []
        self.labels = []
        self.num_cameras = num_cameras
        
        
        hists_dataset_path = "data/DART/DART_hist_data_500000_shape_rot_trans_sampling_11_22.npz"
        if self.set == "test":
      
            hists = np.load(hists_dataset_path)['hist'][400000:500000,:,:64]
            self.x_data.append(hists)
            mano_params_pose = np.load(hists_dataset_path)['pose'][400000:500000,0,...]
            mano_params_shape = np.load(hists_dataset_path)['shape'][400000:500000,0,:]
            global_trans = np.load(hists_dataset_path)['hand_trans'].reshape(-1, 3)[400000:500000,0:3]
            global_rot = np.load(hists_dataset_path)['hand_rot'].reshape(-1, 3)[400000:500000,0:3]
            mano_params = np.concatenate([mano_params_pose,mano_params_shape, global_trans,global_rot],axis=1)
            self.labels.append(mano_params)
             
        else:
            hists = np.load(hists_dataset_path)['hist'][:400000,:,:64]
            self.x_data.append(hists)
            mano_params_pose = np.load(hists_dataset_path)['pose'][:400000,0, ...]
            mano_params_shape = np.load(hists_dataset_path)['shape'][:400000,0,:]
            global_trans = np.load(hists_dataset_path)['hand_trans'].reshape(-1, 3)[:400000,0:3]
            global_rot = np.load(hists_dataset_path)['hand_rot'].reshape(-1, 3)[:400000,0:3]
            mano_params = np.concatenate([mano_params_pose,mano_params_shape,global_trans,global_rot],axis=1)
            self.labels.append(mano_params)
            
        self.x_data = np.array(self.x_data)[0]
        self.labels = np.array(self.labels)[0]
    
    def __len__(self):
        if self.set == "test":
            return 100000
        else:
            return 400000

    
    def __getitem__(self, idx):
        return self.x_data[idx], self.labels[idx]
