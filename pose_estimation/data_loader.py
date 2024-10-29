import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import json


class CustomDataset(Dataset):
    def __init__(self, set="test",num_cameras=16):
        self.set = set
        self.path = f"FreiHAND_dataset_real/{set}"
        self.x_data = []
        self.h_data = []
        self.labels = []
        self.num_cameras = num_cameras

        theta = np.linspace(0, 2 * np.pi, self.num_cameras, endpoint=False).reshape(self.num_cameras, 1)
        phi = np.full(theta.shape, np.arctan(1.732))
        h = np.full(theta.shape, -0.001)
        r = np.full(theta.shape, 0.045)
        
        params = np.hstack([h, r, theta, phi])
        
        
        if self.set == "test":
            for i in range(8000, 9000):
                sensor_response = np.load(f"FreiHAND_dataset_real/train/simulation/{i}/sensor_response.npz")
                hists = sensor_response["hists_jitter"]
                self.x_data.append(hists)
                
                with open(f"FreiHAND_dataset_real/train/mano/{i}.json", 'r') as file:
                    mano_params = json.load(file)
                    
                    self.labels.append(mano_params[3:58])
                    
                self.h_data.append(params)
        else:
            for i in range(8000):
                sensor_response = np.load(f"FreiHAND_dataset_real/train/simulation/{i}/sensor_response.npz")
                hists = sensor_response["hists_jitter"]
                self.x_data.append(hists)
                
                with open(f"FreiHAND_dataset_real/train/mano/{i}.json", 'r') as file:
                    mano_params = json.load(file)
                    
                    self.labels.append(mano_params[3:58])
                
                self.h_data.append(params)
            
        self.x_data = np.array(self.x_data)
        self.h_data = np.array(self.h_data)
        self.labels = np.array(self.labels)
    
    def __len__(self):
        if self.set == "test":
            return 1000
        else:
            return 8000

    
    def __getitem__(self, idx):
        return self.x_data[idx], self.h_data[idx], self.labels[idx]
