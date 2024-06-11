import json
import torch
import mano
from mano.utils import Mesh
import trimesh
import math 
import torch.nn as nn
from mano_estimator.calibration.sim_hist import sim_hist
import numpy as np
import torch.optim as optim

class CalibrationModel(nn.Module):
    def __init__(self, shape, pose, cam_loc, cam_norm, first_cam_loc_param):
        super(CalibrationModel, self).__init__()
        # Load the MANO model
        model_path = 'MANO/mano_v1_2/models/MANO_RIGHT.pkl'
        n_comps = 45
        batch_size = 1
    
        self.rh_model = mano.load(model_path=model_path,
                                  is_rhand=True,
                                  num_pca_comps=n_comps,
                                  batch_size=batch_size,
                                  flat_hand_mean=False)
        self.pose = pose
        self.shape = shape
        
        self.cam_loc = cam_loc
        self.cam_norm = cam_norm
        
        self.first_cam_loc_param = first_cam_loc_param
        
        global_orient = torch.tensor([0.0, 0.0, 0.0]).unsqueeze(0)
        global_orient[0][1] += math.pi / 2
        self.global_orient = global_orient
        
        self.transl = torch.tensor([-0.1, -0.01, 0.0]).unsqueeze(0)

        self.output = self.rh_model(betas=self.shape,
                               global_orient=self.global_orient,
                               hand_pose=self.pose,
                               transl=self.transl,
                               return_verts=True,
                               return_tips=True)
        
        self.vertices = self.output.vertices
        
        self.mesh = self.rh_model.hand_meshes(self.output)[0]
        
        
    def forward(self,):
        """
        optimize only on one camera, there is only one histogram to be rendered
        """
        
        self.cam_loc = torch.cat([self.first_cam_loc_param, self.cam_loc[:, 1:]], dim=1)
        hists = sim_hist(self.vertices[0], self.mesh, self.cam_loc, self.cam_norm)  
        
        return hists

    def compute_loss(self, predicted_hists, ground_truth_hists):
        """
        Compute the mean absolute error (MAE) between predicted and ground truth histograms.
        """

        loss_fn = nn.MSELoss()
        hist_loss = loss_fn(predicted_hists.to(torch.float32), ground_truth_hists.to(torch.float32))

        return hist_loss
    

if __name__ == "__main__":
    
    with open('calibration/data/mano_param.json', 'r') as json_file:
        json_data = json.load(json_file)
    
    shapes = torch.tensor([json_data["target_shape"]], dtype=torch.float32, requires_grad=True)
   
    poses = torch.tensor([json_data["target_pose"]], dtype=torch.float32, requires_grad=True)
    
    data = np.load('calibration/data/impulse_response.npz')
    
    cam_loc = torch.tensor(data['camera_loc'][0], dtype=torch.float32, requires_grad=True).unsqueeze(0)
    
    cam_norm = torch.tensor(data['camera_normal'][0], dtype=torch.float32, requires_grad=True).unsqueeze(0)
    
    cam_loc_gt = torch.tensor(data['camera_loc'][1], dtype=torch.float32, requires_grad=True).unsqueeze(0)
    
    cam_norm_gt = torch.tensor(data['camera_normal'][1], dtype=torch.float32, requires_grad=True).unsqueeze(0)
    
    first_cam_loc_param = nn.Parameter(cam_loc[0, 0].clone().unsqueeze(0).unsqueeze(0))
    
    model = CalibrationModel(shapes, poses, cam_loc, cam_norm, first_cam_loc_param)
    
    optimizer = optim.Adam([model.first_cam_loc_param], lr=0.001)
    
    with open('MANO/training_mano.json', 'r') as file:
        datas = json.load(file)

    data = np.load('calibration/data/impulse_response.npz')
    hists_gt = torch.from_numpy(data['hists'])[0].unsqueeze(0)
    # cycle_scale = torch.from_numpy(data["num_cycles"])
    
    for epoch in range(50):  # Example for 10 epochs
        optimizer.zero_grad()
        predicted_hists = model()[0]
        predicted_hists.retain_grad() 
        model.cam_loc.retain_grad() 
        # exit(0)
        loss = model.compute_loss(predicted_hists.unsqueeze(0), hists_gt.to(torch.float32))
        loss.backward(retain_graph=True)
        
        optimizer.step()
        print(f"Epoch {epoch + 1}, Loss: {loss.item()}")
        print(f"model cam_loc: {model.cam_loc.clone().detach().numpy()} ")
        print(f"gt    cam_loc: {cam_loc_gt.clone().detach().numpy()}")

