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
    def __init__(self, shape, pose):
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
        self.pose = nn.Parameter(pose)
        self.shape = nn.Parameter(shape)
     
        global_orient = torch.tensor([0.0, 0.0, 0.0]).unsqueeze(0)
        global_orient[0][1] += math.pi / 2
        self.global_orient = global_orient
        
        self.transl = torch.tensor([-0.1, -0.01, 0.0]).unsqueeze(0)


    def forward(self,):
        
        output = self.rh_model(betas=self.shape,
                               global_orient=self.global_orient,
                               hand_pose=self.pose,
                               transl=self.transl,
                               return_verts=True,
                               return_tips=True)
        
        vertices = output.vertices
        
        mesh = self.rh_model.hand_meshes(output)[0]
        
        hists = sim_hist(vertices[0], mesh)  
        
        return hists

    def compute_loss(self, predicted_hists, ground_truth_hists):
        """
        Compute the mean absolute error (MAE) between predicted and ground truth histograms.
        """
        loss_fn = nn.MSELoss()
        return loss_fn(predicted_hists.to(torch.float64), ground_truth_hists.to(torch.float64))
    

if __name__ == "__main__":
    with open('calibration/data/mano_param.json', 'r') as json_file:
        json_data = json.load(json_file)

    shapes = torch.tensor([json_data["prediction_shape"]], dtype=torch.float32, requires_grad=True)
   
    poses = torch.tensor([json_data["prediction_pose"]], dtype=torch.float32, requires_grad=True)
            
    model = CalibrationModel(shapes, poses)
    optimizer = optim.Adam([model.pose, model.shape], lr=0.001) 

    with open('MANO/training_mano.json', 'r') as file:
        datas = json.load(file)

    data = np.load('calibration/sensor_response.npz')
    hists_gt = torch.from_numpy(data['hists_jitter'])
    print("hists size is ", hists_gt.size())
    cycle_scale = torch.from_numpy(data["num_cycles"])
    
    for epoch in range(10):  # Example for 10 epochs
        optimizer.zero_grad()
        predicted_hists = model()
        loss = model.compute_loss(predicted_hists*cycle_scale, hists_gt)
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch + 1}, Loss: {loss.item()}")
















