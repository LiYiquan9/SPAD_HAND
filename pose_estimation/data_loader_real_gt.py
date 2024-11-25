import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from pytorch3d.transforms.rotation_conversions import matrix_to_axis_angle
import json
import os

# HAND_MEAN = torch.tensor([0.1116787156904433, -0.04289217484104638, 0.41644183681244606, 0.10881132513711185, 0.0659856788782267, 0.7562200100001023, -0.09639296514009964, 0.09091565922041477, 0.18845929069701614, -0.11809503934386674, -0.05094385260705537, 0.529584499976084, -0.14369840869244885, -0.05524170001527845, 0.7048571406917286, -0.01918291683986582, 0.09233684821597642, 0.3379135244947721, -0.4570329833266365, 0.19628394516488204, 0.6254575328442732, -0.21465237881086027, 0.06599828649166892, 0.5068942070038754, -0.36972435736649994, 0.06034462636097784, 0.07949022787634759, -0.14186969453172144, 0.08585263331718808, 0.6355282566897771, -0.3033415874850632, 0.05788097522832922, 0.6313892099233043, -0.17612088501838064, 0.13209307627166406, 0.37335457646458126, 0.8509642789706306, -0.2769227420650918, 0.09154806978240378, -0.49983943762160615, -0.026556472160458842, -0.05288087673273012, 0.5355591477841195, -0.0459610409551833, 0.2773580212595623])


class RealGTDataset(Dataset):
    def __init__(self, set="test",num_cameras=8):
        self.set = set
        self.path = ""
        self.x_data = []
        self.labels = []
        self.num_cameras = num_cameras
        
        dataset_path_list = [
            "data/real_data/captures/2024-10-23_carter_250",
            # "data/real_data/2024-10-30_casey_250"
        ]
        
        for dataset_path in dataset_path_list:
            labels_path = f"{dataset_path}/gt/labels_corrected.json"
            
            with open(labels_path, 'r') as file:
                label_data = json.load(file)

            hists_dataset_path = f"{dataset_path}/tof"
            
            if self.set == "test":
                for j in range(0,50):
                    if not os.path.exists(f"{hists_dataset_path}/{(j+1):06d}.json"):
                        continue
                    
                    hists = json.load(open(f"{hists_dataset_path}/{(j+1):06d}.json"))

                    hists_target = []
                    for i in range(1, 9):
                        response = np.array(hists[str(i)]["hists"][0][1:]).astype(np.float64)
                        response = np.sum(response, axis=0)  # combine 9 sub-cameras
                        response /= (int(1500e3 * 9)*1.0)
                        hists_target.append(torch.tensor(response))
                    sensor_data = torch.stack(hists_target).cpu().numpy()

                    self.x_data.append(sensor_data)

                    j_string = f"{j+1:06d}"

                    mano_params_pose = (torch.from_numpy(np.array(label_data[j_string]['pose_aa']))).reshape(1,45).cpu().numpy()
                    mano_params_shape = torch.tensor(label_data[j_string]['shape']).reshape(1,10).cpu().numpy()
                    global_trans = torch.tensor(label_data[j_string]['hand_transl']).reshape(1,3).cpu().numpy()
                    global_rot = torch.from_numpy(np.array(label_data[j_string]['wrist_aa']).reshape(1, 3)).cpu().numpy()
                    
                    mano_params = np.concatenate([mano_params_pose,mano_params_shape, global_trans,global_rot],axis=1)
                    
                    self.labels.append(mano_params)
                
            else:
                for j in range(50,250):
                    
                    if not os.path.exists(f"{hists_dataset_path}/{(j+1):06d}.json"):
                        continue
    
                    hists = json.load(open(f"{hists_dataset_path}/{(j+1):06d}.json"))

                    hists_target = []
                    for i in range(1, 9):
                        response = np.array(hists[str(i)]["hists"][0][1:]).astype(np.float64)
                        response = np.sum(response, axis=0)  # combine 9 sub-cameras
                        response /= (int(1500e3 * 9)*1.0)
                        hists_target.append(torch.tensor(response))
                    sensor_data = torch.stack(hists_target).cpu().numpy()

                    self.x_data.append(sensor_data)
   
                    j_string = f"{j+1:06d}"
                    mano_params_pose = (torch.from_numpy(np.array(label_data[j_string]['pose_aa']))).reshape(1,45).cpu().numpy()
                    mano_params_shape = torch.tensor(label_data[j_string]['shape']).reshape(1,10).cpu().numpy()
                    global_trans = torch.tensor(label_data[j_string]['hand_transl']).reshape(1,3).cpu().numpy()
                    global_rot = torch.from_numpy(np.array(label_data[j_string]['wrist_aa']).reshape(1, 3)).cpu().numpy()
                    
                    mano_params = np.concatenate([mano_params_pose,mano_params_shape, global_trans,global_rot],axis=1)
                    
                    self.labels.append(mano_params)

        self.x_data = np.array(self.x_data)[:,:,:64]
        self.labels = np.array(self.labels)[:,0,:]
        
    def __len__(self):
        if self.set == "test":
            return 50 # 100-2
        else:
            return 200 # 400-8

    
    def __getitem__(self, idx):
        return self.x_data[idx], self.labels[idx]

