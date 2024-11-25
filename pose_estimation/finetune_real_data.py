import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from data_loader_real_gt import RealGTDataset
from model import MANOEstimator
import logging
from torch.optim import lr_scheduler
import json
from datetime import datetime
from manotorch.manolayer import ManoLayer
from utils import axis_angle_to_6d, rot_6d_to_axis_angle
import os
import numpy as np
import wandb 

wandb.init(project="spad_hand_pose_estimation", name="spad_hand_pose_estimator_training")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"device: {device}")

batch_size = 16

num_cameras = 8

trainset = RealGTDataset("train")

trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)

testset = RealGTDataset("test")

testloader = DataLoader(testset, batch_size=batch_size, shuffle=True)

data = np.load('data/mean_std_data.npz')

mean_loaded = data['mean']
std_loaded = data['std']

MEAN = torch.from_numpy(mean_loaded).cuda()
STD = torch.from_numpy(std_loaded).cuda()

def normalize_hists(hists):
    
    return (hists - MEAN) / (STD + 3e-9)

# Model
mano_estimator = MANOEstimator(device=device, num_cameras=num_cameras, rot_type="6d")

optimizer = optim.Adam(mano_estimator.parameters(), lr=1e-4)

scheduler = lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)

criterion = nn.SmoothL1Loss()

l1_metrics = nn.L1Loss()

mano_layer = ManoLayer(
            mano_assets_root="SPAD-Hand-Sim/data", flat_hand_mean=False
        ).cuda()

# shape = torch.tensor([-1.1349,  0.6011, -1.7715, -0.7393, -0.9420, -1.2335, -0.8998,  1.0710,
#           0.8010,  1.2894]).cuda()

checkpoint_path = "pose_estimation/results/train/2024-11-22_10-38-49/model_199.pth"

# Training loop
def train(model, trainloader, testloader, optimizer, criterion, scheduler, epochs, rot_type="aa", save_model_interval=25):
    
    now = datetime.now()
    utc_time = now.strftime("%Y-%m-%d_%H-%M-%S")
    os.makedirs(f'pose_estimation/results/train/{utc_time}')
    
    logging.basicConfig(filename=f'pose_estimation/results/train/{utc_time}/training_log.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', filemode='w')

    model.load_state_dict(torch.load(checkpoint_path))

    model.to(device)
    
    epoch_data = []
    
    for epoch in range(epochs):
        model.train()
        for batch_idx, (x, y) in enumerate(trainloader):

            noise = torch.randn(x.size()).to(device)*0.00
            
            x = torch.tensor(x).float().to(device)

            x = x + noise
            x = normalize_hists(x).float()

            y = torch.tensor(y).float().to(device)
                    
            y_pose = y[...,:45]
            y_shape = y[...,45:55]
            y_trans = y[...,55:58]
            y_rot = y[...,58:61]
            
            optimizer.zero_grad()

            outputs = model(x)
            
            if rot_type == "6d":
                y_pose_6d = axis_angle_to_6d(y_pose.reshape(-1,3)).reshape(y.shape[0],90)
                y_rot_6d = axis_angle_to_6d(y_rot.reshape(-1,3)).reshape(y.shape[0],6)
                outputs_pose_6d = outputs[...,:90]
                outputs_pose = rot_6d_to_axis_angle(outputs_pose_6d.reshape(-1,6)).reshape(outputs.shape[0],45)
                outputs_shape = outputs[...,90:100]
                outputs_trans = outputs[...,100:103]
                outputs_rot_6d = outputs[...,103:109]
                outputs_rot = rot_6d_to_axis_angle(outputs_rot_6d.reshape(-1,6)).reshape(outputs.shape[0],3)
            else:      
                raise Exception('suport only 6d totation type')
            
            # pose_pred = torch.cat([outputs_rot, outputs_pose], dim=1)
            # vertices_pred = (mano_layer(pose_pred, outputs_shape).verts + outputs_trans.unsqueeze(1)).reshape(outputs_pose.shape[0],-1)
            # joint_pred = (mano_layer(pose_pred, outputs_shape).joints + outputs_trans.unsqueeze(1)).reshape(outputs_pose.shape[0],-1)
            
            pose_pred = torch.cat([y_rot, outputs_pose], dim=1)
            vertices_pred = (mano_layer(pose_pred, y_shape).verts + y_trans.unsqueeze(1)).reshape(outputs_pose.shape[0],-1)
            joint_pred = (mano_layer(pose_pred, y_shape).joints + y_trans.unsqueeze(1)).reshape(outputs_pose.shape[0],-1)
            
            
            pose_gt = torch.cat([y_rot, y_pose], dim=1)
            vertices_gt = (mano_layer(pose_gt, y_shape).verts+ y_trans.unsqueeze(1)).reshape(y_pose.shape[0],-1)
            joint_gt = (mano_layer(pose_gt, y_shape).joints+ y_trans.unsqueeze(1)).reshape(y_pose.shape[0],-1)

            
            if rot_type == "6d":
                pose_6d_loss = criterion(outputs_pose_6d, y_pose_6d)
                rot_6d_loss = criterion(outputs_rot_6d, y_rot_6d)
            else:
                raise Exception('suport only 6d totation type')
                
            pose_loss = criterion(outputs_pose, y_pose)
            shape_loss = criterion(outputs_shape, y_shape)
            trans_loss = criterion(outputs_trans, y_trans)
            rot_loss = criterion(outputs_rot, y_rot)
            vertices_loss = criterion(vertices_pred*1000, vertices_gt*1000)
            joint_loss = criterion(joint_gt*1000, joint_pred*1000)

            loss = 1.0*pose_6d_loss + 0.01 * shape_loss + 0.01 * trans_loss + 0.01*rot_loss + 0.05 * vertices_loss + 0.05 * joint_loss

            wandb.log({
                "train_pose_6d_loss": pose_6d_loss.item(),
                "train_shape_loss": shape_loss.item(),
                "train_trans_loss": trans_loss.item(),
                "train_rot_6d_loss": rot_6d_loss.item(),
                "train_vertices_loss": vertices_loss.item(),
                "train_joint_loss": joint_loss.item(),
                "train_total_loss": loss.item()
            })

            logging.info(f"train epoch {epoch} pose:{pose_6d_loss:.3f} shape:{shape_loss:.3f} trans:{trans_loss:.3f} rot:{rot_6d_loss:.3f} vertices:{vertices_loss:.3f} joint:{joint_loss:.3f}")

            logging.info(f"metrics: pose:{l1_metrics(outputs_pose, y_pose):.3f} shape:{l1_metrics(outputs_shape, y_shape):.3f} trans:{l1_metrics(outputs_trans, y_trans):.3f} rot:{l1_metrics(outputs_rot, y_rot):.3f} vertices:{torch.norm((vertices_pred*1000).reshape(-1, 778, 3) - (vertices_gt*1000).reshape(-1, 778, 3), dim=2).mean():.3f} joint:{torch.norm((joint_pred*1000).reshape(-1, 21, 3) - (joint_gt*1000).reshape(-1, 21, 3), dim=2).mean():.3f}")
            
            loss.backward()
            optimizer.step()
            
            # running_loss += loss.item() * x.size(0)
            
            data = {
                "prediction_pose": outputs_pose[0].tolist(),  
                "target_pose": y_pose[0].tolist(),
                "prediction_shape": outputs_shape[0].tolist(),
                "target_shape": y_shape[0].tolist(),
                "prediction_trans": outputs_trans[0].tolist(),
                "target_trans": y_trans[0].tolist(),
                "prediction_rot": outputs_rot[0].tolist(),
                "target_rot": y_rot[0].tolist(),
                "epoch": epoch ,
                "batch": batch_idx ,
                "dataset":"train"
            }
            epoch_data.append(data)
        
        # logging.info(f"Train Epoch: {epoch}, Loss: {running_loss*1.0 / (len(trainloader)):.4f}")
        
        
        scheduler.step()
                
        if epoch % 1 == 0:
            model.eval()

            for batch_idx, (x, y) in enumerate(testloader):
                x = torch.tensor(x).float().to(device)
                x = normalize_hists(x).float()

                y = torch.tensor(y).float().to(device)
                y_pose = y[...,:45]
                y_shape = y[...,45:55]
                y_trans = y[...,55:58]
                y_rot = y[...,58:61]
                
                # Forward pass
                outputs = model(x)
                
                if rot_type == "6d":
                    y_pose_6d = axis_angle_to_6d(y_pose.reshape(-1,3)).reshape(y.shape[0],90)
                    y_rot_6d = axis_angle_to_6d(y_rot.reshape(-1,3)).reshape(y.shape[0],6)
                    outputs_pose_6d = outputs[...,:90]
                    outputs_pose = rot_6d_to_axis_angle(outputs_pose_6d.reshape(-1,6)).reshape(outputs.shape[0],45)
                    outputs_shape = outputs[...,90:100]
                    outputs_trans = outputs[...,100:103]
                    outputs_rot_6d = outputs[...,103:109]
                    outputs_rot = rot_6d_to_axis_angle(outputs_rot_6d.reshape(-1,6)).reshape(outputs.shape[0],3)
                else:      
                    raise Exception('suport only 6d totation type')
                
                # pose_pred = torch.cat([outputs_rot, outputs_pose], dim=1)
                # vertices_pred = (mano_layer(pose_pred, outputs_shape).verts + outputs_trans.unsqueeze(1)).reshape(outputs_pose.shape[0],-1)
                # joint_pred = (mano_layer(pose_pred, outputs_shape).joints + outputs_trans.unsqueeze(1)).reshape(outputs_pose.shape[0],-1)
                pose_pred = torch.cat([y_rot, outputs_pose], dim=1)
                vertices_pred = (mano_layer(pose_pred, y_shape).verts + y_trans.unsqueeze(1)).reshape(outputs_pose.shape[0],-1)
                joint_pred = (mano_layer(pose_pred, y_shape).joints + y_trans.unsqueeze(1)).reshape(outputs_pose.shape[0],-1)
            
                pose_gt = torch.cat([y_rot, y_pose], dim=1)
                vertices_gt = (mano_layer(pose_gt, y_shape).verts+ y_trans.unsqueeze(1)).reshape(y_pose.shape[0],-1)
                joint_gt = (mano_layer(pose_gt, y_shape).joints+ y_trans.unsqueeze(1)).reshape(y_pose.shape[0],-1)

                if rot_type == "6d":
                    pose_6d_loss = criterion(outputs_pose_6d, y_pose_6d)
                    rot_6d_loss = criterion(outputs_rot_6d, y_rot_6d)
                else:
                    raise Exception('suport only 6d totation type')

                pose_loss = criterion(outputs_pose, y_pose)
                shape_loss = criterion(outputs_shape, y_shape)
                trans_loss = criterion(outputs_trans, y_trans)
                rot_loss = criterion(outputs_rot, y_rot)
                vertices_loss = criterion(vertices_pred*1000, vertices_gt*1000)
                joint_loss = criterion(joint_gt*1000, joint_pred*1000)
            
                logging.info(f"test epoch {epoch} pose:{pose_6d_loss:.3f} shape:{shape_loss:.3f} trans:{trans_loss:.3f} rot:{rot_6d_loss:.3f} vertices:{vertices_loss:.3f} joint:{joint_loss:.3f}")

                logging.info(f"metrics: pose:{l1_metrics(outputs_pose, y_pose):.3f} shape:{l1_metrics(outputs_shape, y_shape):.3f} trans:{l1_metrics(outputs_trans, y_trans):.3f} rot:{l1_metrics(outputs_rot, y_rot):.3f} vertices:{torch.norm((vertices_pred*1000).reshape(-1, 778, 3) - (vertices_gt*1000).reshape(-1, 778, 3), dim=2).mean():.3f} joint:{torch.norm((joint_pred*1000).reshape(-1, 21, 3) - (joint_gt*1000).reshape(-1, 21, 3), dim=2).mean():.3f}")
            
                
                data = {
                "prediction_pose": outputs_pose[0].tolist(),  
                "target_pose": y_pose[0].tolist(),
                "prediction_shape": outputs_shape[0].tolist(),
                "target_shape": y_shape[0].tolist(),
                "prediction_trans": outputs_trans[0].tolist(),
                "target_trans": y_trans[0].tolist(),
                "prediction_rot": outputs_rot[0].tolist(),
                "target_rot": y_rot[0].tolist(),
                "epoch": epoch ,
                "batch": batch_idx ,
                "dataset":"test"
                }
                epoch_data.append(data)
                
        
        if (epoch+1)%save_model_interval == 0:
            model_save_path = f'pose_estimation/results/train/{utc_time}/model_{epoch}.pth'
            torch.save(mano_estimator.state_dict(), model_save_path)
            logging.info(f"Model state dictionary saved to {model_save_path}")
    
            with open(f'pose_estimation/results/train/{utc_time}/model_output_data_{epoch}.json', 'w') as f:
                    json.dump(epoch_data, f, indent=4)
        


if __name__ == "__main__":
     
    train(mano_estimator, trainloader, testloader, optimizer, criterion, scheduler, epochs=500, rot_type="6d", save_model_interval=100)
    
    
      