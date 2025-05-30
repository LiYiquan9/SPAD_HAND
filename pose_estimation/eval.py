import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from data_loader import CustomDataset
from model import MANOEstimator
import logging
from torch.optim import lr_scheduler
import json
from datetime import datetime
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"device: {device}")

batch_size = 8

testset = CustomDataset("test")

testloader = DataLoader(testset, batch_size=batch_size, shuffle=True)

# Model
mano_estimator = MANOEstimator(device=device)

optimizer = optim.Adam(mano_estimator.parameters(), lr=1e-4)

scheduler = lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)

criterion = nn.SmoothL1Loss()

checkpoint_path = 'path to checkpoint'

# Evaluation
def eval(model, testloader, criterion):
    
    now = datetime.now()
    utc_time = now.strftime("%Y-%m-%d_%H-%M-%S")
    os.makedirs(f'results/eval/{utc_time}')
    
    logging.basicConfig(filename=f'results/eval/{utc_time}/evalutating_log.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', filemode='w')

    model.load_state_dict(torch.load(checkpoint_path))
    
    model.to(device)
    
    epoch = 0
    
    epoch_data = []
    
    model.eval()
    
    test_loss = 0.0
    for batch_idx, (x, h, y) in enumerate(testloader):
        x = torch.tensor(x).float().to(device)
        h = torch.tensor(h).float().to(device)
        y = torch.tensor(y).float().to(device)
        
        # use indices to downsaple cameras (optional)
        # indices = torch.tensor([0,2,4,6,8,10,12,14])
        # x, h = x[:,indices,:], h[:,indices,:]

        y_pose = y[...,:45]
        y_shape = y[...,45:55]
    
        # Forward pass
        outputs = model(x, h)
        
        outputs_pose = outputs[...,:45]
        outputs_shape = outputs[...,45:55]
    
        loss = 0.1 * criterion(outputs_shape, y_shape) + criterion(outputs_pose, y_pose)

        test_loss += loss.item()
        if batch_idx % 10 == 0 and batch_idx != 0:  # print data to evaluate model

            data = {
            "prediction_pose": outputs[0][:45].tolist(),  
            "target_pose": y[0][:45].tolist(),
            "prediction_shape": outputs[0][45:55].tolist(),
            "target_shape": y[0][45:55].tolist(),
            "epoch": epoch + 1,
            "batch": batch_idx + 1,
            "loss": test_loss / (10),
            "dataset":"test"
            }
            epoch_data.append(data)
            
            logging.info(f"Test Epoch: {epoch+1}, Batch: {batch_idx+1}, Loss: {test_loss / (10):.4f}")
            
    avg_loss = test_loss * 10/ len(testloader)
    logging.info(f"Average Loss: {avg_loss:.4f}")
    
    with open(f'results/eval/{utc_time}/epoch_data.json', 'w') as f:
            json.dump(epoch_data, f, indent=4)
   

if __name__ == "__main__":
     
    eval(mano_estimator, testloader, criterion)
 