import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
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

trainset = CustomDataset("train")

trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)

testset = CustomDataset("test")

testloader = DataLoader(testset, batch_size=batch_size, shuffle=True)

# Model
mano_estimator = MANOEstimator(device=device, num_cameras=16)

optimizer = optim.Adam(mano_estimator.parameters(), lr=1e-4)

scheduler = lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)

criterion = nn.SmoothL1Loss()

# Training loop
def train(model, trainloader, testloader, optimizer, criterion, scheduler, epochs):
    
    now = datetime.now()
    utc_time = now.strftime("%Y-%m-%d_%H-%M-%S")
    os.makedirs(f'results/train/{utc_time}')
    
    logging.basicConfig(filename=f'results/train/{utc_time}/training_log.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', filemode='w')

    model.to(device)
    
    epoch_data = []
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for batch_idx, (x, h, y) in enumerate(trainloader):
            
            # use indices to downsample cameras (optional) 
            # indices = torch.tensor([0,2,4,6,8,10,12,14])
            # x, h = x[:,indices,:], h[:,indices,:]

            noise = torch.randn(x.size()).to(device)*1
            
            x = torch.tensor(x).float().to(device)
            x = x + noise
            
            h = torch.tensor(h).float().to(device)
            y = torch.tensor(y).float().to(device)
            
            y_pose = y[...,:45]
            y_shape = y[...,45:55]
            
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(x, h)
            
            outputs_pose = outputs[...,:45]
            outputs_shape = outputs[...,45:55]

            loss = 0.1 * criterion(outputs_shape, y_shape) + criterion(outputs_pose, y_pose)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            if batch_idx % 100 == 0 and batch_idx != 0: # print data to evaluate model
   
                if batch_idx == 100:
                    data = {
                        "prediction_pose": outputs[0][:45].tolist(),  
                        "target_pose": y[0][:45].tolist(),
                        "prediction_shape": outputs[0][45:55].tolist(),
                        "target_shape": y[0][45:55].tolist(),
                        "epoch": epoch + 1,
                        "batch": batch_idx + 1,
                        "loss": running_loss / (100),
                        "dataset":"train"
                    }
                    epoch_data.append(data)
                
                logging.info(f"Train Epoch: {epoch+1}, Batch: {batch_idx+1}, Loss: {running_loss / (100):.4f}")
                running_loss = 0.0
        
        scheduler.step()
                
        if epoch % 1 == 0:
            model.eval()
            
            test_loss = 0.0
            for batch_idx, (x, h, y) in enumerate(testloader):
                x = torch.tensor(x).float().to(device)
                h = torch.tensor(h).float().to(device)
                y = torch.tensor(y).float().to(device)
                
                # use indices to downsample cameras (optional)
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
                    test_loss = 0.0
                
    with open(f'results/train/{utc_time}/epoch_data.json', 'w') as f:
            json.dump(epoch_data, f, indent=4)
        
    model_save_path = f'results/train/{utc_time}/model_epoch_{epochs}.pth'
    torch.save(mano_estimator.state_dict(), model_save_path)
    logging.info(f"Model state dictionary saved to {model_save_path}")


if __name__ == "__main__":
     
    train(mano_estimator, trainloader, testloader, optimizer, criterion, scheduler, epochs=5)
    
    
    