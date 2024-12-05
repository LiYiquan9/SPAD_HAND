import argparse
import json
import logging
import os
from datetime import datetime

import torch
import yaml
from model import MANOEstimator
from data_loaders.sim_data_loader import SimDataset
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device: {device}")


# Evaluation
def eval(checkpoint_path, eval_dataset_path, output_dir):

    logging.basicConfig(
        filename=f"{output_dir}/evalutating_log.log",
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        filemode="w",
    )

    batch_size = 8
    testset = SimDataset(eval_dataset_path, "test")
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=True)

    # Set up model for evaluation
    model = MANOEstimator(device=device, num_cameras=8, rot_type="6d")
    criterion = nn.SmoothL1Loss()
    model.load_state_dict(torch.load(checkpoint_path))
    model.to(device)
    model.eval()

    epoch = 0
    epoch_data = []

    test_loss = 0.0
    for batch_idx, (x, y) in tqdm(enumerate(testloader), total=len(testloader)):
        x = torch.tensor(x).float().to(device)
        y = torch.tensor(y).float().to(device)

        # use indices to downsaple cameras (optional)
        # indices = torch.tensor([0,2,4,6,8,10,12,14])
        # x, h = x[:,indices,:], h[:,indices,:]

        y_pose = y[..., :45]
        y_shape = y[..., 45:55]

        # Forward pass
        outputs = model(x)

        outputs_pose = outputs[..., :45]
        outputs_shape = outputs[..., 45:55]

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
                "dataset": "test",
            }
            epoch_data.append(data)

            logging.info(
                f"Test Epoch: {epoch+1}, Batch: {batch_idx+1}, Loss: {test_loss / (10):.4f}"
            )

    avg_loss = test_loss * 10 / len(testloader)
    logging.info(f"Average Loss: {avg_loss:.4f}")

    with open(f"{output_dir}/epoch_data.json", "w") as f:
        json.dump(epoch_data, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate a pose estimation model on simulated data"
    )
    parser.add_argument(
        "-o",
        "--opts",
        type=str,
        help="Path to YAML file containing evaluation options",
        required=True,
    )
    args = parser.parse_args()

    # The opts filename will be used as the directory name for output
    opts_fname = os.path.basename(args.opts).split(".")[0]

    # Open and read in opts YAML file
    with open(args.opts, "r") as f:
        opts = yaml.safe_load(f)

    assert opts["test_data_type"] == "sim", "Data type must be sim for this script"

    # Create output directory and copy yaml tile to it
    output_dir = f"pose_estimation/results/eval/{opts_fname}"
    os.makedirs(output_dir, exist_ok=True)
    with open(f"{output_dir}/opts.yaml", "w") as f:
        yaml.dump(opts, f)

    eval(opts["checkpoint_path"], opts["eval_dataset_path"], output_dir)
