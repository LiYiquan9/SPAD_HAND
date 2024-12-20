"""
Train a 6D pose estimation model.
"""

import argparse
import os

import yaml
from data_loader import PoseEstimation6DDataset
from model import PoseEstimation6DModel


def train(
    dset_path: str, output_dir: str, epochs: int, batch_size: int, save_model_interval: int
) -> None:
    """
    Train a 6D pose estimation model (works on real or simulated data)

    Args:
        dset_path (str): Path to the dataset
        output_dir (str): Directory to save model checkpoints and logs
        epochs (int): Number of epochs to train for
        batch_size (int): Batch size
        save_model_interval (int): How often to save model checkpoints

    Returns:
        None
    """
    # load dataset
    train_dataset = PoseEstimation6DDataset(dset_path, split="train")
    test_dataset = PoseEstimation6DDataset(dset_path, split="test")

    print(f"Training dataset has {len(train_dataset)} samples")
    print(f"Testing dataset has {len(test_dataset)} samples")

    # load model
    model = PoseEstimation6DModel()

    # train model
    # TODO

    # save model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a pose estimation model")
    parser.add_argument(
        "-o",
        "--opts",
        type=str,
        help="Path to YAML file containing training options",
        required=True,
    )
    args = parser.parse_args()

    # The opts filename will be used as the directory name for output
    opts_fname = os.path.basename(args.opts).split(".")[0]

    # Open and read in opts YAML file
    with open(args.opts, "r") as f:
        opts = yaml.safe_load(f)

    # Create output directory and copy yaml tile to it
    output_dir = f"6d_pose_estimation/results/train/{opts_fname}"
    os.makedirs(output_dir, exist_ok=True)
    with open(f"{output_dir}/opts.yaml", "w") as f:
        yaml.dump(opts, f)

    train(
        opts["dset_path"],
        output_dir,
        epochs=opts["epochs"],
        batch_size=opts["batch_size"],
        save_model_interval=opts["save_model_interval"],
    )
