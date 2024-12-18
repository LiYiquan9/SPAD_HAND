"""
Visualize simulated data compared to real data of a mesh in a given 6D pose.
"""

import os
import argparse
from spad_mesh.sim.model import MeshHist

def vis_sim_data(real_data_path):
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-rd",
        "--real_data",
        type=str,
        required=True,
        help="Path to real data. Should be folder with 'gt' and 'realsense' subfolders.",
    )

    args = parser.parse_args()
    vis_sim_data(args.real_data)
