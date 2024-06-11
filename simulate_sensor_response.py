import argparse
import os
import shutil

from sim import load_opt, simulate_sensor_response


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--opt", type=str, help="simulation options")
    parser.add_argument("--name", type=str, help="job name")
    args = parser.parse_args()

    # create experiment folder
    os.makedirs(os.path.join("experiments", "simulation"), exist_ok=True)
    root = os.path.join("experiments", "simulation", args.name)
    os.makedirs(root, exist_ok=True)
    opt = load_opt(args.opt, mode="sensor")
    shutil.copyfile(args.opt, os.path.join(root, "sensor_opt.yaml"))
    opt["_root"] = root

    simulate_sensor_response(opt)