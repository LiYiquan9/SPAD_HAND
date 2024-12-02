## Estimation of hand pose from wrist-mounted SPAD measurements

# Installation

## Prepare data
- Download [MANO Data](https://github.com/otaheri/MANO) and place it in MANO/mano_v1_2/models/MANO_LEFT.pkl
- Download DART_pose_data.npz from [the shared Google Drive](https://drive.google.com/file/d/1VroeUgyKbNCiTyURj4jhN9bwxdDi6SXe/view?usp=drive_link). Place it in data/DART/DART_pose_data.npz


## Install dependencies (Yiquan's version):
```sh
conda create -n spad_hand python=3.10
conda activate spad_hand
pip install -r requirements.txt
cd manotorch && pip install -e . && cd ..
```

 - Cuda version on my device is 11.8


## Install dependencies (Carter's version):
```sh
mamba create -n spad_hand python=3.10
mamba activate spad_hand
mamba install pytorch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 pytorch-cuda=12.1 -c pytorch -c nvidia
mamba install -c iopath iopath
mamba install pytorch3d -c pytorch3d
mamba install matplotlib tqdm scipy
cd manotorch && pip install -e . && cd ..
git clone https://github.com/NVlabs/nvdiffrast
cd nvdiffrast
pip install .
mamba install trimesh
pip install chumpy
pip install opencv-python
mamba install ninja
# now you have enough to run sim_hist_batch.py. The below is to be able to run train, eval, etc.
cd manotorch
pip install -e .
```

If you try to run sim_hist_batch.py now, you'll get an ImportError in chumpy's code. So go to the error in your install where the error occurs, e.g. 
`/home/carter/miniforge3/envs/spad_hand/lib/python3.10/site-packages/chumpy/__init__.py` and delete line 11, which is:
```python
from numpy import bool, int, float, complex, object, unicode
```

## (On ThinkPad only) set up symbolic link to real-world data
```sh
ln -s ~/projects/SPAD-Hand-Pose-Capture/data/captures data/real_data
```

# Usage

## Generate simulation data

```sh
python SPAD-Hand-Sim/sim_hist_batch.py 
```
 - Please check updated simulation code in [SPAD-Hand-Sim](https://github.com/adrenaline21/SPAD-Hand-Sim)

## Train pose estimation model
To train a model, you will need to pass an opts file as a parameter, which describes how to train the model. All opts files for training should be in `pose_estimation/results/train`. The name of the opts file is also used to label the traiing run. For example,
```sh
python pose_estimation/train_sim.py -o pose_estimation/results/train/pt_sim
```

## Finetune pose estimation model on real data
Again, you need to pass an opts file as a parameter. All opts files for training should be in `pose_estimation/results/train`, for example:
```sh
python pose_estimation/finetune_real_data.py -o pose_estimation/results/train/pt_sim_ft_carter
```

## Evaluate on real data
An opts file is again required. If you create a new opts file, keep it in `pose_estimation/results/eval`
```sh
python pose_estimation/eval_real_data.py -o pose_estimation/results/eval/pt_sim_ft_carter_test_carter
```

## Calibrate hand mano parameters/camera parameters
 - Please check updated code in [SPAD-Hand-Sim](https://github.com/adrenaline21/SPAD-Hand-Sim)


## Demo
 - run Demo.ipynb to check the hand mesh for given mano parameters

 - Mano model demo is adapted from repo:[MANO](https://github.com/otaheri/MANO)


