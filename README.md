## Prepare data

 - [MANO](https://github.com/otaheri/MANO) go for this repo and download mano model data
 - I download the model in path: MANO/mano_v1_2/models/MANO_LEFT.pkl
 <!-- - [FreiHAND](https://lmb.informatik.uni-freiburg.de/projects/freihand/) go for this website and download hand dataset
 - I download the dataset in path: FreiHAND_dataset/mano, I then store the mesh in FreiHAND_dataset/mesh and simulated histograms in FreiHAND_dataset/simulation
 - I prepared a example data for simulated hists based on hand mesh data in FreiHAND: [simulated_hists](https://drive.google.com/file/d/1k7VrNyP7q7CQ2-1AlsigkLZFGM5-vggZ/view?usp=drive_link) -->
- Prepare DART pose data in DART/DART_pose_data.npz (in google drive)



## Install dependencies:

```sh
conda create -n spad_hand python=3.10
conda activate spad_hand
pip install -r requirements.txt
cd manotorch && pip install -e . && cd ..
```

 - Cuda version on my device is 11.8

## Generate simulation data

```sh
python SPAD-Hand-Sim/sim_hist_batch.py 
```
 - Please check updated simulation code in [SPAD-Hand-Sim](https://github.com/adrenaline21/SPAD-Hand-Sim)

## Train pose estimation model

```sh
python pose_estimation/train.py

python pose_estimation/eval.py
```

## Finetune pose estimation model on real data

```sh
python pose_estimation/finetune_real_data.py

python pose_estimation/eval_real_data.py
```

## Calibrate hand mano parameters/camera parameters

 - Please check updated code in [SPAD-Hand-Sim](https://github.com/adrenaline21/SPAD-Hand-Sim)



## Demo

 - run Demo.ipynb to check the hand mesh for given mano parameters

 - Mano model demo is adapted from repo:[MANO](https://github.com/otaheri/MANO)


