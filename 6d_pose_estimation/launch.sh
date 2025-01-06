# generate synthetic data
python gen_sim_dataset.py -n 50000 -c object_on_surface -m your_path/data/obj_meshes/two.stl --real_data_path your_path/data/two_16_poses_real

# tain model on sim data
python train.py -o your_path/opts/6d_pose_estimation/train/pt_sim.yaml

# evaluate model on sim data
python eval_sim.py -o your_path/opts/6d_pose_estimation/eval/pt_sim.yaml

# evaluate model on real data
python eval_real.py -o your_path/opts/6d_pose_estimation/eval/pt_real.yaml

# optimize random data (poses are added a random noise, use optimization to recover the original poses)
python optimize_random_data.py -rd your_path/data/two_16_poses_real/003 -op your_path/data/obj_meshes/two.stl