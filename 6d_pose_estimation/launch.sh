# generate synthetic data
python gen_sim_dataset.py -o /home/yiquan/spad_hand_carter/SPAD_HAND__Carter/opts/6d_pose_estimation/gen_sim_data/two_albedo_115_50000.yaml

# tain model on sim data
python train.py -o /home/yiquan/spad_hand_carter/SPAD_HAND__Carter/opts/6d_pose_estimation/train/pt_sim_two_random_albedo_15cam.yaml

# evaluate model on sim data
python eval_sim.py -o your_path/opts/6d_pose_estimation/eval/pt_sim.yaml

# evaluate model on real data
python eval_real.py -o your_path/opts/6d_pose_estimation/eval/pt_real.yaml

# optimize random data (poses are added a random noise, use optimization to recover the original poses)
python optimize_random_data.py -rd /home/yiquan/spad_hand_carter/SPAD_HAND__Carter/data/carter_1_29_two_poses/cow_refined_gt/001 -op /home/yiquan/spad_hand_carter/SPAD_HAND__Carter/data/obj_meshes/two.stl


# visualize the sim and real

# gray 0.25
bash batch_compare_real_sim.sh /home/yiquan/spad_hand_carter/SPAD_HAND__Carter/data/carter_1_29_two_poses/matte_gray_refined_gt

# white 1.15
bash batch_compare_real_sim.sh /home/yiquan/spad_hand_carter/SPAD_HAND__Carter/data/carter_1_29_two_poses/matte_white_refined_gt

# glossy 1.0
bash batch_compare_real_sim.sh /home/yiquan/spad_hand_carter/SPAD_HAND__Carter/data/carter_1_29_two_poses/glossy_white_refined_gt

# cow 0.7
bash batch_compare_real_sim.sh /home/yiquan/spad_hand_carter/SPAD_HAND__Carter/data/carter_1_29_two_poses/cow_refined_gt

# plane
bash batch_compare_real_sim.sh /home/yiquan/spad_hand_carter/SPAD_HAND__Carter/data/carter_1_29_two_poses/plane_only
