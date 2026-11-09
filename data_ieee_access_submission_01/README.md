# go1_rl_training

Backup 
cd ~/go1_rl_training

# 1. Is this a git repo at all?
git status

# 2. Is a remote (GitHub) configured, and what URL?
git remote -v

# 3. What branch are you on, and is it tracking a remote branch?
git branch -vv

# If github already set up then
git add -A
git commit -m "write some note"
git push


Check policy
./isaaclab.sh -p scripts/environments/random_agent.py \
  --task Isaac-RelPose-Flat-Unitree-Go1-v0 \
  --num_envs 16



Train Velocity
./isaaclab.sh -p scripts/environments/random_agent.py \
  --task Isaac-Velocity-Flat-Unitree-Go1-v0 \
  --num_envs 16

./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py --task Isaac-Velocity-Flat-Unitree-Go1-v0 \
    --num_envs 2048 --headless --experiment_name go1_velocity --max_iterations 10000

python scripts/reinforcement_learning/rsl_rl/train.py --task Isaac-Velocity-Flat-Unitree-Go1-v0 \
    --num_envs 4096 --headless --experiment_name go1_velocity --max_iterations 5000


./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
  --task Isaac-Velocity-Flat-Unitree-Go1-Play-v0 \
  --num_envs 8 \
  --checkpoint logs/rsl_rl/unitree_go1_flat/model_go1_vel.pt


python scripts/rsl_rl/play.py   --task Isaac-Velocity-Flat-Unitree-Go1-Play-v0   --num_envs 8   --checkpoint logs/rsl_rl/unitree_go1_flat/2026-05-02_17-36-36/model_4500.pt


tensorboard --logdir logs/rsl_rl/unitree_go1_flat

################ GO1 Velocity #########################

python scripts/rsl_rl/train.py \
  --task Isaac-Go1-Velocity-Flat-v0 \
  --num_envs 4096 --headless \
  --max_iterations 5000

python scripts/rsl_rl/play.py  \
  --task Isaac-Go1-Velocity-Flat-Play-v0  \
  --num_envs 8 \
  --checkpoint logs/rsl_rl/go1_velocity_flat/2026-05-02_18-29-54/model_4999.pt

python scripts/rsl_rl/play_vel_plot.py  \
  --task Isaac-Go1-Velocity-Flat-Play-v0  \
  --num_envs 1 \
  --checkpoint logs/rsl_rl/go1_velocity_flat/2026-08-11_17-07-42/model_2999.pt


python scripts/rsl_rl/play_traj5.py  \
  --task Isaac-Go1-Velocity-Flat-Play-v0  \
  --num_envs 8


tensorboard --logdir logs/rsl_rl/go1_velocity_flat



################ GO1 P2P #########################

python scripts/rsl_rl/train.py \
  --task Isaac-Go1-P2P-v0 \
  --num_envs 4096 --headless \
  --max_iterations 10000

python scripts/rsl_rl/play.py  \
  --task Isaac-Go1-P2P-Play-v0  \
  --num_envs 8 \
  --checkpoint logs/rsl_rl/go1_p2p_rough/2026-05-05_18-29-54/model_4999.pt


python scripts/rsl_rl/play_6d.py  \
  --task Isaac-Go1-P2P-Play-v0  \
  --num_envs 8 \
  --checkpoint logs/rsl_rl/go1_p2p_rough/2026-05-05_18-29-54/model_4999.pt


tensorboard --logdir logs/rsl_rl/go1_p2p_rough


################ GO1 P2P_Rel #########################

python scripts/rsl_rl/train.py \
  --task Isaac-Go1-P2P_Rel-v0 \
  --num_envs 4096 --headless \
  --max_iterations 5000

python scripts/rsl_rl/play.py  \
  --task Isaac-Go1-P2P_Rel-Play-v0  \
  --num_envs 8 \
  --checkpoint logs/rsl_rl/go1_p2p_rel_rough/2026-05-05_18-29-54/model_4999.pt


python scripts/rsl_rl/play_6d.py  \
  --task Isaac-Go1-P2P-Play-v0  \
  --num_envs 8 \
  --checkpoint logs/rsl_rl/go1_p2p_rel_rough/2026-05-05_18-29-54/model_4999.pt

python scripts/rsl_rl/play_p2p_time_single.py --task Isaac-Go1-P2P_Rel_Time-Play-v0

tensorboard --logdir logs/rsl_rl/go1_p2p_rel_rough



################ GO1-P2P-Rel-Time #########################
python scripts/rsl_rl/train.py \
  --task Isaac-Go1-P2P_Rel_Time-v0 \
  --num_envs 1 \
  --max_iterations 10\
  --headless

python scripts/rsl_rl/train.py \
  --task Isaac-Go1-P2P_Rel_Time-v0 \
  --num_envs 4096 --headless \
  --max_iterations 5000

tensorboard --logdir logs/rsl_rl/go1_p2p_rel_time_rough


python scripts/rsl_rl/play_p2p.py  \
  --task Isaac-Go1-P2P_Rel_Time-Play-v0  \
  --num_envs 8 \
  --checkpoint logs/rsl_rl/go1_p2p_rel_time_rough/2026-05-05_18-29-54/model_4999.pt


python scripts/rsl_rl/play_6d.py  \
  --task Isaac-Go1-P2P_Rel_Time-Play-v0  \
  --num_envs 8 \
  --checkpoint logs/rsl_rl/go1_p2p_rel_time_rough/2026-05-05_18-29-54/model_4999.pt

python scripts/rsl_rl/play_p2p_time_single.py  \
  --task Isaac-Go1-P2P_Rel_Time-Play-v0  \
  --num_envs 1 \
  --checkpoint logs/rsl_rl/go1_p2p_rel_time_rough/2026-06-08_10-01-57/model_1999.pt

python scripts/rsl_rl/play_p2p_time_single3_plot.py  \
  --task Isaac-Go1-P2P_Rel_Time-Play-v0  \
  --num_envs 1 \
  --plot_mode live\
    --checkpoint logs/rsl_rl/go1_p2p_rel_time_rough/model_go1_p2pt_range_1.5-3.0.pt

python scripts/rsl_rl/play_p2p_time_single3_ortho_savedata.py  \
  --task Isaac-Go1-P2P_Rel_Time-Play-v0  \
  --num_envs 1 \
  --plot_mode live

# clean, watch + export
python scripts/rsl_rl/play_p2p_time_single3_ortho_savedata.py --task Isaac-Go1-P2P_Rel_Time-Play-v0 --export_csv data_01_clean.csv

# with drift noise
python scripts/rsl_rl/play_p2p_time_single3_ortho_savedata.py --task Isaac-Go1-P2P_Rel_Time-Play-v0 --noise_model drift --noise_k 2 --export_csv data_02_drift_k2.csv

# with sensor noise
python scripts/rsl_rl/play_p2p_time_single3_ortho_savedata.py --task Isaac-Go1-P2P_Rel_Time-Play-v0 --sensor_noise all --sensor_k 2 --export_csv data_03_sensors_k2.csv


################ Go1 sim2sim #########################

# GO1 Sim2sim mujoco
python scripts/rsl_rl/play_mujoco_p2pt.py \
    --policy logs/rsl_rl/go1_p2p_rel_time_rough/exported/policy.pt \
    --xml  .././mujoco_menagerie/unitree_go1/scene_isaac_match.xml

python scripts/rsl_rl/play_mujoco_p2pt.py \
    --policy logs/rsl_rl/go1_p2p_rel_time_rough/exported/policy.pt \
    --xml .././mujoco_menagerie/unitree_go1/scene_isaac_match.xml \
    --topdown 

python scripts/rsl_rl/play_mujoco_p2pt.py \
    --policy logs/rsl_rl/go1_p2p_rel_time_rough/exported/policy.pt \
    --xml .././mujoco_menagerie/unitree_go1/scene_isaac_match.xml \
    --topdown --follow


python scripts/rsl_rl/play_p2p_time_single3_plot.py  \
  --task Isaac-Go1-P2P_Rel_Time-Play-v0  \
  --num_envs 1 \
  --checkpoint logs/rsl_rl/go1_p2p_rel_time_rough/exported/policy.pt


python scripts/rsl_rl/play_p2p_time_single3_plot.py  \
  --task Isaac-Go1-P2P_Rel_Time-Play-v0  \
  --num_envs 1 \
  --plot_mode live\
    --checkpoint logs/rsl_rl/go1_p2p_rel_time_rough/2026-08-20_12-56-33/model_14997.pt



python scripts/rsl_rl/test_stand_mujoco_clean.py --xml .././mujoco_menagerie/unitree_go1/scene_isaac_match.xml


################ MC-Velocity #########################
pip uninstall mc-tasks -y
rm -rf mc_tasks.egg-info
pip install -e .

python scripts/rsl_rl/train.py \
  --task Isaac-MC-Velocity-Flat-v0 \
  --num_envs 1 \
  --max_iterations 10\
  --headless

python scripts/rsl_rl/train.py \
  --task Isaac-MC-Velocity-Flat-v0 \
  --num_envs 4096 --headless \
  --max_iterations 2000


python scripts/rsl_rl/play_6d.py  \
  --task Isaac-MC-Velocity-Flat-Play-v0  \
  --num_envs 8 \
  --checkpoint logs/rsl_rl/mc_velocity_flat/2026-05-05_18-29-54/model_4999.pt

python scripts/rsl_rl/play.py  \
  --task Isaac-MC-Velocity-Flat-Play-v0 \
  --num_envs 1 \
  --checkpoint logs/rsl_rl/mc_velocity_flat/2026-07-19_14-44-01/model_9999.pt

python scripts/rsl_rl/play_vel_plot.py  \
  --task Isaac-MC-Velocity-Flat-Play-v0  \
  --num_envs 1 \

python scripts/rsl_rl/play.py  \
  --task Isaac-MC-Velocity-Flat-Play-v0  \
  --num_envs 16 \
  --checkpoint logs/rsl_rl/mc_velocity_flat/2026-08-07_13-03-35/model_1000.pt


python scripts/rsl_rl/play.py \
  --task Isaac-MC-Velocity-Flat-Play-v0 \
  --num_envs 1 --video --headless \
  --video_length 1000 


python scripts/rsl_rl/play.py \
  --task Isaac-MC-Velocity-Flat-v0 \
  --num_envs 1 --anim_recording_enabled \
  --anim_recording_start_time 0 --anim_recording_stop_time 10

python scripts/rsl_rl/play.py \
  --task Isaac-MC-Velocity-Flat-v0 \
  --num_envs 1 --anim_recording_enabled \
  --anim_recording_start_time 0 \
  --anim_recording_stop_time 10

tensorboard --logdir logs/rsl_rl/mc_velocity_flat


################ MC-P2P-Rel-Time #########################
pip uninstall mc-tasks -y
rm -rf mc_tasks.egg-info
pip install -e .

python scripts/rsl_rl/train.py \
  --task Isaac-MC-P2P_Rel_Time-v0 \
  --num_envs 1 \
  --max_iterations 10\
  --headless

python scripts/rsl_rl/train.py \
  --task Isaac-MC-P2P_Rel_Time-v0 \
  --num_envs 4096 --headless \
  --max_iterations 2000

python scripts/rsl_rl/play_p2p.py  \
  --task Isaac-MC-P2P_Rel_Time-Play-v0  \
  --num_envs 8 \
  --checkpoint logs/rsl_rl/mc_p2p_rel_time_rough/2026-05-05_18-29-54/model_4999.pt


python scripts/rsl_rl/play_6d.py  \
  --task Isaac-MC-P2P_Rel_Time-Play-v0  \
  --num_envs 8 \
  --checkpoint logs/rsl_rl/mc_p2p_rel_time_rough/2026-05-05_18-29-54/model_4999.pt

python scripts/rsl_rl/play_p2p_time_single.py  \
  --task Isaac-MC-P2P_Rel_Time-Play-v0  \
  --num_envs 1 \
  --checkpoint logs/rsl_rl/mc_p2p_rel_time_rough/2026-07-19_14-44-01/model_9999.pt

python scripts/rsl_rl/play_p2p_time_single3_plot.py  \
  --task Isaac-MC-P2P_Rel_Time-Play-v0  \
  --num_envs 1 \
  --plot_mode live # or both or save

python scripts/rsl_rl/play_p2p_time_single3_plot.py  \
  --task Isaac-MC-P2P_Rel_Time-Play-v0  \
  --num_envs 1 \
  --plot_mode live \
  --checkpoint logs/rsl_rl/mc_p2p_rel_time_rough/2026-07-30_11-49-54/model_150.pt



python scripts/rsl_rl/play_p2p_time_single.py \
  --task Isaac-MC-P2P_Rel_Time-Play-v0 \
  --num_envs 1 --video --headless \
  --video_length 1000 


python scripts/rsl_rl/play_p2p_time_single.py \
  --task Isaac-MC-P2P_Rel_Time-Play-v0 \
  --num_envs 1 --anim_recording_enabled \
  --anim_recording_start_time 0 --anim_recording_stop_time 10

python scripts/rsl_rl/play_p2p_time_single.py \
  --task Isaac-MC-P2P_Rel_Time-Play-v0 \
  --num_envs 1 --anim_recording_enabled \
  --anim_recording_start_time 0 \
  --anim_recording_stop_time 10


tensorboard --logdir logs/rsl_rl/mc_p2p_rel_time_rough



################ GO2  #########################

python scripts/reinforcement_learning/rsl_rl/play.py  \
  --task Isaac-Velocity-Flat-Unitree-Go2-Play-v0  \
  --num_envs 8 \
  --checkpoint logs/rsl_rl/unitree_go2_flat/2026-05-01_18-38-44/model_9999.pt


# Train spot
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py --task Isaac-Velocity-Flat-Spot-v0 \
    --num_envs 2048 --headless --experiment_name spot_vel --max_iterations 10000



./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-RelPose-Flat-Unitree-Go1-v0 \
  --num_envs 2048 --headless 
  --experiment_name go1_relpose_time_opt



#View training
cd ~/IsaacLab2.3.0/logs/rsl_rl/go1_relpose_time_opt
tensorboard --logdir .

cd ~/IsaacLab2.3.0/logs/rsl_rl/unitree_go1_flat
tensorboard --logdir .

./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
  --task Isaac-RelPose-Flat-Unitree-Go1-Play-v0 \
  --num_envs 16 \
  --checkpoint logs/rsl_rl/unitree_go1_flat/2025-12-01_16-36-29/model_1499.pt
  
  ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
  --task Isaac-RelPose-Flat-Unitree-Go1-Play-v0 \
  --num_envs 1 \
  --checkpoint logs/rsl_rl/unitree_go1_flat/2025-12-01_16-36-29/model_1499.pt



  ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py  \
   --task Isaac-RelPose-Flat-Unitree-Go1-Play-v0   --num_envs 1 \
    --checkpoint logs/rsl_rl/unitree_go1_flat/2025-12-08_09-56-09/model_2950.pt

./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
  --task Isaac-RelPose-Flat-Unitree-Go1-Play-v0   --num_envs 1 \
   --checkpoint logs/rsl_rl/unitree_go1_flat/2026-03-17_15-46-58/model_300.pt


# Keyboard control
./isaaclab.sh -p scripts/environments/teleoperation/teleop_se3_agent.py \
  --task Isaac-Velocity-Flat-Unitree-Go1-v0 \
  --num_envs 1 \
  --checkpoint logs/rsl_rl/unitree_go1_flat/model_go1_vel.pt




#Approach 2
Just use veolcity command and preplan the path

  ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play_trajectory.py \
  --task Isaac-Velocity-Flat-Unitree-Go1-Play-v0 \
  --num_envs 1 \
  --checkpoint logs/rsl_rl/unitree_go1_flat/model_go1_vel.pt




python scripts/rsl_rl/train.py   --task Isaac-Go1-P2P-v0   --num_envs 4096 --headless

tensorboard --logdir logs/rsl_rl/unitree_go1_flat


python scripts/rsl_rl/play.py   --task Isaac-Go1-P2P-Play-v0   --num_envs 8   --checkpoint logs/rsl_rl/go1_p2p_rough/2026-04-25_19-15-38/model_1050.pt

