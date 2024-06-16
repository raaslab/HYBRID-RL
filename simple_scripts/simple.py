import metaworld
import numpy as np
import random
import time
import matplotlib.pyplot as plt
import torch
import h5py

# Initialize the environment
mt1 = metaworld.MT1("stick-pull-v2")
env = mt1.train_classes["stick-pull-v2"]()
task = random.choice(mt1.train_tasks)
env.set_task(task)
obs = env.reset()
env.render()  # This call should initialize the viewer

# Load waypoints from hdf5 file
hdf5_path = '/home/amisha/ibrl/release/data/metaworld/StickPull_frame_stack_1_96x96_end_on_success/dataset.hdf5'
waypoints = []
gripper_controls = []

with h5py.File(hdf5_path, 'r') as f:
    for demo_name in f["data"].keys():
        demo_group = f["data"][demo_name]
        if demo_name == "demo_4":  # specify or loop through demos if needed
            prop_data = np.array(demo_group["obs"]["prop"][:34+1])
            waypoints = prop_data[:, :-1]  # Exclude the last column for gripper control
            gripper_controls = prop_data[:, -1]  # Last column for gripper control

max_steps = 100000
kp = 4.0

def run(waypoints, gripper_controls, obs):
    for waypoint, gripper_control in zip(waypoints, gripper_controls):
        for _ in range(max_steps):
            if env.viewer is not None:  # Additional check after attempting to render
                env.viewer.cam.fixedcamid = 5
                env.viewer.cam.type = 2  # Using fixed camera type
            env.render()

            current_pos = obs[:3]  # Assuming the first 3 values of obs are x, y, z coordinates
            error = waypoint - current_pos
            control_action = kp * error
            action = np.zeros(4)
            action[:3] = control_action
            action[3] = gripper_control  # Control the gripper

            action = np.clip(action, env.action_space.low, env.action_space.high)
            obs, reward, done, info = env.step(action)
            print("Dones: ", done)
            print("Current Error Norm: ", torch.norm(torch.tensor(error)))

            if np.linalg.norm(error) < 0.04:
                print("Reached the waypoint!")
                break

            time.sleep(0.08)

run(waypoints, gripper_controls, obs)

time.sleep(20)
env.close()
