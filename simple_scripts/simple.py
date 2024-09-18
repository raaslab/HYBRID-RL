# import metaworld
# import numpy as np
# import random
# import time
# import matplotlib.pyplot as plt
# import torch
# import h5py

# print(metaworld.ML1.ENV_NAMES)  # Check out the available environments

# # Initialize the environment
# mt1 = metaworld.MT1("disassemble-v2")
# env = mt1.train_classes["disassemble-v2"]()
# # mt1 = metaworld.MT1("assembly-v2")
# # env = mt1.train_classes["assembly-v2"]()
# task = random.choice(mt1.train_tasks)
# env.set_task(task)
# obs = env.reset()
# env.render()  # This call should initialize the viewer

# # Load waypoints from hdf5 file
# hdf5_path = '/home/amisha/ibrl/release/data/metaworld/Assembly_frame_stack_1_96x96_end_on_success/dataset.hdf5'
# waypoints = []
# gripper_controls = []

# with h5py.File(hdf5_path, 'r') as f:
#     for demo_name in f["data"].keys():
#         demo_group = f["data"][demo_name]
#         if demo_name == "demo_4":  # specify or loop through demos if needed
#             prop_data = np.array(demo_group["obs"]["prop"][:34+1])
#             waypoints = prop_data[:, :-1]  # Exclude the last column for gripper control
#             gripper_controls = prop_data[:, -1]  # Last column for gripper control

# max_steps = 100000
# kp = 4.0

# def run(waypoints, gripper_controls, obs):
#     for waypoint, gripper_control in zip(waypoints, gripper_controls):
#         for _ in range(max_steps):
#             # if env.viewer is not None:  # Additional check after attempting to render
#                 # env.viewer.cam.fixedcamid = 2
#                 # env.viewer.cam.type = 2  # Using fixed camera type
#             env.render()

#             current_pos = obs[:3]  # Assuming the first 3 values of obs are x, y, z coordinates
#             error = waypoint - current_pos
#             control_action = kp * error
#             action = np.zeros(4)
#             action[:3] = control_action
#             action[3] = gripper_control  # Control the gripper

#             action = np.clip(action, env.action_space.low, env.action_space.high)
#             obs, reward, done, info = env.step(action)
#             print("Dones: ", done)
#             print("Current Error Norm: ", torch.norm(torch.tensor(error)))

#             if np.linalg.norm(error) < 0.04:
#                 print("Reached the waypoint!")
#                 break

#             time.sleep(0.08)

# run(waypoints, gripper_controls, obs)

# time.sleep(10)
# env.close()

import metaworld
import numpy as np
import random
import time

# Initialize the MetaWorld environment
mt1 = metaworld.MT1("sweep-into-v2")  # Change to any task you're interested in
env = mt1.train_classes["sweep-into-v2"]()
task = random.choice(mt1.train_tasks)
env.set_task(task)
obs = env.reset()

# Define the camera ID to use
camera_id = 5

# Function to render the environment from a camera view
def render_views(env, camera_id):
    if env.viewer is not None:
        env.viewer.cam.fixedcamid = camera_id
        env.viewer.cam.type = 2  # Using fixed camera type
        print("Rendering Camera View", camera_id)

    start_time = time.time()
    while time.time() - start_time < 100:  # Keep the viewer open for 10 seconds
        env.render()
        time.sleep(0.05)  # Sleep to reduce CPU usage, adjust as needed

# Render the environment using the specified camera view
render_views(env, camera_id)

env.close()

