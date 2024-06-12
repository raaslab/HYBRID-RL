import metaworld
import numpy as np
import random
import time
import matplotlib.pyplot as plt
import torch


# Initialize the environment
mt1 = metaworld.MT1("assembly-v2")
env = mt1.train_classes["assembly-v2"]()
task = random.choice(mt1.train_tasks)
env.set_task(task)
obs = env.reset()
# Define the waypoint and include an extra parameter for the fourth action dimension
waypoint = np.array([0.130, 0.600, 0.019])
max_steps = 1000
gripper_control = -1  # assuming the fourth dimension is for the gripper

def run(waypoint, obs):
    kp = 0.7
    for _ in range(max_steps):
        env.render()
        print(img.shape)
        current_pos = obs[:3]  # Assuming the first 3 values of obs are x, y, z coordinates
        error = waypoint - current_pos
        control_action = kp * error
        action = np.zeros(4)  # Now we need a vector of 4, the last for the gripper
        action[:3] = control_action
        action[3] = gripper_control  # Control the gripper, set as needed
        # Clip the action to ensure it’s within the action space limits
        action = np.clip(action, env.action_space.low, env.action_space.high)
        # Step the environment
        obs, reward, done, info = env.step(action)

        print("Dones: ", done)


        if np.linalg.norm(error) < 0.04:
            print("Reached the waypoint!")
            break
        # print(error)
        print(torch.norm(torch.tensor(error)))
        time.sleep(0.01)

run(waypoint, obs)

time.sleep(4)


env.close()