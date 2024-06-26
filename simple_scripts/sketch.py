import metaworld
import numpy as np
import random
import time

import torch
from scipy.spatial.transform import Rotation as R
import cv2
import h5py
import camera_param
sensed_gripper_positions = []
target_gripper_positions = []
# Flag to track if the gripper has opened for the first time
gripper_opened = False
# Initialize the environment
mt1 = metaworld.MT1("stick-pull-v2")
env = mt1.train_classes["stick-pull-v2"]()
task = random.choice(mt1.train_tasks)
env.set_task(task)
obs = env.reset()
first_frame = env.render("rgb_array")  # Capture the first frame
# Load waypoints and gripper controls from HDF5
hdf5_path = 'release/data/metaworld/StickPull_frame_stack_1_96x96_end_on_success/dataset.hdf5'
waypoints = []
gripper_controls = []

with h5py.File(hdf5_path, 'r') as f:
    for demo_name in f["data"].keys():
        demo_group = f["data"][demo_name]
        if demo_name == "demo_0":  # adjust as needed
            prop_data = np.array(demo_group["obs"]["prop"])
            waypoints = prop_data[:, :-1]  # All but last for positions
            gripper_controls = prop_data[:, -1]  # Last column for gripper state

max_steps = 100000
kp = 5.0
camera_id = 2     # we want camera_id 2
camera_extrinsic, camera_intrinsic = camera_param.get_parameters(camera_id=camera_id)

trajectory_2d = []

def run(waypoints, gripper_controls, obs):
    for waypoint, gripper_control in zip(waypoints, gripper_controls):
        for step in range(max_steps):
            # Extract sensed and target gripper joint positions
            sensed_gripper_position = obs[-1]  # Assuming the last element of obs is the gripper joint position
            target_gripper_position = gripper_control
            sensed_gripper_positions.append(sensed_gripper_position)
            target_gripper_positions.append(target_gripper_position)
            if env.viewer is not None:
                env.viewer.cam.fixedcamid = camera_id
                env.viewer.cam.type = 2
                # env.viewer.cam.azimuth = 180
            img = env.render("rgb_array")
            # img = cv2.resize(img, (300, 300))
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            cv2.imshow("Rotated View", rgb_img)
            if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit the display before the end of the loop
                break

            current_pos = obs[:3]
            error = waypoint - current_pos
            control_action = kp * error
            action = np.zeros(4)
            action[:3] = control_action
            action[3] = gripper_control
            action = np.clip(action, env.action_space.low, env.action_space.high)
            obs, reward, done, info = env.step(action)

            # Project the 3D end-effector position to the camera space
            end_effector_pos_3d = obs[:3]

            # Apply the coordinate transformation
            cv_to_mujoco_transform = np.array([[0, 1, 0],
                                               [1, 0, 0],
                                               [0, 0, -1]])
            end_effector_pos_3d = cv_to_mujoco_transform @ end_effector_pos_3d

            end_effector_pos_camera = camera_extrinsic @ np.concatenate((end_effector_pos_3d, [1]))[:, None]
            print("end_effector_pos_camera",end_effector_pos_camera)
            # breakpoint()
            end_effector_pos_camera = end_effector_pos_camera / end_effector_pos_camera[3]
            print("end_effector_pos_camera_2",end_effector_pos_camera)
            end_effector_pos_2d = camera_intrinsic @ end_effector_pos_camera[:3]
            end_effector_pos_2d =end_effector_pos_2d[:2] / end_effector_pos_2d[2]
            print("end_effector_pos_2d",end_effector_pos_2d)

            
            image_width = env.sim.model.vis.global_.offwidth
            image_height = env.sim.model.vis.global_.offheight
            fx, fy = camera_intrinsic[0, 0], camera_intrinsic[1, 1]
            cx, cy = camera_intrinsic[0, 2], camera_intrinsic[1, 2]
            # x_pixel = (end_effector_pos_2d[0] / end_effector_pos_2d[2]) * fx + cx
            # y_pixel = (end_effector_pos_2d[1] / end_effector_pos_2d[2]) * fy + cy

            # Ensure coordinates are within image dimensions
            x_pixel = np.clip(end_effector_pos_2d[0], 0, image_width - 1)
            y_pixel = np.clip(end_effector_pos_2d[1], 0, image_height - 1)
            
            scaled_end_effector_pos_2d = np.array([x_pixel, y_pixel])
            print("scaled_end_effector_pos_2d",scaled_end_effector_pos_2d)
            trajectory_2d.append(scaled_end_effector_pos_2d[:2])
            
            if np.linalg.norm(error) < 0.04:
                # print("Reached the waypoint!")
                break
            time.sleep(0.01)




run(waypoints, gripper_controls, obs)
# trajectory_2d = normalize_trajectory(trajectory_2d)
# print(trajectory_2d)

# Threshold for closing action
epsilon = 0.5  # Adjust this value as needed

# Lists to store the key time steps for closing and opening the gripper
closing_steps = []
opening_steps = []


trajectory_2d = np.array(trajectory_2d)

# Overlay the trajectory on the first frame
first_frame_rgb = cv2.cvtColor(first_frame[:, ::-1, :], cv2.COLOR_BGR2RGB)
height, width, _ = first_frame_rgb.shape

trajectory_2d_rot = trajectory_2d


# Color grading for relative temporal motion
trajectory_img = np.zeros((height, width, 3), dtype=np.uint8)
for i in range(len(trajectory_2d_rot) - 1):
    p1 = (int(width - trajectory_2d_rot[i, 0, 0]), int(trajectory_2d_rot[i, 1, 0]))
    p2 = (int(width - trajectory_2d_rot[i+1, 0, 0]), int(trajectory_2d_rot[i+1, 1, 0]))

    # Get the normalized time step and height for the current point
    normalized_time_step = (i+1) / len(trajectory_2d_rot)
    normalized_height = (obs[2] - 0.05) / (0.3 - 0.05)
    cv2.line(trajectory_img, p1, p2, (0, int(255 * normalized_height), int(1000 * normalized_time_step)), 2)
    # cv2.line(trajectory_img, p1, p2, (0, 0, int(1000 * (i+1) / len(trajectory_2d_rot))), 5)

# Blend the trajectory with the first frame
# trajectory_img=trajectory_img[:, ::-1, :]
blended_img = cv2.addWeighted(first_frame_rgb, 0.5, trajectory_img, 0.5, 0)

# Iterate through the sensed and target gripper joint positions
for i in range(len(sensed_gripper_positions)):
    sensed_position = sensed_gripper_positions[i]
    target_position = target_gripper_positions[i]
    delta = target_position - sensed_position

    # Check for opening action
    if delta < 0 or target_position <= epsilon:
        if not gripper_opened:
            opening_steps.append(i)
            gripper_opened = True

    # Check for closing action
    if delta > 0 and target_position > epsilon:
        if gripper_opened and (i == 0 or (i > 0 and (sensed_gripper_positions[i-1] - target_gripper_positions[i-1]) <= 0)):
            closing_steps.append(i)

# Draw interaction markers on the first frame
for step in opening_steps:
    p1 = (int(width - trajectory_2d_rot[step, 0, 0]), int(trajectory_2d_rot[step, 1, 0]))
    cv2.circle(blended_img, p1, 5, (255, 0, 0), -1)  # Blue circles for opening action

for step in closing_steps:
    p1 = (int(width - trajectory_2d_rot[step, 0, 0]), int(trajectory_2d_rot[step, 1, 0]))
    cv2.circle(blended_img, p1, 5, (0, 255, 0), -1)  # Green circles for closing action



# Display the overlaid image
cv2.imshow("Overlaid Trajectory", blended_img)
cv2.waitKey(0)
cv2.destroyAllWindows()


time.sleep(4)
env.close()