
import metaworld
import numpy as np
import random
import time
import matplotlib.pyplot as plt
import torch
from scipy.spatial.transform import Rotation as R
import cv2
import h5py
import camera_param

# Initialize the environment
mt1 = metaworld.MT1("assembly-v2")
env = mt1.train_classes["assembly-v2"]()
task = random.choice(mt1.train_tasks)
env.set_task(task)
obs = env.reset()
env.render("rgb_array")
first_frame = env.render("rgb_array")  # Capture the first frame
# Load waypoints and gripper controls from HDF5
hdf5_path = '/home/amisha/ibrl/release/data/metaworld/Assembly_frame_stack_1_96x96_end_on_success/dataset.hdf5'
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
            if env.viewer is not None:
                env.viewer.cam.fixedcamid = camera_id
                env.viewer.cam.type = 2
                # env.viewer.cam.azimuth = 180
            img = env.render("rgb_array")
            # img = cv2.resize(img, (224, 224))
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
            end_effector_pos_camera = camera_extrinsic @ np.concatenate((end_effector_pos_3d, [1]))[:, None]
            end_effector_pos_camera = end_effector_pos_camera[:3] / end_effector_pos_camera[3]
            end_effector_pos_2d = camera_intrinsic @ end_effector_pos_camera

            image_width = env.sim.model.vis.global_.offwidth
            image_height = env.sim.model.vis.global_.offheight
            fx, fy = camera_intrinsic[0, 0], camera_intrinsic[1, 1]
            cx, cy = camera_intrinsic[0, 2], camera_intrinsic[1, 2]
            scaled_x = (end_effector_pos_2d[0] / fx) * image_width + cx
            scaled_y = (end_effector_pos_2d[1] / fy) * image_height + cy
            scaled_end_effector_pos_2d = np.array([scaled_x, scaled_y])
            trajectory_2d.append(scaled_end_effector_pos_2d[:2])

            if np.linalg.norm(error) < 0.04:
                print("Reached the waypoint!")
                break
            time.sleep(0.1)




run(waypoints, gripper_controls, obs)
# trajectory_2d = normalize_trajectory(trajectory_2d)
print(trajectory_2d)


# Plot using Matplotlib
trajectory_2d = np.array(trajectory_2d)

print(trajectory_2d.shape)

# Rotation matrix for 180 degrees
rotation_matrix = np.array([[-1, 0], [0, -1]])

# Rotate points - 180 degrees
trajectory_2d_rot= []
for i in trajectory_2d:
    rot_point = np.dot(i.reshape(1, 2), rotation_matrix)
    trajectory_2d_rot.append(rot_point)
trajectory_2d_rot = np.array(trajectory_2d_rot).reshape(len(trajectory_2d_rot), 2, 1)
print(trajectory_2d_rot.shape)


# Overlay the trajectory on the first frame
first_frame_rgb = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)
height, width, _ = first_frame_rgb.shape
for i in range(len(trajectory_2d_rot) - 1):
    p1 = (int(trajectory_2d_rot[i, 0, 0]), int(trajectory_2d_rot[i, 1, 0]))
    p2 = (int(trajectory_2d_rot[i + 1, 0, 0]), int(trajectory_2d_rot[i + 1, 1, 0]))

    # Handle wraparound for x-coordinate
    if p1[0] < 0 and p2[0] >= 0:
        cv2.line(first_frame_rgb, (p1[0] + width, p1[1]), p2, (0, 255, 0), 5)
    elif p1[0] >= 0 and p2[0] < 0:
        cv2.line(first_frame_rgb, p1, (p2[0] + width, p2[1]), (0, 255, 0), 5)
    else:
        cv2.line(first_frame_rgb, p1, p2, (0, 255, 0), 5)

    # Handle wraparound for y-coordinate
    if p1[1] < 0 and p2[1] >= 0:
        cv2.line(first_frame_rgb, (p1[0], p1[1] + height), p2, (0, 255, 0), 5)
    elif p1[1] >= 0 and p2[1] < 0:
        cv2.line(first_frame_rgb, p1, (p2[0], p2[1] + height), (0, 255, 0), 5)
    else:
        cv2.line(first_frame_rgb, p1, p2, (0, 255, 0), 5)

# Display the overlaid image
cv2.imshow("Overlaid Trajectory", first_frame_rgb)
cv2.waitKey(0)
cv2.destroyAllWindows()

# plt.plot(trajectory_2d_rot[0, 0], trajectory_2d_rot[0, 1], 'o', markersize=5, color='red')      # Start point
# plt.plot(trajectory_2d_rot[-1, 0], trajectory_2d_rot[-1, 1], 'o', markersize=5, color='green')  # end point
# plt.plot(trajectory_2d_rot[:, 0], trajectory_2d_rot[:, 1])
# plt.show()

time.sleep(4)
env.close()
