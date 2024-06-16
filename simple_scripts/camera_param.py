import gym
import metaworld
import numpy as np
from scipy.spatial.transform import Rotation as R
import random

def get_parameters(camera_id):
    # Load the benchmark (using ML1 for single-task environments)
    ml1 = metaworld.ML1('assembly-v2')  # load training environments
    env = ml1.train_classes['assembly-v2']()  # Create an environment instance
    task = random.choice(ml1.train_tasks)
    env.set_task(task)

    # Initialize the environment
    env.reset()
    sim = env.sim

    # Get default camera settings, assuming the default camera
    # camera_id = -1  # Assuming the default camera

    # Extrinsic Parameters
    cam_pos = sim.model.cam_pos[camera_id]  # Camera position
    cam_quat = sim.model.cam_quat[camera_id]  # Camera quaternion

    # Convert quaternion to rotation matrix
    rotation_matrix = R.from_quat(cam_quat).as_matrix()

    # # Flip the camera view upside down by applying a 180-degree rotation around the x-axis
    flip_matrix = np.eye(4)
    flip_matrix[:3, :3] = R.from_rotvec([np.pi, 0, 0]).as_matrix()
    rotation_matrix = flip_matrix[:3, :3] @ rotation_matrix



    # The extrinsic matrix is [R|t], where t is the translation vector (camera position)
    extrinsic_matrix = np.eye(4)
    extrinsic_matrix[:3, :3] = rotation_matrix
    extrinsic_matrix[:3, 3] = cam_pos

    # Intrinsic Parameters
    width = sim.model.vis.global_.offwidth  # width from visualization settings
    height = sim.model.vis.global_.offheight  # height from visualization settings
    fov_y = sim.model.cam_fovy[camera_id]  # Field of view in y-axis
    focal_length = (height / 2) / np.tan(np.radians(fov_y / 2))  # Focal length
    cx, cy = width / 2, height / 2  # Principal points (center of image)

    # Intrinsic matrix: assume no skew, and aspect ratio is 1
    intrinsic_matrix = np.array([
        [focal_length, 0, cx],
        [0, focal_length, cy],
        [0, 0, 1]
    ])

    print("Extrinsic Matrix:\n", extrinsic_matrix)
    print("Intrinsic Matrix:\n", intrinsic_matrix)

    return extrinsic_matrix, intrinsic_matrix
