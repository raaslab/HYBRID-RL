import gym
import metaworld
import numpy as np
from scipy.spatial.transform import Rotation as R
import random

def get_parameters(camera_id):
    # Load the benchmark (using ML1 for single-task environments)
    ml1 = metaworld.ML1('stick-pull-v2')  # load training environments
    env = ml1.train_classes['stick-pull-v2']()  # Create an environment instance
    task = random.choice(ml1.train_tasks)
    env.set_task(task)

    # Initialize the environment
    env.reset()
    sim = env.sim

    # Extrinsic Parameters
    cam_pos = sim.model.cam_pos[camera_id]  # Camera position
    print("cam_pos",cam_pos)
    cam_quat = sim.model.cam_quat[camera_id]  # Camera quaternion
    print("cam_quat",cam_quat)
    # Convert quaternion to rotation matrix
    rotation_matrix = R.from_quat(cam_quat).as_matrix()
    print("rotation_matrix",rotation_matrix)
    # correction_matrix = R.from_euler('x', 45, degrees=True).as_matrix()
    
    R_inverse = rotation_matrix.T
    print("rotation_matrix",rotation_matrix)
    cam_pos_inv = -R_inverse @ cam_pos
    print("cam_pos_inv",cam_pos_inv)
    # The extrinsic matrix is [R|t], where t is the translation vector (camera position)
    extrinsic_matrix = np.eye(4)
    extrinsic_matrix[:3, :3] = rotation_matrix
    extrinsic_matrix[:3, 3] = cam_pos_inv

    # Intrinsic Parameters
    width = sim.model.vis.global_.offwidth  # width from visualization settings
    height = sim.model.vis.global_.offheight  # height from visualization settings
    aspect_ratio = width / height
    # breakpoint()
    fov_y = np.radians(sim.model.cam_fovy[camera_id])  # Field of view in y-axis
    
    fov_x = 2 * np.arctan(np.tan(fov_y / 2) * aspect_ratio)
    # breakpoint()
    focal_length_y = (height / 2) / np.tan(fov_y / 2) # Focal length


    focal_length_x = (width / 2) / np.tan(fov_x / 2) # Focal length
    cx, cy = width / 2, height / 2  # Principal points (center of image)

    # Intrinsic matrix: assume no skew, and aspect ratio is 1
    intrinsic_matrix = np.array([
        [focal_length_x, 0, cx],
        [0, focal_length_y, cy],
        [0, 0, 1]
    ])

    print("Extrinsic Matrix:\n", extrinsic_matrix)
    print("Intrinsic Matrix:\n", intrinsic_matrix)


    return extrinsic_matrix, intrinsic_matrix
