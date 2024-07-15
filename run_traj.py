import datetime
import glob
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import tyro

import csv
import numpy as np
import pandas as pd

from zmq_core.robot_node import ZMQClientRobot
from robots.robot import Robot
from env.env import RobotEnv

@dataclass
class Args:
    agent: str = "none"
    wrist_camera_port: int = 5000
    base_camera_port: int = 5001
    
    hostname: str = "192.168.77.243"
    robot_port: int = 50003  # for trajectory
    robot_ip: str = "192.168.77.21" 
    robot_type: str = None  # only needed for quest agent or spacemouse agent
    hz: int = 100
    start_pose: Optional[Tuple[float, ...]] = None

    gello_port: Optional[str] = None
    mock: bool = False
    use_save_interface: bool = False
    data_dir: str = "~/bc_data"
    bimanual: bool = False
    verbose: bool = False

def main(args):
    camera_clients = {
    # you can optionally add camera nodes here for imitation learning purposes
    # "wrist": ZMQClientCamera(port=args.wrist_camera_port, host=args.hostname),
    # "base": ZMQClientCamera(port=args.base_camera_port, host=args.hostname),
    }
    robot_client = ZMQClientRobot(port=args.robot_port, host=args.hostname)
    env = RobotEnv(robot_client, control_rate_hz=args.hz, camera_dict=camera_clients)

    traj_file = "/home/sj/Assistive_Feeding_Gello/csv/90close/test/output18.csv"

    if args.agent == "ur":
        end_eff_pos_data = pd.read_csv(traj_file, skipfooter=1, usecols=range(7, 10), engine='python').astype(np.float64)
        end_eff_quat_data = pd.read_csv(traj_file, skipfooter=1, usecols=range(10, 13), engine='python').astype(np.float64)
        end_eff_pos_data = np.concatenate((end_eff_pos_data, end_eff_quat_data), axis=1)
        # end_eff_pos_data = np.array([-0.12230709501381903,-0.28115410794389206,0.29961265005846593,-2.2113353082307734,-2.218063376173432,-0.051748804815462957])
        print("CSV read successfully")

    # print(end_eff_pos_data)

    if args.start_pose is None:
        print('in if condition')
        reset_pose = (end_eff_pos_data[0]) # Change this to your own reset joints
    else:
        reset_pose = np.array(args.start_pose)

    print("reset_pose", reset_pose)
    curr_pose = env.get_obs()["ee_pos_quat"]
    
    print("Current Pose", curr_pose)

    curr_pose = np.array(curr_pose)

    print("len(reset_pose)", len(reset_pose), "len(curr_pose)", len(curr_pose))
    if len(reset_pose) == len(curr_pose):
        max_delta = (np.abs(curr_pose - np.array(reset_pose))).max()
        print("max_delta", max_delta)   
        steps = min(int(max_delta / 0.001), 100)

        for pose in np.linspace(curr_pose[:3], reset_pose[:3], steps):
            pose = np.concatenate((pose, curr_pose[3:]))
            print("pose", pose)
            env.update_desired_ee_pose(pose)
            time.sleep(0.001)


if __name__ == "__main__":
    main(tyro.cli(Args))