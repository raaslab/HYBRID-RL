from dataclasses import dataclass, field
import numpy as np
import torch
from scipy.spatial.transform import Rotation

from env.mocks import MockGripper, MockRobot
from env.lift import LiftEEConfig
from env.drawer import DrawerEEConfig
from env.hang import HangEEConfig
from env.towel import TowelEEConfig

from robots.ur import URRobot
from zmq_core.robot_node import ZMQClientRobot
from env.env import RobotEnv
from robots.robotiq_gripper import RobotiqGripper

import pyrallis
try:
    import rtde_control
    import rtde_receive

    URTDE_IMPORTED = True
except ImportError:
    print("[robots] Skipping URTDE")
    URTDE_IMPORTED = False


class ActionSpace:
    def __init__(self, low: list[float], high: list[float]):
        self.low = np.array(low, dtype=np.float32)
        self.high = np.array(high, dtype=np.float32)

    def assert_in_range(self, actionl: list[float]):
        action: np.ndarray = np.array(actionl)

        correct = (action <= self.high).all() and (action >= self.low).all()
        if correct:
            return True

        for i in range(self.low.size):
            check = action[i] >= self.low[i] and action[i] <= self.high[i]
            print(f"{self.low[i]:.2f} <= {action[i]:.2f} <= {self.high[i]:.2f}? {check}")
        return False

    def clip_action(self, action: np.ndarray):
        clipped = np.clip(action, self.low, self.high)
        return clipped


@dataclass
class URTDEControllerConfig:
    task: str = "drawer"
    # robot_ip should be local because this runs on the nuc that connects to the robot
    robot_ip_address: str = "localhost"
    controller_type: str = "CARTESIAN_DELTA"
    max_delta: float = 0.05
    mock: int = 0


class URTDEController:
    """Controller run on server.

    All parameters should be python native for easy integration with 0rpc
    """

    # # Define the bounds for the Franka Robot
    JOINT_LOW = np.array(
        [-2.6172, -1.832, -1.832, -1.832, 1.046, 1.046, 0], dtype=np.float32
    )
    JOINT_HIGH = np.array(
        [-0.5228, -1.308, -1.308, -1.308, 2.094, 2.094, 1], dtype=np.float32
    )

    def __init__(self, cfg: URTDEControllerConfig) -> None:
        self.cfg = cfg
        assert self.cfg.controller_type in {
            "CARTESIAN_DELTA",
            "CARTESIAN_IMPEDANCE",
        }

        if cfg.task == "lift":
            self.ee_config = LiftEEConfig()
        elif cfg.task == "drawer":  
            self.ee_config = DrawerEEConfig()
        elif cfg.task == "hang":
            self.ee_config = HangEEConfig()
        elif cfg.task == "towel":
            self.ee_config = TowelEEConfig()
        else:
            assert False, "unknown env"

        self.action_space = ActionSpace(*self.get_action_space())

        if self.cfg.mock:
            self._robot = MockRobot()
            self._gripper = MockGripper()
        else:
            assert URTDE_IMPORTED, "Attempted to load robot without URTDE package."
            robot_client = ZMQClientRobot(port=args.robot_port, host=args.hostname)
            self._robot = RobotEnv(robot_client, control_rate_hz=args.hz, camera_dict=args.camera_clients)
            self._gripper = RobotiqGripper()
            self._gripper.connect(hostname=args.robot_ip, port=63352)
            print("gripper connected")

        self._robot.set_home_pose(self.ee_config.home)
        self._robot.go_home(blocking=False)
        print("home position", self.ee_config.home)

        # self._go_home(self.ee_config.home)
        ee_pos = self._robot.get_ee_pose()
        print("current ee pos:", ee_pos)

        if hasattr(self._gripper, "_max_position") :
            # Should grab this from robotiq2f
            self._max_gripper_width = self._gripper._max_position
        else:
            self._max_gripper_width = 255  # default, from Robotiq Value

        self.desired_gripper_qpos = 255

    def hello(self):
        return "hello"

    def get_action_space(self) -> tuple[list[float], list[float]]:
        if self.cfg.controller_type == "CARTESIAN_DELTA":
            high = [self.cfg.max_delta] * 3 + [self.cfg.max_delta * 4] * 3
            low = [-x for x in high]
        elif self.cfg.controller_type == "CARTESIAN_IMPEDANCE":
            low = self.ee_config.ee_range_low
            high = self.ee_config.ee_range_high
        else:
            raise ValueError("Invalid Controller type provided")

        # Add the gripper action space
        low.append(0.0)
        high.append(1.0)
        return low, high

    def update_gripper(self, gripper_action: float, blocking=False) -> None:
        # We always run the gripper in absolute position
        gripper_action = max(min(gripper_action, 1), 0)
        width = self._max_gripper_width * (1 - gripper_action)

        self.desired_gripper_qpos = gripper_action

        self._gripper.move(
            position=width,
            speed=0.1,
            force=0.01,
        )

    def update(self, action: list[float]) -> None:
        """
        Updates the robot controller with the action
        """
        assert len(action) == 6, f"wrong action dim: {len(action)}"
        # assert self._robot.is_running_policy(), "policy not running"

        '''
        # if not self._robot.is_running_policy():
        #     print("restarting cartesian impedance controller")
        #     self._robot.start_cartesian_impedance()
        #     time.sleep(1)
        '''
        assert self.action_space.assert_in_range(action)

        robot_action: np.ndarray = np.array(action[:-1])
        gripper_action: float = action[-1]

        if self.cfg.controller_type == "CARTESIAN_DELTA":
            pos = self._robot.get_ee_pose()
            ee_pos, ee_ori = np.split(pos, [3])
            delta_pos, delta_ori = np.split(robot_action, [3])

            # compute new pos and new quat
            new_pos = ee_pos + delta_pos
            # TODO: this can be made much faster using purpose build methods instead of scipy.
            new_rot = (delta_ori * ee_ori).astype(np.float32)

            # clip
            new_pos, new_rot = self.ee_config.clip(new_pos, new_rot)
            end_eff_pos = np.concatenate((new_pos, new_rot))
            self._robot.update_desired_ee_pose(end_eff_pos)

        else:
            raise ValueError("Invalid Controller type provided")

        # Update the gripper
        self.update_gripper(gripper_action, blocking=False)

    def _go_home(self, home):
        print("going home")
        self._robot.set_home_pose(home)
        self._robot.go_home(blocking=False)

    def reset(self, randomize: bool) -> None:
        print("reset env")

        self.update_gripper(0)  # open the gripper

        self.ee_config.reset(self)

        # if self._robot.is_running_policy():
        #     self._robot.terminate_current_policy()

        home = self.ee_config.home
        if randomize:
            # TODO: adjust this noise
            high = 0.1 * np.ones_like(home)
            noise = np.random.uniform(low=-high, high=high)
            print("home noise:", noise)
            home = home + noise

        self._robot.go_home(blocking=False)

        # assert not self._robot.is_running_policy()
        # self._robot.start_cartesian_impedance()
        time.sleep(1)

    def get_state(self) -> dict[str, list[float]]:
        """
        Returns the robot state dictionary.
        For VR support MUST include [ee_pos, ee_quat]
        """
        ee_pos, ee_quat = self._robot.get_ee_pose()
        gripper_state = self._gripper.get_current_position()
        gripper_pos = 1 - (gripper_state / self._max_gripper_width) # 0 is open and 1 is closed

        state = {
            "robot0_eef_pos": list(ee_pos),
            "robot0_eef_quat": list(ee_quat),
            "robot0_gripper_qpos": [gripper_pos],
            "robot0_desired_gripper_qpos": [self.desired_gripper_qpos],
        }

        return state

import datetime
import glob
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple


@dataclass
class PolyMainConfig:
    server: int = 1
    server_loc: str = "local"  # local/fr2
    controller: URTDEControllerConfig = field(
        default_factory=lambda: URTDEControllerConfig()
    )

    goto: str = ""
    goto_delta: int = 0

    def __post_init__(self):
        address_book = {
            "local": "tcp://192.168.77.243:50003",
            "ur3e": "tcp://192.168.77.21:50003",
        }
        if self.server:
            self.rpc_address = address_book["local"]
        else:
            self.rpc_address = address_book[self.server_loc]

@dataclass
class Args:
    agent: str = "ur"
    hostname: str = "192.168.77.243"
    robot_port: int = 50003  # for trajectory
    robot_ip: str = "192.168.77.21" 
    robot_type: str = None  # only needed for quest agent or spacemouse agent
    hz: int = 100
    start_pose: Optional[Tuple[float, ...]] = None

    use_save_interface: bool = False
    data_dir: str = "~/bc_data"
    verbose: bool = False
    camera_clients = {
    # you can optionally add camera nodes here for imitation learning purposes
    # "wrist": ZMQClientCamera(port=args.wrist_camera_port, host=args.hostname),
    # "base": ZMQClientCamera(port=args.base_camera_port, host=args.hostname),
    }
    controller: URTDEControllerConfig = field(
        default_factory=lambda: URTDEControllerConfig()
    )

if __name__ == "__main__":
    args = Args()
    
    cfg = pyrallis.parse(config_class=PolyMainConfig)  # type: ignore
    np.set_printoptions(precision=4, linewidth=100, suppress=True)

    controller = URTDEController(cfg.controller)

    # time.sleep(3)

    # robot = controller._robot

    # import pandas as pd

    # traj_file = "/home/sj/Assistive_Feeding_Gello/csv/90close/test/output18.csv"

    # if args.agent == "ur":
    #     end_eff_pos_data = pd.read_csv(traj_file, skipfooter=1, usecols=range(7, 10), engine='python').astype(np.float64)
    #     end_eff_quat_data = pd.read_csv(traj_file, skipfooter=1, usecols=range(10, 13), engine='python').astype(np.float64)
    #     end_eff_pos_data = np.concatenate((end_eff_pos_data, end_eff_quat_data), axis=1)
    #     # end_eff_pos_data = np.array([-0.12230709501381903,-0.28115410794389206,0.29961265005846593,-2.2113353082307734,-2.218063376173432,-0.051748804815462957])
    #     print("CSV read successfully")

    # robot.move_to_eef_positions(end_eff_pos_data[0], delta=False)

    # time.sleep(3)

    # controller.reset(randomize=False)