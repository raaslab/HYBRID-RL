from dataclasses import dataclass, field
import numpy as np
import torch
from scipy.spatial.transform import Rotation
import zerorpc

from env.mocks import MockGripper, MockRobot
from env.lift import LiftEEConfig
from env.drawer import DrawerEEConfig
from env.hang import HangEEConfig
from env.towel import TowelEEConfig

from robots.ur import URRobot
from zmq_core.robot_node import ZMQClientRobot
from env.env import RobotEnv

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")

print(sys.path)
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
            self._gripper = self._robot.gripper

        self._robot.set_home_pose(self.ee_config.home)
        self._robot.go_home(blocking=False)
        ee_pos = self._robot.get_ee_pose()
        print("current ee pos:", ee_pos)

        if hasattr(self._gripper, "_max_position") :
            # Should grab this from robotiq2f
            self._max_gripper_width = self._gripper._max_position
        else:
            self._max_gripper_width = 255  # default, from Robotiq Value

        self.desired_gripper_qpos = 0

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
        assert len(action) == 7, f"wrong action dim: {len(action)}"
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
            ee_pos, ee_quat = pos[:3], pos[3:]
            delta_pos, delta_ori = np.split(robot_action, [3])

            # compute new pos and new quat
            new_pos = ee_pos + delta_pos
            # TODO: this can be made much faster using purpose build methods instead of scipy.
            old_rot = Rotation.from_quat(ee_quat)
            delta_rot = Rotation.from_euler("xyz", delta_ori)
            new_rot = (delta_rot * old_rot).as_euler("xyz")

            # clip
            new_pos, new_rot = self.ee_config.clip(new_pos, new_rot)
            new_quat = Rotation.from_euler("xyz", new_rot).as_quat()  # type: ignore
            self._robot.update_desired_ee_pose(
                torch.from_numpy(new_pos).float(), torch.from_numpy(new_quat).float()
            )

        # '''
        # elif self.cfg.controller_type == "CARTESIAN_IMPEDANCE":
        #     pos, rot = np.split(robot_action, [3])
        #     pos, rot = self.ee_config.clip(pos, rot)

        #     ori = Rotation.from_euler("xyz", rot).as_quat()  # type: ignore
        #     self._robot.update_desired_ee_pose(
        #         torch.from_numpy(pos).float(), torch.from_numpy(ori).float()
        #     )
        # '''
        else:
            raise ValueError("Invalid Controller type provided")

        # Update the gripper
        self.update_gripper(gripper_action, blocking=False)

    def _go_home(self, home):
        print("going home")
        self._robot.set_home_pose(torch.from_numpy(home))
        self._robot.go_home(blocking=True)

        ee_pos, _ = self._robot.get_ee_pose()
        # ee_pos = ee_pos.numpy()
        if np.abs(ee_pos - self.ee_config.init_ee_pos).max() > 0.02:
            print("going home, 2nd try")
            # go home again
            self._robot.set_home_pose(torch.from_numpy(home))
            self._robot.go_home(blocking=True)

    def reset(self, randomize: bool) -> None:
        print("reset env")

        self.update_gripper(0)  # open the gripper

        self.ee_config.reset(self)

        if self._robot.is_running_policy():
            self._robot.terminate_current_policy()

        home = self.ee_config.home
        if randomize:
            # TODO: adjust this noise
            high = 0.1 * np.ones_like(home)
            noise = np.random.uniform(low=-high, high=high)
            print("home noise:", noise)
            home = home + noise

        self._go_home(home, blocking=False)

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


class URTDEControllerClient:
    def __init__(self, remote_ip_address, task, positional=False):
        self.positional = positional

        self.client = zerorpc.Client(heartbeat=20)
        self.client.connect(remote_ip_address)
        print(f"Connected? {self.client.hello()}")

        self.action_space = ActionSpace(*self.client.get_action_space())  # type: ignore
        self.action_dim = len(self.action_space.low)

        # TODO: ideally we should just have 1 ee_config on the server,
        # but we want to log & shift more compute on the workstation
        # TODO: de-duplicate
        if task == "lift":
            self.ee_config = LiftEEConfig()
        elif task == "drawer":
            self.ee_config = DrawerEEConfig()
        elif task == "hang":
            self.ee_config = HangEEConfig()
        elif task == "towel":
            self.ee_config = TowelEEConfig()
        else:
            assert False, "unknown env"

    def update(self, action: np.ndarray) -> None:
        if self.positional:
            self.client.update(action.tolist())
            return

        # delta actions need to be rescaled
        low = self.action_space.low
        high = self.action_space.high
        action = low + (0.5 * (action + 1.0) * (high - low))
        action = self.action_space.clip_action(action)
        # print("action from client update:", action)
        self.client.update(action.tolist())

    def reset(self, randomize: bool) -> None:
        print(f"controller randomize? {randomize}")
        self.client.reset(randomize)

    def get_state(self) -> tuple[dict[str, np.ndarray], bool]:
        state = self.client.get_state()

        for k, v in state.items():
            assert isinstance(v, list)
            state[k] = np.array(v, dtype=np.float32)

        in_good_range = self.ee_config.ee_in_good_range(
            state["robot0_eef_pos"], state["robot0_eef_quat"], True
        )
        return state, in_good_range


def goto(target, ip_address, task):
    client = URTDEControllerClient(ip_address, task, positional=True)
    client.reset(False)

    state, _ = client.get_state()
    print("current state")
    for k, v in state.items():
        print(k, v)

    curr_pos = state["robot0_eef_pos"]
    curr_rot = Rotation.from_quat(state["robot0_eef_quat"]).as_euler("xyz")
    curr = np.concatenate([curr_pos, curr_rot, state["robot0_gripper_qpos"]])

    dest = [float(x) for x in target.split(",")]
    if len(dest) == 3:
        dest.extend(curr_rot.tolist())
        dest.append(state["robot0_gripper_qpos"].item())
    else:
        assert False

    dest = np.array(dest).astype(np.float32)
    print("current:")
    print(curr)
    print("destination:")
    print(dest)

    num_points = 50
    alphas = np.linspace(0, 1, num=num_points)
    inter_dests = []
    for a in alphas:
        inter_dests.append(a * dest + (1 - a) * curr)

    for x in inter_dests:
        # print(x)
        client.update(x)
        time.sleep(0.1)

    state, _ = client.get_state()
    print("new state")
    for k, v in state.items():
        print(k, v)

    return


def goto_delta(ip_address, task):
    client = URTDEControllerClient(ip_address, task, positional=False)
    client.reset(False)

    state, in_good_range = client.get_state()
    print(f"current state, {in_good_range=}")
    for k, v in state.items():
        print(k, v)

    curr_pos = state["robot0_eef_pos"]
    curr_rot = Rotation.from_quat(state["robot0_eef_quat"]).as_euler("xyz")
    curr = np.concatenate([curr_pos, curr_rot, state["robot0_gripper_qpos"]])

    # EE_LOW = np.array([0.45, -0.15, 0.19, -np.pi, -np.pi, -np.pi], dtype=np.float32)
    # EE_HIGH = np.array([0.7, 0.15, 0.4, np.pi, np.pi, np.pi], dtype=np.float32)

    """
    action space meaning:
    x: + -> move away from the base
    y: + -> move right (see from the desk), i.e. move left (see from the back of the base)
    z: + -> move upward
    rot_x: rotate along x-axis, + -> clockwise see from the back of the base
    """

    for i in range(100):
        x = np.array([0.2, 0.3, 0.0, 0.0, 0.0, 0.0, 1]).astype(np.float32)
        client.update(x)
        time.sleep(0.1)

        state, in_good_range = client.get_state()
        for k, v in state.items():
            if k == "robot0_eef_pos":
                print(k, v)

        print(f"{in_good_range=}")
        curr_rot = Rotation.from_quat(state["robot0_eef_quat"]).as_euler("xyz")
        print(f"curr rot-x: {curr_rot[0] / np.pi}")
        print(f"curr rot-y: {curr_rot[1] / np.pi}")
        print(f"curr rot-z: {curr_rot[2] / np.pi}")

    return

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
            "local": "tcp://192.168.77.243:4242",
            "fr2": "tcp://192.168.77.21:4242",
        }
        if self.server:
            self.rpc_address = address_book["local"]
        else:
            self.rpc_address = address_book[self.server_loc]


if __name__ == "__main__":
    import time
    import pyrallis
    from rich.traceback import install

    install()
    cfg = pyrallis.parse(config_class=PolyMainConfig)  # type: ignore
    np.set_printoptions(precision=4, linewidth=100, suppress=True)

    if cfg.server:
        controller = URTDEController(cfg.controller)
        s = zerorpc.Server(controller)
        s.bind(cfg.rpc_address)
        s.run()
    elif cfg.goto_delta:
        goto_delta(cfg.rpc_address, cfg.task)
    elif cfg.goto:
        goto(cfg.goto, cfg.rpc_address, cfg.task)

@dataclass
class Args:
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