from collections import defaultdict
from dataclasses import dataclass

import torch
import numpy as np
import cv2
from urtde_controller2 import URTDEController
from env.cameras import RealSenseCamera
from common_utils import ibrl_utils as utils

from env.ur3e_utils import Rate
from env.lift import Lift
from env.drawer import Drawer
from env.hang import Hang
from env.towel import Towel

from urtde_controller2 import Args
import pyrallis
from scipy.spatial.transform import Rotation as R


_ROBOT_CAMERAS = {
    "ur3e": {
        # "agentview": "042222070680",
        # "robot0_eye_in_hand": "241222076578",
        # "frontview": "838212072814",
        "corner2": "944622074035",
    }
}

PROP_KEYS = ["robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"]


# Euler - Quat Conversion Functions
def eul_2_quat(rpy):
    rotation = R.from_rotvec(rpy)
    return rotation.as_quat()

def quat_2_eul(quat):
    rotation = R.from_quat(quat)
    return rotation.as_euler('xyz', degrees=False)



@dataclass
class UR3eEnvConfig:
    task: str = "lift"
    episode_length: int = 200
    robot: str = "ur3e"  # ur3e, fr3
    control_hz: float = 15.0
    image_size: int = 224
    # rl_image_size: int = 96
    rl_image_size: int = 224
    use_depth: int = 0
    # rl_camera: str = "robot0_eye_in_hand"
    rl_camera: str = "corner2"
    randomize: int = 0
    show_camera: int = 1
    drop_after_terminal: int = 1
    record: int = 0
    save_dir: str = ""

    def __post_init__(self):
        
        self.rl_cameras = self.rl_camera.split("+")

        if self.robot == "ur3e":
            self.remote_ip_address = "tcp://172.16.0.1:4242"
        elif self.robot == "local":
            self.remote_ip_address = "tcp://0.0.0.0:4242"
        else:
            assert False, f"unknown robot {self.robot}"


class UR3eEnv:
    """
    A simple Gym Environment for controlling robots.

    gripper: -1: open, 1: close
    """

    def __init__(self, device, cfg: UR3eEnvConfig):
        self.device = device
        self.cfg = cfg

        self.cameras = {}
        assert not self.cfg.use_depth
        for camera in self.cfg.rl_cameras:
            self.cameras[camera] = RealSenseCamera(
                _ROBOT_CAMERAS[self.cfg.robot][camera],
                width=cfg.image_size,
                height=cfg.image_size,
                depth=cfg.use_depth,
            )

        self.reward_camera = {}
        if self.cfg.task == "drawer":
            self.reward_camera["agentview"] = RealSenseCamera(
                _ROBOT_CAMERAS[self.cfg.robot]["agentview"],
                width=cfg.image_size,
                height=cfg.image_size,
                depth=cfg.use_depth,
            )

        self.record_camera = {}
        self.video_frames = defaultdict(list)
        if self.cfg.record:
            record_camera_name = "agentview"
            if record_camera_name in self.cameras:
                self.record_camera[record_camera_name] = self.cameras[record_camera_name]
            elif record_camera_name in self.reward_camera:
                self.record_camera[record_camera_name] = self.reward_camera[record_camera_name]
            else:
                self.record_camera[record_camera_name] = RealSenseCamera(
                    _ROBOT_CAMERAS[self.cfg.robot][record_camera_name],
                    width=cfg.image_size,
                    height=cfg.image_size,
                    depth=cfg.use_depth,
                )

        self.resize_transform = None
        if cfg.rl_image_size != cfg.image_size:
            self.resize_transform = utils.get_rescale_transform(cfg.rl_image_size)

        self.observation_shape: tuple[int, ...] = (3, cfg.rl_image_size, cfg.rl_image_size)
        self.prop_shape: tuple[int] = (7,)
        args = Args()
    
        # cfg2 = pyrallis.parse(config_class=PolyMainConfig)  # type: ignore
        np.set_printoptions(precision=4, linewidth=100, suppress=True)

        self.controller = URTDEController(args.controller, cfg.task)

        self.action_dim = len(self.controller.action_space.low)

        self.time_step = 0
        self.terminal = True

        # for compatibility
        self.rl_cameras = cfg.rl_cameras
        self.state_shape = (-1,)

        if cfg.task == "lift":
            self.task = Lift(verbose=False)
        elif cfg.task == "drawer":
            self.task = Drawer()
        elif cfg.task == "hang":
            self.task = Hang()
        elif cfg.task == "towel":
            self.task = Towel()
        else:
            assert False, f"unknown task {self.task}"

    def get_image_from_camera(self, cameras):
        # TODO: maybe fuse the functions that reads from camera?
        obs = {}
        for name, camera in cameras.items():
            frames = camera.get_frames()
            assert len(frames) == 1
            image = frames[""]
            image = torch.from_numpy(image).permute([2, 0, 1])
            if self.resize_transform is not None:
                image = self.resize_transform(image)
            obs[name] = image
        return obs

    def observe(self):
        props, in_good_range = self.controller.get_state()
        # props = self.controller.get_state()
        if not in_good_range:
            print("Warning[UR3eEnv]: bad range, should have restarted")

        # # ----- Convert Euler to Quat ------ #
        # quat = eul_2_quat(props['robot0_eef_quat'])
        # props["robot0_eef_quat"] = quat
        # # --------------------------------- #


        prop = torch.from_numpy(
            np.concatenate([props[prop_key] for prop_key in PROP_KEYS]).astype(np.float32)
        )
        assert prop.size(0) == self.prop_shape[0], f"{prop.size(0)=}, {self.prop_shape[0]=}"

        rl_obs = {"prop": prop.to(self.device)}
        high_res_images = {}

        for name, camera in self.cameras.items():
            frames = camera.get_frames()
            assert len(frames) == 1
            image = frames[""]
            if name == "frontview":
                image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)

            key = f"{name}"
            high_res_images[key] = image

            rl_image_obs = torch.from_numpy(image).permute([2, 0, 1]).to(self.device)
            if self.resize_transform is not None:
                # set the device here because transform is 5x faster on GPU
                rl_image_obs = self.resize_transform(rl_image_obs.to(self.device))
            rl_obs[key] = rl_image_obs

        if self.cfg.show_camera:
            images = []
            for _, v in high_res_images.items():
                # np_image = v.cpu().permute([1, 2, 0]).numpy()
                images.append(v)
            image = np.hstack(images)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.imshow("img", image)
            cv2.waitKey(1)

        if self.cfg.record:
            for k, camera in self.record_camera.items():
                if k in high_res_images:
                    self.video_frames[k] = high_res_images[k]
                else:
                    frames = camera.get_frames()
                    assert len(frames) == 1
                    self.video_frames[k] = frames[""]

        return rl_obs

    def get_reward_terminal(self):
        if self.cfg.task == "lift":
            props, in_good_range = self.controller.get_state()
            reward: float = self.task.reward(props)
        elif self.cfg.task == "drawer":
            _, in_good_range = self.controller.get_state()
            reward_obs = self.get_image_from_camera(self.reward_camera)
            reward: float = self.task.reward(reward_obs)
        elif self.cfg.task in ["hang", "towel"]:
            props, in_good_range = self.controller.get_state()
            reward_obs = self.observe()
            reward_obs.update(props)
            reward: float = self.task.reward(reward_obs)
        else:
            assert False

        success = reward > 0

        self.terminal = success
        if self.time_step >= self.cfg.episode_length:
            self.terminal = True
        if not in_good_range:
            self.terminal = True

        if success and self.cfg.drop_after_terminal:
            self.release_gripper()

        # print(f"step: {self.time_step}, terminal: {self.terminal}, {reward=}")
        if self.terminal and self.cfg.record:
            pass

        return reward, self.terminal, success

    def apply_action(self, action: torch.Tensor):
        # print(">>>>>>>>>>>>>>>>> apply action", action.size())
        if isinstance(action, np.ndarray):
            self.controller.update(action)
        else:
            self.controller.update(action.numpy())
        self.time_step += 1
        return

    # ============= gym style api for compatibility with data collection ============= #
    def reset(self) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """return observation and high resolution observation"""
        # print(f"{self.cfg.randomize=}")
        self.controller.reset(randomize=bool(self.cfg.randomize))

        self.time_step = 0
        self.episode_reward = 0
        self.episode_extra_reward = 0
        self.terminal = False
        self.video_frames = []

        # self.task.reset()
        # self.controller.reset()
        return self.observe(), {}

    # ============= gym style api for compatibility with data collection ============= #
    def step(
        self, action: torch.Tensor
    ) -> tuple[dict[str, torch.Tensor], float, bool, bool, dict[str, torch.Tensor]]:
        # Immediately update with the action.
        # Note that `action` has been scaled to [-1, 1],
        # `self.controller.update` will perform the unscale

        with Rate(self.cfg.control_hz):
            # print(f"[env] step: {self.time_step}")
            # print(f"Action: {action.dim()}")
            assert action.dim() == 1, "multi-action open loop not supported yet"
            assert action.min() >= -1, action.min()

            # self.controller.update(action.numpy())
            self.apply_action(action.numpy())
            self.time_step += 1

        rl_obs = self.observe()
        reward, terminal, success = self.get_reward_terminal()

        if terminal and self.cfg.drop_after_terminal:
            self.release_gripper()
        #     # release the gripper
        #     action[-1] = -1
        #     self.controller.update(action.cpu().numpy())

        return rl_obs, reward, terminal, success, {}

    def release_gripper(self):
        action = np.zeros(self.action_dim)
        action[-1] = 0
        self.controller.update(action)


def test():
    np.set_printoptions(precision=4)

    cfg = UR3eEnvConfig()
    env = UR3eEnv("cuda", cfg)

    env.reset()
    # while True:
    print("-------------------------")
    obs = env.observe()
    for k, v in obs.items():
        print(k, v.size())



    action = np.array([0.0016, -0.0041, -0.0028, 0, 0, 0, 0.0497])    
    env.apply_action(action)

    # with Rate(10.0):
    # print("--- Random Action ----")
    # action_rand = np.random.uniform(low=-0.01, high=0.01, size=6).astype(np.float32)
    # action = np.concatenate((action_rand, [0.0]))
    # env.apply_action(action)

    # obs = env.observe()
    # for k, v in obs.items():
    #     print(k, v.size())


if __name__ == "__main__":
    test()
