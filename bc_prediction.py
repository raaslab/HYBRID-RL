import torch
from common_utils import ibrl_utils as utils
from typing import Optional
import common_utils
import sys
import yaml
import copy
import train_bc
import os
from dataclasses import dataclass, field
from rl import replay
# from env.robosuite_wrapper import PixelRobosuite
from bc.bc_policy import BcPolicy, BcPolicyConfig
from bc.dataset import DatasetConfig, RobomimicDataset
from env.scripts.ur3e_wrapper import UR3eEnv, UR3eEnvConfig
import pyrallis
import numpy as np
import time



@dataclass
class MainConfig(common_utils.RunConfig):

    dataset: DatasetConfig = field(default_factory=lambda: DatasetConfig())
    policy: BcPolicyConfig = field(default_factory=lambda: BcPolicyConfig())
    seed: int = 1
    # env
    task_name: str = "lift"
    episode_length: int = 200
    end_on_success: int = 1
    # render image in higher resolution for recording or using pretrained models
    image_size: int = 224
    rl_image_size: int = 224
    rl_camera: str = "corner2"
    obs_stack: int = 1
    prop_stack: int = 1
    state_stack: int = 1
    # agent
    use_state: int = 0
    # q_agent: QAgentConfig = field(default_factory=lambda: QAgentConfig())
    stddev_max: float = 1.0
    stddev_min: float = 0.1
    stddev_step: int = 500000
    nstep: int = 3
    discount: float = 0.99
    replay_buffer_size: int = 500
    batch_size: int = 256
    num_critic_update: int = 1
    update_freq: int = 2
    bc_policy: str = "exps/bc/real_robot_2024_07_20_21_01_54/model1.pt"
    # rl with preload data
    mix_rl_rate: float = 1  # 1: only use rl, <1, mix in some bc data
    preload_num_data: int = 0
    preload_datapath: str = ""
    freeze_bc_replay: int = 1
    # pretrain rl policy with bc and finetune
    pretrain_only: int = 1
    pretrain_num_epoch: int = 0
    pretrain_epoch_len: int = 10000
    load_pretrained_agent: str = ""
    load_policy_only: int = 1
    add_bc_loss: int = 0
    # others
    env_reward_scale: float = 1
    num_warm_up_episode: int = 50
    num_eval_episode: int = 10
    save_per_success: int = -1
    mp_eval: int = 0  # eval with multiprocess
    num_train_step: int = 200000
    log_per_step: int = 5000
    # log
    save_dir: str = "exps/rl/robomimic_test"
    use_wb: int = 0

    def __post_init__(self):
        self.rl_cameras = self.rl_camera.split("+")

        if self.bc_policy in ["none", "None"]:
            self.bc_policy = ""

        if self.bc_policy:
            print(f"Using BC policy {self.bc_policy}")
            os.path.exists(self.bc_policy)

        if self.pretrain_num_epoch > 0:
            assert self.preload_num_data > 0

        self.stddev_min = min(self.stddev_max, self.stddev_min)

        if self.preload_datapath:
            self.num_warm_up_episode += self.preload_num_data

        if self.task_name == "TwoArmTransport":
            self.robots: list[str] = ["ur3e", "Panda"]
        else:
            self.robots: str = "ur3e"

    @property
    def bc_cameras(self) -> list[str]:
        if not self.bc_policy:
            return []

        bc_cfg_path = os.path.join(os.path.dirname(self.bc_policy), f"cfg.yaml")
        bc_cfg = pyrallis.load(train_bc.MainConfig, open(bc_cfg_path, "r"))  # type: ignore
        return bc_cfg.dataset.rl_cameras

    @property
    def stddev_schedule(self):
        return f"linear({self.stddev_max},{self.stddev_min},{self.stddev_step})"


class Workspace:
    def __init__(self, cfg: MainConfig, from_main=True):
        self.work_dir = cfg.save_dir
        print(f"workspace: {self.work_dir}")


        if from_main:
            common_utils.set_all_seeds(cfg.seed)
            sys.stdout = common_utils.Logger(cfg.log_path, print_to_stdout=True)

            pyrallis.dump(cfg, open(cfg.cfg_path, "w"))  # type: ignore
            print(common_utils.wrap_ruler("config"))
            with open(cfg.cfg_path, "r") as f:
                print(f.read(), end="")
            print(common_utils.wrap_ruler(""))

        self.cfg = cfg
        self.cfg_dict = yaml.safe_load(open(cfg.cfg_path, "r"))

        self.global_step = 0
        self.global_episode = 0
        self.train_step = 0
        self._setup_env()
        # self.dataset = RobomimicDataset(cfg.dataset)

        print(self.eval_env.observation_shape)
        # self.agent = QAgent(
        #     self.cfg.use_state,
        #     self.train_env.observation_shape,
        #     self.train_env.prop_shape,
        #     self.train_env.action_dim,
        #     self.cfg.rl_camera,
        #     cfg.q_agent,
        # )

        if not from_main:
            return

        if cfg.load_pretrained_agent and cfg.load_pretrained_agent != "None":
            print(f"loading loading pretrained agent from {cfg.load_pretrained_agent}")
            critic_states = copy.deepcopy(self.agent.critic.state_dict())
            self.agent.load_state_dict(torch.load(cfg.load_pretrained_agent))
            if cfg.load_policy_only:
                # avoid overwriting critic
                self.agent.critic.load_state_dict(critic_states)
                self.agent.critic_target.load_state_dict(critic_states)

        # self.ref_agent = copy.deepcopy(self.agent)
        # override to always use RL even when self.agent is ibrl
        # self.ref_agent.cfg.act_method = "rl"

        # set up bc related stuff
        self.bc_policy: Optional[torch.nn.Module] = None
        if cfg.bc_policy:
            bc_policy, _, bc_env_params = self.load_model(cfg.bc_policy, "cuda")
            # Temporary Removal
            # assert bc_env_params["obs_stack"] == self.eval_env_params["obs_stack"]

            # self.agent.add_bc_policy(copy.deepcopy(bc_policy))
            self.bc_policy = bc_policy

        print("BC Policy Loaded..!")
        self._setup_replay()


    def _setup_env(self):
        self.rl_cameras: list[str] = list(set(self.cfg.rl_cameras + self.cfg.bc_cameras))
        if self.cfg.use_state:
            self.rl_cameras = []
        print(f"rl_cameras: {self.rl_cameras}")

        if self.cfg.save_per_success > 0:
            for cam in ["agentview", "robot0_eye_in_hand"]:
                if cam not in self.rl_cameras:
                    print(f"Adding {cam} to recording camera because {self.cfg.save_per_success=}")
                    self.rl_cameras.append(cam)

        self.obs_stack = self.cfg.obs_stack
        self.prop_stack = self.cfg.prop_stack

        # self.train_env = PixelRobosuite(
        #     env_name=self.cfg.task_name,
        #     robots=self.cfg.robots,
        #     episode_length=self.cfg.episode_length,
        #     reward_shaping=False,
        #     image_size=self.cfg.image_size,
        #     rl_image_size=self.cfg.rl_image_size,
        #     camera_names=self.rl_cameras,
        #     rl_cameras=self.rl_cameras,
        #     env_reward_scale=self.cfg.env_reward_scale,
        #     end_on_success=bool(self.cfg.end_on_success),
        #     use_state=bool(self.cfg.use_state),
        #     obs_stack=self.obs_stack,
        #     state_stack=self.cfg.state_stack,
        #     prop_stack=self.prop_stack,
        #     record_sim_state=bool(self.cfg.save_per_success > 0),
        # )
        self.eval_env_params = dict(
            # env_name=self.cfg.task_name,
            task=self.cfg.task_name,
            # robots=self.cfg.robots,
            robot=self.cfg.robots,
            episode_length=self.cfg.episode_length,
            # reward_shaping=False,
            image_size=self.cfg.image_size,
            rl_image_size=self.cfg.rl_image_size,
            # camera_names=self.rl_cameras,
            # rl_cameras=self.rl_cameras,
            rl_camera="corner2",
            # use_state=self.cfg.use_state,
            # obs_stack=self.obs_stack,
            # state_stack=self.cfg.state_stack,
            # prop_stack=self.prop_stack,
            # New Params for Hardware
            # use_depth = 0,
            # record = 0,
            # drop_after_terminal=0,
            # show_camera = 0,
            # control_hz = 15.0,
        )
        
        cfg = UR3eEnvConfig(**self.eval_env_params)
        # self.eval_env = PixelRobosuite(**self.eval_env_params)  # type: ignore
        self.eval_env = UR3eEnv("cuda", cfg)  # type: ignore


    def _setup_replay(self):
        use_bc = False
        if self.cfg.mix_rl_rate < 1:
            use_bc = True
        if self.cfg.save_per_success > 0:
            use_bc = True
        if self.cfg.pretrain_num_epoch > 0 or self.cfg.add_bc_loss:
            assert self.cfg.preload_num_data
            use_bc = True

        self.replay = replay.ReplayBuffer(
            self.cfg.nstep,
            self.cfg.discount,
            frame_stack=1,
            max_episode_length=self.cfg.episode_length,
            replay_size=self.cfg.replay_buffer_size,
            use_bc=use_bc,
            save_per_success=self.cfg.save_per_success,
            save_dir=self.cfg.save_dir,
        )

        if self.cfg.preload_num_data:
            replay.add_demos_to_replay(
                self.replay,
                self.cfg.preload_datapath,
                num_data=self.cfg.preload_num_data,
                rl_cameras=self.rl_cameras,
                use_state=self.cfg.use_state,
                obs_stack=self.obs_stack,
                state_stack=self.cfg.state_stack,
                prop_stack=self.prop_stack,
                reward_scale=self.cfg.env_reward_scale,
                record_sim_state=bool(self.cfg.save_per_success > 0),
            )
        if self.cfg.freeze_bc_replay:
            assert self.cfg.save_per_success <= 0, "cannot save a non-growing replay"
            self.replay.freeze_bc_replay = True


    def _load_model(self, weight_file, env, device, cfg: Optional[MainConfig] = None):
        if cfg is None:
            cfg_path = os.path.join(os.path.dirname(weight_file), f"cfg.yaml")
            cfg = pyrallis.load(MainConfig, open(cfg_path, "r"))  # type: ignore

        print("_load_model: observation shape: ", env.observation_shape)
        print("_load_model: prop shape: ", env.prop_shape)
        print("_load_model: action dim: ", env.action_dim)
        if cfg.dataset.use_state:
            # policy = StateBcPolicy(env.observation_shape, env.action_dim, cfg.state_policy)
            print("I won't give state bc policy.. Bye!")
        else:
            policy = BcPolicy(
                # Temporary Change
                # prop_shape, action_dim is made to 4 
                # env.observation_shape, env.prop_shape, env.action_dim, env.rl_cameras, cfg.policy
                env.observation_shape, 4, 4, env.rl_cameras, cfg.policy
            )
        policy.load_state_dict(torch.load(weight_file))
        return policy.to(device)





    # function to load bc models
    def load_model(self, weight_file, device, *, verbose=True):
        run_folder = os.path.dirname(weight_file)
        cfg_path = os.path.join(run_folder, f"cfg.yaml")
        if verbose:
            print(common_utils.wrap_ruler("config of loaded agent"))
            with open(cfg_path, "r") as f:
                print(f.read(), end="")
            print(common_utils.wrap_ruler(""))

        # cfg = pyrallis.load(MainConfig, open(cfg_path, "r"))  # type: ignore

        assert not self.cfg.dataset.real_data
        env_params = dict(
            # env_name=cfg.task_name,
            task=self.cfg.task_name,
            # robots=cfg.robots,
            robot=self.cfg.robots,
            episode_length=self.cfg.dataset.eval_episode_len,
            reward_shaping=False,
            image_size=self.cfg.image_size,
            rl_image_size=self.cfg.rl_image_size,
            camera_names=self.cfg.dataset.rl_cameras,
            rl_cameras=self.cfg.dataset.rl_cameras,
            device=device,
            use_state=self.cfg.dataset.use_state,
            obs_stack=self.cfg.dataset.obs_stack,
            state_stack=self.cfg.dataset.state_stack,
            prop_stack=self.cfg.dataset.prop_stack,
            use_depth = 0,
            record = 0,
            show_camera = 0,
        )
        # env = PixelRobosuite(**env_params)  # type: ignore
        # env = UR3eEnv(**env_params)  # type: ignore

        if self.cfg.dataset.use_state:
            print(f"state_stack: {self.cfg.dataset.state_stack}, observation shape: {self.eval_env.observation_shape}")
        else:
            print(f"obs_stack: {self.cfg.dataset.obs_stack}, observation shape: {self.eval_env.observation_shape}")

        policy = self._load_model(weight_file, self.eval_env, device, self.cfg)
        return policy, self.eval_env, env_params

    def bc_predict(self):
        # warm up stage, fill the replay with some episodes
        # it can either be human demos, or generated by the bc, or purely random
        obs, _ = self.eval_env.reset()
        input_prop = torch.cat((obs['prop'][:3], obs['prop'][6:]))
        obs["prop"] = input_prop
        print(f"prop shape: {obs['prop'].shape}")
        print(obs['prop'].is_cuda)
        print(f"corner2 shape: {obs['corner2'].shape}")
        print(obs['corner2'].is_cuda)
        self.replay.new_episode(obs)
        total_reward = 0
        num_episode = 0
        print("Starting BC Prediction")

        for i in range(200):
            if self.bc_policy is not None:
                # we have a BC policy
                with torch.no_grad(), utils.eval_mode(self.bc_policy):
                    action = self.bc_policy.act(obs, eval_mode=True)


            action = action.numpy().flatten()
            action = np.concatenate((action[:3], np.array([0, 0, 0]), action[3:]))
            print("Action: ", action)
            
            #### ----- Take step -------- ##
            obs, reward, terminal, success, image_obs = self.eval_env.step(torch.tensor(action))
            # Cahnge input_prop to 4
            input_prop = torch.cat((obs['prop'][:3], obs['prop'][6:]))
            obs["prop"] = input_prop  

            time.sleep(0.1)
            # reply = {"action": action}
            # self.replay.add(obs, reply, reward, terminal, success, image_obs)

            if terminal:
                print("terminal is true")
                num_episode += 1
                total_reward += self.eval_env.episode_reward
                # break

        print(f"Warm up done. #episode: {self.replay.size()}")
        print(f"#episode from warmup: {num_episode}, #reward: {total_reward}")



if __name__ == "__main__":
    cfg = pyrallis.parse(config_class=MainConfig)  # type: ignore
    workspace = Workspace(cfg)
    workspace.bc_predict()