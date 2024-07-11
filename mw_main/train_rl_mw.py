import os
import sys
from typing import Any, Optional
from dataclasses import dataclass, field
import yaml
import copy
import pprint
import pyrallis
import torch
import numpy as np
import matplotlib.pyplot as plt
import common_utils
from common_utils import ibrl_utils as utils
from rl.q_agent import QAgent, QAgentConfig
from env.metaworld_wrapper import PixelMetaWorld
import mw_replay
import train_bc_mw
from eval_mw import run_eval
import cv2
import torch.nn as nn
from torchvision import models
# from mask import SegmentedHybridResNet
from mode_classifier_image import HybridResNet,device
from waypoint import WaypointPredictor




def predict_waypoint(model, corner2_image, prop):
    with torch.no_grad():  # Ensure no gradients are computed during prediction
        corner2_image = torch.tensor(corner2_image, dtype=torch.float32).unsqueeze(0)  # Add batch dimension
        prop = torch.tensor(prop, dtype=torch.float32).unsqueeze(0)  # Add batch dimension
        waypoint = model(corner2_image, prop)
    return waypoint.numpy().flatten()

BC_POLICIES = {
    "assembly": "/home/amisha/ibrl/exps/bc/metaworld/data_seed_0_Assembly/model1.pt",
    "boxclose": "/home/amisha/ibrl/exps/bc/metaworld/data_seed_0_BoxClose/model1.pt",
    "coffeepush": "/home/amisha/ibrl/exps/bc/metaworld/data_seed_1_CoffeePush/model1.pt",
    "stickpull": "/home/amisha/ibrl/exps/bc/metaworld/data_seed_0_StickPull/model1.pt",
}

BC_DATASETS = {
    "assembly": "release/data/metaworld/Assembly_frame_stack_1_96x96_end_on_success/dataset.hdf5",
    "boxclose": "release/data/metaworld/BoxClose_frame_stack_1_96x96_end_on_success/dataset.hdf5",
    "coffeepush": "release/data/metaworld/CoffeePush_frame_stack_1_96x96_end_on_success/dataset.hdf5",
    "stickpull": "release/data/metaworld/StickPull_frame_stack_1_96x96_end_on_success/dataset.hdf5",
}

    
@dataclass
class MainConfig(common_utils.RunConfig):
    seed: int = 0
    # Sparse control parameters
    Kp = 15
    # env
    episode_length: int = 200
    # agent
    q_agent: QAgentConfig = field(default_factory=lambda: QAgentConfig())
    stddev_max: float = 1.0
    stddev_min: float = 0.1
    stddev_step: int = 500000
    nstep: int = 3
    discount: float = 0.99
    replay_buffer_size: int = 500
    batch_size: int = 256
    num_critic_update: int = 1
    update_freq: int = 2
    bc_policy: str = ""
    use_bc: int = 1
    # load demo
    mix_rl_rate: float = 1  # 1: only use rl, <1, mix in some bc data
    preload_num_data: int = 0
    preload_datapath: str = ""
    env_reward_scale: int = 1
    # others
    num_train_step: int = 60000
    log_per_step: int = 5000
    num_warm_up_episode: int = 50
    num_eval_episode: int = 10
    # rft
    pretrain_num_epoch: int = 0
    pretrain_epoch_len: int = 10000
    add_bc_loss: int = 0
    # log
    use_wb: int = 0
    save_dir: str = ""

    def __post_init__(self):
        self.preload_datapath = self.bc_policy
        if self.preload_datapath in BC_DATASETS:
            self.preload_datapath = BC_DATASETS[self.preload_datapath]
            dataset_name = self.bc_policy.split('/')[-1]       # for saving dir

        # self.save_dir = f"exps/rl/metaworld/hyrl/hyrl_seed_{self.seed}_{dataset_name}_kp11"
        self.save_dir = f"exps/rl/metaworld/hyrl/randomization_test"



    @property
    def stddev_schedule(self):
        return f"linear({self.stddev_max},{self.stddev_min},{self.stddev_step})"


class Workspace:
    def __init__(self, cfg: MainConfig):
        self.Kp = cfg.Kp
        self.work_dir = cfg.save_dir
        print(f"workspace: {self.work_dir}")

        # # Create a video window
        self.window_name = 'Metaworld Environment'
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, 600, 600) 

        common_utils.set_all_seeds(cfg.seed)
        sys.stdout = common_utils.Logger(cfg.log_path, print_to_stdout=True)
        pyrallis.dump(cfg, open(cfg.cfg_path, "w"))  # type: ignore
        print(common_utils.wrap_ruler("config"))
        with open(cfg.cfg_path, "r") as f:
            print(f.read(), end="")
        print(common_utils.wrap_ruler(""))

        self.cfg = cfg
        self.cfg_dict = yaml.safe_load(open(cfg.cfg_path, "r"))
        # we need bc policy to construct the environment :(, hack!
        assert cfg.bc_policy != "", "bc policy must be set to find the correct env config"
        self.env_params: dict[str, Any]
        if cfg.bc_policy in BC_POLICIES:
            cfg.bc_policy = BC_POLICIES[cfg.bc_policy]
        bc_policy, _, self.env_params = train_bc_mw.load_model(cfg.bc_policy, "cuda")

        self.bc_policy = None
        if cfg.use_bc:
            self.bc_policy = bc_policy

        self.global_step = 0
        self.global_episode = 0
        self.train_step = 0
        self.num_success = 0
        self._setup_env()

        # Waypoint Predictor Initialization
        dataset_name = cfg.bc_policy.split('/')[-2].split('_')[-1].lower()
        self.waypoint_predictor = WaypointPredictor().cuda()
        waypoint_path = f"waypoint_models/waypoint_{dataset_name}.pth"
        print(f"Using waypoint_model_path: {waypoint_path} ")
        self.waypoint_predictor.load_state_dict(torch.load(waypoint_path, map_location="cuda"))
        self.waypoint_predictor.eval()  # Set the model to evaluation mode

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.classifier_model = HybridResNet()
        mode_path ="/home/amisha/ibrl/mode_models_jul_03/mode_assembly.pth"
        # mode_path = f"mode_models_augmented/mode_{dataset_name}.pth"
        print(f"Using mode_model_path: {mode_path}")
        # model_state_dict = torch.load(mode_path, map_location=device)
        # new_state_dict = {k: v for k, v in model_state_dict.items() if 'state_processor' not in k and 'classifier.weight' not in k}
        # self.classifier_model.load_state_dict(new_state_dict, strict=False)

        # # Manually handle the classifier weights if needed
        # if 'classifier.weight' in model_state_dict:
        #     with torch.no_grad():
        #         self.classifier_model.classifier.weight[:,:] = model_state_dict['classifier.weight'][:,:2048]  # Adjust as necessary
        #         self.classifier_model.classifier.bias[:] = model_state_dict['classifier.bias']

        self.classifier_model.load_state_dict(torch.load(mode_path, map_location=device))
        self.classifier_model.to(device)
        self.classifier_model.eval()





        assert not cfg.q_agent.use_prop, "not implemented"
        self.agent = QAgent(
            False,
            self.train_env.observation_shape,
            (4,),  # prop shape, does not matter as we do not use prop in metaworld
            self.train_env.num_action,
            rl_camera="obs",
            cfg=cfg.q_agent,
        )
        if self.bc_policy is not None:
            self.agent.add_bc_policy(bc_policy=copy.deepcopy(self.bc_policy))

        self._setup_replay()
        self.ref_agent: Optional[QAgent] = None

    def _setup_env(self):
        # camera_names = [self.cfg.rl_camera]
        print(common_utils.wrap_ruler("Env Config"))
        pprint.pprint(self.env_params)
        print(common_utils.wrap_ruler(""))

        if "end_on_success" not in self.env_params:
            self.env_params["end_on_success"] = True
        self.env_params["episode_length"] = self.cfg.episode_length
        self.env_params["env_reward_scale"] = self.cfg.env_reward_scale
        self.train_env = PixelMetaWorld(**self.env_params)  # type: ignore

        eval_env_params = self.env_params.copy()
        eval_env_params["env_reward_scale"] = 1.0
        self.eval_env = PixelMetaWorld(**eval_env_params)  # type: ignore

    def _setup_replay(self):
        use_bc = (self.cfg.mix_rl_rate < 1) or self.cfg.add_bc_loss
        assert self.env_params["frame_stack"] == 1
        self.replay = mw_replay.ReplayBuffer(
            self.cfg.nstep,
            self.cfg.discount,
            frame_stack=1,
            max_episode_length=self.cfg.episode_length,  # env_params["episode_length"],
            replay_size=self.cfg.replay_buffer_size,
            use_bc=use_bc,
        )
        if self.cfg.preload_num_data:
            mw_replay.add_demos_to_replay(
                self.replay,
                self.cfg.preload_datapath,
                num_data=self.cfg.preload_num_data,
                rl_camera=self.env_params["rl_camera"],
                use_state=self.env_params["use_state"],
                obs_stack=self.env_params["obs_stack"],
                reward_scale=self.cfg.env_reward_scale,
            )
            self.replay.freeze_bc_replay = True

    def eval(self, seed, policy):
        random_state = np.random.get_state()
        scores = run_eval(
            env=self.eval_env,
            agent=policy,
            num_game=self.cfg.num_eval_episode,
            seed=seed,
            record_dir=None,
            verbose=False,
        )
        np.random.set_state(random_state)
        return float(np.mean(scores))

    def warm_up(self):
        # warm up stage, fill the replay with some episodes
        # it can either be human demos, or generated by the bc, or purely random
        obs, _ = self.train_env.reset()
        for k, v in obs.items():
            print(k, v.size())
        self.replay.new_episode(obs)
        total_reward = 0
        num_episode = 0
        while True:
            if self.bc_policy is not None:
                with torch.no_grad(), utils.eval_mode(self.bc_policy):
                    action = self.bc_policy.act(obs, eval_mode=True)
            else:
                if self.cfg.pretrain_num_epoch > 0:
                    # the policy has been pretrained
                    with torch.no_grad(), utils.eval_mode(self.agent):
                        action = self.agent.act(obs, eval_mode=True)
                else:
                    action = torch.zeros(self.train_env.action_dim)
                    action = action.uniform_(-1.0, 1.0)

            obs, reward, terminal, success, image_obs = self.train_env.step(action.numpy())
            reply = {"action": action}
            self.replay.add(obs, reply, reward, terminal, success, image_obs)

            if terminal:
                num_episode += 1
                total_reward += self.train_env.episode_reward
                if self.replay.size() < self.cfg.num_warm_up_episode:
                    self.replay.new_episode(obs)
                    obs, _ = self.train_env.reset()
                else:
                    break

        print(f"Warm up done. #episode: {self.replay.size()}")
        print(f"#episode from warmup: {num_episode}, #reward: {total_reward}")

    def train(self):
        stat = common_utils.MultiCounter(
            self.work_dir,
            bool(self.cfg.use_wb),
            wb_exp_name=self.cfg.wb_exp,
            wb_run_name=self.cfg.wb_run,
            wb_group_name=self.cfg.wb_group,
            config=self.cfg_dict,
        )
        self.agent.set_stats(stat)
        saver = common_utils.TopkSaver(save_dir=self.work_dir, topk=1)
        self.warm_up()
        self.num_success = self.replay.num_success
        stopwatch = common_utils.Stopwatch()
        # moving_to_waypoint = False
        # current_waypoint = None

        obs, image_obs = self.train_env.reset()
        self.replay.new_episode(obs)
        print(obs.keys())  # To see all keys in the observation dictionary
        print(image_obs.keys())  # If image_obs is a dictionary, check its structure
        while self.global_step < self.cfg.num_train_step:    
            current_prop = obs['prop']  # Adapt these keys based on how your observations are structured
            current_image = image_obs['corner2']
            mode_img = np.zeros((400,400))
            # object_pos = self.train_env.first_obs_pos
            mode = self.determine_mode(current_image)
            # mode = "dense"
            # print(f"Determined Mode: {mode}")
            ### Act based on mode ###
            if mode == 'sparse':
                # if not moving_to_waypoint:
                    # Only predict new waypoint if we're not already moving to one
                with torch.no_grad():
                    predicted_waypoint = self.waypoint_predictor(
                        torch.tensor(current_image, dtype=torch.float32, device="cuda").unsqueeze(0),
                        torch.tensor(current_prop[-1], dtype=torch.float32, device="cuda").unsqueeze(0).unsqueeze(-1)
                    ).squeeze(0)  # Keep it as a GPU tensor
                    # current_waypoint = predicted_waypoint
                    # moving_to_waypoint = True

                action = self.servoing(obs, predicted_waypoint)
                mode = self.determine_mode(current_image)
                # # Check if waypoint is reached
                # if self.waypoint_reached(obs['prop'][:3], current_waypoint):
                #     moving_to_waypoint = False
                #     mode = 'dense' 
                # print(f"Waypoint Reached Status: {self.waypoint_reached(obs['prop'][:3], current_waypoint)} ")
            ### act ###
            if mode == 'dense':
                with stopwatch.time("act"), torch.no_grad(), utils.eval_mode(self.agent):
                    stddev = utils.schedule(self.cfg.stddev_schedule, self.global_step)
                    action = self.agent.act(obs, stddev=stddev, eval_mode=False)
                    stat["data/stddev"].append(stddev)
                # print(f"Dense Mode Action: {action}")
            with stopwatch.time("env step"):
                obs, reward, terminal, success, image_obs = self.train_env.step(action.numpy())
                # # ----> Render the environment <----
                try:
                    img = self.train_env.env.env.render(mode='rgb_array')

                    mode_img = cv2.putText(mode_img, str(reward), (mode_img.shape[1]//2 - 50, mode_img.shape[0]//2-50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2, cv2.LINE_AA)
                    mode_img = cv2.putText(mode_img, mode, (mode_img.shape[1]//2 - 80, mode_img.shape[0]//2), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 2, cv2.LINE_AA)
                    if mode=="sparse":
                        mode_img = cv2.putText(mode_img, f"X: {predicted_waypoint[0]}", 
                                               (50, mode_img.shape[0]//2+50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2, cv2.LINE_AA)
                        mode_img = cv2.putText(mode_img, f"Y: {predicted_waypoint[1]}", 
                                               (50, mode_img.shape[0]//2+80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2, cv2.LINE_AA)
                        mode_img = cv2.putText(mode_img, f"Z: {predicted_waypoint[2]}", 
                                               (50, mode_img.shape[0]//2+110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2, cv2.LINE_AA)                       
                    cv2.imshow(self.window_name,  cv2.cvtColor(img, cv2.COLOR_BGR2RGB))                    
                    cv2.imshow("Mode",  mode_img)                    
                    cv2.waitKey(1)
                except Exception as e:
                    print(f"Error rendering image: {e}")   


            with stopwatch.time("add"):
                assert isinstance(terminal, bool)
                reply = {"action": action}
                self.replay.add(obs, reply, reward, terminal, success, image_obs)
                self.global_step += 1
                # print(f"Global step after increment: {self.global_step}")
            # print(f"Global Step: {self.global_step}, Reward: {reward}, Terminal: {terminal}, Success: {success}")
            if terminal:
                # print(f"Terminal condition met at global step: {self.global_step}")
                with stopwatch.time("reset"):
                    self.global_episode += 1
                    stat["score/train_score"].append(success)
                    stat["data/episode_len"].append(self.train_env.time_step)
                    if self.replay.bc_replay is not None:
                        stat["data/bc_replay"].append(self.replay.size(bc=True))
                    # reset env
                    obs, image_obs = self.train_env.reset()
                    # mode = self.determine_mode(current_image)
                    mode="sparse"
                    print(f"prop: {obs['prop']}")
                    self.replay.new_episode(obs)

            ### logging ###
            if self.global_step % self.cfg.log_per_step == 0:
                self.log_and_save(stopwatch, stat, saver)

            ### train ###
            if self.global_step % self.cfg.update_freq == 0:
                with stopwatch.time("train"):
                    self.rl_train(stat)
                    self.train_step += 1

        # cv2.destroyAllWindows()

    def log_and_save(
        self,
        stopwatch: common_utils.Stopwatch,
        stat: common_utils.MultiCounter,
        saver: common_utils.TopkSaver,
    ):
        elapsed_time = stopwatch.elapsed_time_since_reset
        stat["other/speed"].append(self.cfg.log_per_step / elapsed_time)
        stat["other/elapsed_time"].append(elapsed_time)
        stat["other/episode"].append(self.global_episode)
        stat["other/step"].append(self.global_step)
        stat["other/train_step"].append(self.train_step)
        stat["other/replay"].append(self.replay.size())
        stat["score/num_success"].append(self.replay.num_success)

        with stopwatch.time("eval"):
            eval_score = self.eval(seed=self.global_step, policy=self.agent)
            stat["score/score"].append(eval_score)

        saved = saver.save(self.agent.state_dict(), eval_score, save_latest=True)
        print(f"saved?: {saved}")

        stat.summary(self.global_step, reset=True)
        stopwatch.summary(reset=True)
        print("total time:", common_utils.sec2str(stopwatch.total_time))
        print(common_utils.get_mem_usage())

    def rl_train(self, stat: common_utils.MultiCounter):
        stddev = utils.schedule(self.cfg.stddev_schedule, self.global_step)
        for i in range(self.cfg.num_critic_update):
            if self.cfg.mix_rl_rate < 1:
                rl_bsize = int(self.cfg.batch_size * self.cfg.mix_rl_rate)
                bc_bsize = self.cfg.batch_size - rl_bsize
                batch = self.replay.sample_rl_bc(rl_bsize, bc_bsize, "cuda:0")
            else:
                batch = self.replay.sample(self.cfg.batch_size, "cuda:0")

            update_actor = i == self.cfg.num_critic_update - 1

            if update_actor and self.cfg.add_bc_loss:
                bc_batch = self.replay.sample_bc(self.cfg.batch_size, "cuda:0")
            else:
                bc_batch = None

            if self.cfg.add_bc_loss:
                metrics = self.agent.update(batch, stddev, update_actor, bc_batch, self.ref_agent)
            else:
                metrics = self.agent.update(batch, stddev, update_actor)

            stat.append(metrics)
            stat["data/discount"].append(batch.bootstrap.mean().item())

    def pretrain_policy(self):
        stat = common_utils.MultiCounter(
            self.work_dir,
            bool(self.cfg.use_wb),
            wb_exp_name=self.cfg.wb_exp,
            wb_run_name=self.cfg.wb_run,
            wb_group_name=self.cfg.wb_group,
            config=self.cfg_dict,
        )
        saver = common_utils.TopkSaver(save_dir=self.work_dir, topk=1)

        for epoch in range(self.cfg.pretrain_num_epoch):
            for _ in range(self.cfg.pretrain_epoch_len):
                batch = self.replay.sample_bc(self.cfg.batch_size, "cuda")
                metrics = self.agent.pretrain_actor_with_bc(batch)
                stat.append(metrics)

            eval_seed = epoch * self.cfg.pretrain_epoch_len
            score = self.eval(eval_seed, policy=self.agent)
            stat["pretrain/score"].append(score)
            saved = saver.save(self.agent.state_dict(), score, save_latest=True)

            stat.summary(epoch, reset=True)
            print(f"saved?: {saved}")
            print(common_utils.get_mem_usage())

    # def determine_mode(self, obs, object_pos):
    #     difference = object_pos - obs["prop"][:3]
    #     threshold = torch.norm(difference).item()
    #     # print(f"Threshold: {threshold}")
    #     # if threshold > SPARSE_THRESHOLD and obs["prop"][3] >= 1.0:  # Define your threshold
    #     if threshold > SPARSE_THRESHOLD:  # Define your threshold
    #         return 'sparse'
    #     else:
    #         return 'dense'


    def waypoint_reached(self, current_position, waypoint, threshold=0.035):
        """
        Check if the current position is close enough to the waypoint.
        
        Args:
        current_position (np.array or torch.Tensor): Current position of the end-effector
        waypoint (torch.Tensor): Target waypoint (on GPU)
        threshold (float): Distance threshold to consider waypoint as reached
        
        Returns:
        bool: True if waypoint is reached, False otherwise
        """
        if isinstance(current_position, np.ndarray):
            current_position = torch.from_numpy(current_position).to(waypoint.device)
        distance = torch.norm(current_position - waypoint).item()
        return distance < threshold



    def determine_mode(self, corner2_image):
        # Ensure the image is in [C, H, W] format
        if corner2_image.shape != (3, 96, 96):  # Assuming the expected shape is [C, H, W]
            corner2_image = corner2_image.permute(2, 0, 1)  # Change from [H, W, C] to [C, H, W]
        # prop = prop.to(device).float()

        # # Extract only the last value of the prop tensor
        # last_prop_value = prop[-1].unsqueeze(0)  # Adds an extra dimension to match batch size of 1

        corner2_image = corner2_image.to("cuda").float()
        if len(corner2_image.shape) == 3:
            corner2_image = corner2_image.unsqueeze(0)

        with torch.no_grad():
            outputs = self.classifier_model(corner2_image)
            predicted_mode = torch.argmax(outputs, dim=1).item()  # Returns 0 or 1
        return 'sparse' if predicted_mode == 0 else 'dense'


    def servoing(self, obs, waypoint):
        # Initialize the error tensor with a large initial value
        error = torch.tensor(100.0, dtype=torch.float32).to(self.train_env.device)
        gripper_control = -1
        step_count = 0  # Define step_count here
        # while torch.norm(error).item() > SPARSE_THRESHOLD:
            # Compute the error
        error = waypoint - obs["prop"][:3]  # obs["prop"][:3] - first object position

        # Convert the error tensor to a NumPy array
        error_np = error.cpu().numpy()
        # print("error_)))))))))0000000000000000000000",torch.norm(error).item())
        # Compute the control action
        control_action = self.Kp * error_np
        
        action = np.zeros(4)
        action[:3] = control_action
        action[3] = gripper_control  # Control the gripper, set as needed

        # Clip the action to ensure it’s within the action min and max limits
        action = np.clip(action, -1, 1)

        # print("Reached First Object")
        return torch.tensor(action)


def main(cfg: MainConfig):
    workspace = Workspace(cfg)

    if cfg.pretrain_num_epoch > 0:
        print("pretraining:")
        workspace.pretrain_policy()
        workspace.ref_agent = copy.deepcopy(workspace.agent)

    workspace.train()

    if cfg.use_wb:
        wandb.finish()

    assert False


if __name__ == "__main__":
    import wandb
    from rich.traceback import install

    os.environ["MUJOCO_GL"] = "egl"

    install()
    torch.backends.cudnn.allow_tf32 = True  # type: ignore
    torch.backends.cudnn.benchmark = True  # type: ignore

    cfg = pyrallis.parse(config_class=MainConfig)  # type: ignore
    main(cfg)
