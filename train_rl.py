import os
import sys
from dataclasses import dataclass, field
import yaml
import copy
from typing import Optional
import pyrallis
import torch
import numpy as np

import common_utils
from common_utils import ibrl_utils as utils
from evaluate import run_eval, run_eval_mp
from env.robosuite_wrapper import PixelRobosuite
from rl.q_agent_dual import QAgent, QAgentConfig
from rl import replay
import train_bc
from env.scripts.ur3e_wrapper import UR3eEnv, UR3eEnvConfig
from bc.bc_policy import BcPolicy, BcPolicyConfig
from bc.dataset import DatasetConfig, RobomimicDataset
import time


from pynput import keyboard



def filter_obs(obs):
    input_prop = torch.cat((obs['prop'][:3], obs['prop'][6:]))
    obs["prop"] = input_prop
    return obs

def filter_action(action):        
    action = action.numpy().flatten()
    if action[3:] < 0: # if Gripper value is negative, set it to 0
        action[3:] = 0.0
    action = np.concatenate((action[:3], np.array([0, 0, 0]), action[3:]))
    return torch.tensor(action)


@dataclass
class MainConfig(common_utils.RunConfig):

    dataset: DatasetConfig = field(default_factory=lambda: DatasetConfig())
    policy: BcPolicyConfig = field(default_factory=lambda: BcPolicyConfig())
    seed: int = 1
    # env
    task_name: str = "two_stage"
    episode_length: int = 250
    end_on_success: int = 1
    # render image in higher resolution for recording or using pretrained models
    image_size: int = 96
    rl_image_size: int = 96
    rl_camera: str = "corner2+eye_in_hand"
    obs_stack: int = 1
    prop_stack: int = 1
    state_stack: int = 1
    # agent
    use_state: int = 0
    q_agent: QAgentConfig = field(default_factory=lambda: QAgentConfig())
    stddev_max: float = 1.0
    stddev_min: float = 0.1
    stddev_step: int = 500000
    nstep: int = 3
    discount: float = 0.99
    replay_buffer_size: int = 500
    batch_size: int = 128
    num_critic_update: int = 1
    update_freq: int = 2
    # bc_policy: str = "exps/bc/real_robot_2024_07_23_15_58_18/model1.pt"
    bc_policy: str = "exps/2S/real_robot_dual2S_b32_2024_07_29_22_27_21/model1.pt"
    # rl with preload data
    mix_rl_rate: float = 1  # 1: only use rl, <1, mix in some bc data
    preload_num_data: int = 0
    preload_datapath: str = "release/data/real_robot_2S/data_2S.hdf5"
    # preload_datapath: str = "release/data/robomimic/can/processed_data96.hdf5"
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
    num_train_step: int = 20000
    log_per_step: int = 1000
    # log
    save_dir: str = "exps/rl/train_rl_hardw_ibrl_two_stage"
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

        print("Observations: ",self.train_env.observation_shape)
        print("Prop: ", self.train_env.prop_shape)
        print("Action dim: ", self.train_env.action_dim)
        print("Using Cam for QAgent: ", "eye_in_hand") 
        self.agent = QAgent(
            self.cfg.use_state,
            self.train_env.observation_shape,
            # self.train_env.prop_shape,
            (4,),
            # self.train_env.action_dim,
            4, 
            self.cfg.rl_cameras,
            # "eye_in_hand",
            cfg.q_agent,
        )

        if not from_main:
            return


        self.ref_agent = copy.deepcopy(self.agent)
        # override to always use RL even when self.agent is ibrl
        self.ref_agent.cfg.act_method = "rl"

        # set up bc related stuff
        self.bc_policy: Optional[torch.nn.Module] = None
        if cfg.bc_policy:
            bc_policy, _ = self.load_model(cfg.bc_policy, "cuda")
            # assert bc_env_params["obs_stack"] == self.eval_env_params["obs_stack"]

            self.agent.add_bc_policy(copy.deepcopy(bc_policy))
            self.bc_policy = bc_policy

        print("BC Policy Loaded..!")
        self._setup_replay()


        # Keyboard listen for terminating
        self.terminate_episode = False
        self.listener = keyboard.Listener(on_press=self.on_press)
        self.listener.start()

    def _setup_env(self):
        self.rl_cameras: list[str] = list(set(self.cfg.rl_cameras + self.cfg.bc_cameras))
        if self.cfg.use_state:
            self.rl_cameras = []
        print(f"rl_cameras: {self.rl_cameras}")

        if self.cfg.save_per_success > 0:
            for cam in ["corner2", "eye_in_hand"]:
                if cam not in self.rl_cameras:
                    print(f"Adding {cam} to recording camera because {self.cfg.save_per_success=}")
                    self.rl_cameras.append(cam)

        self.obs_stack = self.cfg.obs_stack
        self.prop_stack = self.cfg.prop_stack

        self.train_env_params = dict(
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
            rl_camera="corner2+eye_in_hand",
            randomize = 0,
            # use_state=self.cfg.use_state,
            # obs_stack=self.obs_stack,
            # state_stack=self.cfg.state_stack,
            # prop_stack=self.prop_stack,
            
            # We Added - New Params for Hardware
            # use_depth = 0,
            # record = 0,
            # drop_after_terminal=0,
            show_camera = 0,
            control_hz = 10,
        )
        
        cfg = UR3eEnvConfig(**self.train_env_params)
        print(f"Start Ur3e with Control Hz: {cfg.control_hz}")
        self.train_env = UR3eEnv("cuda", cfg)  # type: ignore



    def _load_model(self, weight_file, env, device, cfg: Optional[MainConfig] = None):
        if cfg is None:
            cfg_path = os.path.join(os.path.dirname(weight_file), f"cfg.yaml")
            cfg = pyrallis.load(MainConfig, open(cfg_path, "r"))  # type: ignore

        print("observation shape: ", env.observation_shape)
  
        policy = BcPolicy(
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


        if self.cfg.dataset.use_state:
            print(f"state_stack: {self.cfg.dataset.state_stack}, observation shape: {self.train_env.observation_shape}")
        else:
            print(f"obs_stack: {self.cfg.dataset.obs_stack}, observation shape: {self.train_env.observation_shape}")

        policy = self._load_model(weight_file, self.train_env, device, self.cfg)
        return policy, self.train_env







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
    # def eval(self, seed, policy) -> float:
    #     random_state = np.random.get_state()

    #     if self.cfg.mp_eval:
    #         scores: list[float] = run_eval_mp(
    #             env_params=self.eval_env_params,
    #             agent=policy,
    #             num_proc=10,
    #             num_game=self.cfg.num_eval_episode,
    #             seed=seed,
    #             verbose=False,
    #         )
    #     else:
    #         scores: list[float] = run_eval(
    #             env_params=self.eval_env_params,
    #             agent=policy,
    #             num_game=self.cfg.num_eval_episode,
    #             seed=seed,
    #             record_dir=None,
    #             verbose=False,
    #         )

    #     np.random.set_state(random_state)
    #     return float(np.mean(scores))  # type: ignore

    def warm_up(self):
        # warm up stage, fill the replay with some episodes
        # it can either be human demos, or generated by the bc, or purely random
        obs, _ = self.train_env.reset()

        obs = filter_obs(obs)


        self.replay.new_episode(obs)
        total_reward = 0
        num_episode = 0
        steps = 0

        while True:
            if self.bc_policy is not None:
                # we have a BC policy
                with torch.no_grad(), utils.eval_mode(self.bc_policy):
                    action = self.bc_policy.act(obs, eval_mode=True)


                ## -------- Extra Layer --------------- ##
                # Check if the has reached z_min
                z_min = 0.009
                x, y, z, g = obs['prop'][0] + action.to("cuda")
                print(f"x: {x}, y: {y}, z: {z}, g: {g}")
                if z <= z_min:
                    action[2] = 0.005
                ## --------------------------------------- ##


            elif self.cfg.load_pretrained_agent or self.cfg.pretrain_num_epoch > 0:
                # the policy has been pretrained/initialized
                with torch.no_grad(), utils.eval_mode(self.agent):
                    action = self.agent.act(obs, eval_mode=True)
            else:
                action = torch.zeros(self.train_env.action_dim)
                action = action.uniform_(-1.0, 1.0)

            inp_action = filter_action(action)

            print(f"WarmUp Action {steps}: {inp_action}")
            obs, reward, terminal, success, image_obs = self.train_env.step(inp_action)

            obs = filter_obs(obs)


            if self.terminate_episode:
                print("\n---------------- Terminal key pressed --------------- \n")
                terminal = True
                self.terminate_episode = False
                time.sleep(2)





            reply = {"action": action}
            self.replay.add(obs, reply, reward, terminal, success, image_obs)

            steps += 1

            print(common_utils.get_mem_usage())
            if terminal:
                num_episode += 1
                # total_reward += self.train_env.episode_reward
                total_reward += reward
                if self.replay.size() < self.cfg.num_warm_up_episode:
                    self.replay.new_episode(obs)
                    print("---------- Episode Finished ----------")
                    obs, _ = self.train_env.reset()

                    obs = filter_obs(obs)

                    time.sleep(2)

                else:
                    break

        print(f"Warm up done. #episode: {self.replay.size()}")
        print(f"#episode from warmup: {num_episode}, #reward: {total_reward}")


    def on_press(self, key):
        try:
            if key.char == 't':  # Change 't' to the key you want to use
                self.terminate_episode = True
        except AttributeError:
            pass




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

        if self.replay.num_episode < self.cfg.num_warm_up_episode:
            self.warm_up()

        stopwatch = common_utils.Stopwatch()
        obs, _ = self.train_env.reset()

        obs = filter_obs(obs)

        print(f"filtered prop: ", obs["prop"].shape)
        # print(f"filtered image: ", obs["corner2"].shape)    

        self.replay.new_episode(obs)
        self.terminate_episode = False
        # terminal = False
        while self.global_step < self.cfg.num_train_step:
            ### act ###
            with stopwatch.time("act"), torch.no_grad(), utils.eval_mode(self.agent):
                stddev = utils.schedule(self.cfg.stddev_schedule, self.global_step)
                action = self.agent.act(obs, eval_mode=False, stddev=stddev)
                stat["data/stddev"].append(stddev)

                inp_action = filter_action(action)
            
            print(f"Train Action {self.train_step}: {inp_action}")

            ### env.step ###
            with stopwatch.time("env step"):
                obs, reward, terminal, success, image_obs = self.train_env.step(inp_action)
            
                obs = filter_obs(obs)


            if self.terminate_episode:
                print("\n---------------- Terminal key pressed --------------- \n")
                terminal = True
                self.terminate_episode = False
                time.sleep(2)

            with stopwatch.time("add"):
                assert isinstance(terminal, bool)
                reply = {"action": action}
                self.replay.add(obs, reply, reward, terminal, success, image_obs)
                self.global_step += 1


            if terminal:
                print(" ------- in terminal --------------")
                print("\n ------- Resetting --------- \n")
                with stopwatch.time("reset"):
                    self.global_episode += 1
                    stat["score/train_score"].append(float(success))
                    stat["data/episode_len"].append(self.train_env.time_step)

                    # reset env
                    obs, _ = self.train_env.reset()
                    terminal = False
                    obs = filter_obs(obs)

                    self.replay.new_episode(obs)
                time.sleep(2)

            ### logging ###
            if self.global_step % self.cfg.log_per_step == 0:
                self.log_and_save(stopwatch, stat, saver)

            ### train ###
            if self.global_step % self.cfg.update_freq == 0:
                with stopwatch.time("train"):
                    print("In rl_train")
                    self.rl_train(stat)
                    self.train_step += 1
        print("While loop ended")

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

        if self.replay.bc_replay is not None:
            stat["data/bc_replay_size"].append(self.replay.size(bc=True))

        # with stopwatch.time("eval"):
        #     eval_seed = (self.global_step // self.cfg.log_per_step) * self.cfg.num_eval_episode
        #     stat["eval/seed"].append(eval_seed)
        #     eval_score = self.eval(seed=eval_seed, policy=self.agent)
        #     stat["score/score"].append(eval_score)

        #     original_act_method = self.agent.cfg.act_method
        #     # if self.agent.cfg.act_method != "rl":
        #     #     with self.agent.override_act_method("rl"):
        #     #         rl_score = self.eval(seed=eval_seed, policy=self.agent)
        #     #         stat["score/score_rl"].append(rl_score)
        #     #         stat["score_diff/hybrid-rl"].append(eval_score - rl_score)

        #     if self.agent.cfg.act_method == "ibrl_soft":
        #         with self.agent.override_act_method("ibrl"):
        #             greedy_score = self.eval(seed=eval_seed, policy=self.agent)
        #             stat["score/greedy_score"].append(greedy_score)
        #             stat["score_diff/greedy-soft"].append(greedy_score - eval_score)
        #     assert self.agent.cfg.act_method == original_act_method

        saved = saver.save(self.agent.state_dict(), stat["score/train_score"].mean(), save_latest=True)
        stat.summary(self.global_step, reset=True)
        print(f"saved?: {saved}")
        stopwatch.summary(reset=True)
        print("total time:", common_utils.sec2str(stopwatch.total_time))
        print(common_utils.get_mem_usage())

    def rl_train(self, stat: common_utils.MultiCounter):
        stddev = utils.schedule(self.cfg.stddev_schedule, self.global_step)
        for i in range(self.cfg.num_critic_update):
            # print("in for loop")
            if self.cfg.mix_rl_rate < 1:
                # print("in if")
                rl_bsize = int(self.cfg.batch_size * self.cfg.mix_rl_rate)
                bc_bsize = self.cfg.batch_size - rl_bsize
                print(f"bc_bsize: {bc_bsize}, rl_bsize: {rl_bsize}")
                batch = self.replay.sample_rl_bc(rl_bsize, bc_bsize, "cuda")
            else:
                # print("in else")
                print(f"batch size: {self.cfg.batch_size}")
                batch = self.replay.sample(self.cfg.batch_size, "cuda")
            
            # in RED-Q, only update actor once
            update_actor = i == self.cfg.num_critic_update - 1
            # print("updated actor")
            bc_batch = None
            if update_actor and self.cfg.add_bc_loss:
                bc_batch = self.replay.sample_bc(self.cfg.batch_size, "cuda")

            metrics = self.agent.update(batch, stddev, update_actor, bc_batch, self.ref_agent)

            stat.append(metrics)
            stat["data/discount"].append(batch.bootstrap.mean().item())




def load_model(weight_file, device):
    cfg_path = os.path.join(os.path.dirname(weight_file), f"cfg.yaml")
    print(common_utils.wrap_ruler("config of loaded agent"))
    with open(cfg_path, "r") as f:
        print(f.read(), end="")
    print(common_utils.wrap_ruler(""))

    cfg = pyrallis.load(MainConfig, open(cfg_path, "r"))  # type: ignore
    cfg.preload_num_data = 0  # override this to avoid loading data
    workplace = Workspace(cfg, from_main=False)

    eval_env = workplace.eval_env
    eval_env_params = workplace.eval_env_params
    agent = workplace.agent
    state_dict = torch.load(weight_file)
    agent.load_state_dict(state_dict)

    if cfg.bc_policy:
        bc_policy = train_bc._load_model(cfg.bc_policy, eval_env, device)
        agent.add_bc_policy(bc_policy)

    agent = agent.to(device)
    return agent, eval_env, eval_env_params


def main():
    cfg = pyrallis.parse(config_class=MainConfig)  # type: ignore
    workspace = Workspace(cfg)
    if cfg.pretrain_num_epoch > 0:
        print("No Pretraining Specified")
    else:
        workspace.train()

    if cfg.use_wb:
        wandb.finish()

    assert False


if __name__ == "__main__":
    import wandb
    from rich.traceback import install

    install()
    os.environ["MUJOCO_GL"] = "egl"
    torch.backends.cudnn.allow_tf32 = True  # type: ignore
    torch.backends.cudnn.benchmark = True  # type: ignore
    main()