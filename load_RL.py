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

from train_rl_hardw_hyrl import load_model





agent, eval_env, eval_env_params = load_model("exps/rl/train_rl_hardw_hyrl_run2/model0.pt", "cuda")

print("I am RL Re-Loaded!!")
