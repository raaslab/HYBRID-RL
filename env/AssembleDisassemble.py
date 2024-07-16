import gym
import metaworld
from gym import spaces

class AssembleDisassembleEnv(gym.Env):
    def __init__(self):
        super(AssembleDisassembleEnv, self).__init__()
        # Initialize MetaWorld
        self.mt1 = metaworld.MT1('assembly-v2')  # Multi-task env, single task
        self.mt2 = metaworld.MT1('disassemble-v2')
        
        self.task1 = self.mt1.train_classes['assembly-v2']()
        self.task2 = self.mt2.train_classes['disassemble-v2']()
        
        # self.task1_env = self.task1.train_tasks.sample(n_tasks=1)[0].make()
        # self.task2_env = self.task2.train_tasks.sample(n_tasks=1)[0].make()
        
        self.task1_env = metaworld.envs.ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE["assembly-v2-goal-observable"]()
        self.task2_env = metaworld.envs.ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE["disassemble-v2-goal-observable"]()

        self.current_task = 1
        
        # This should be adjusted based on the observation and action spaces
        self.observation_space = self.task1_env.observation_space
        self.action_space = self.task1_env.action_space

    def step(self, action):
        if self.current_task == 1:
            observation, reward, done, info = self.task1_env.step(action)
            if done:
                self.current_task = 2  # Switch to disassemble after assemble is done
        else:
            observation, reward, done, info = self.task2_env.step(action)
        
        return observation, reward, done, info

    def reset(self):
        if self.current_task == 1:
            return self.task1_env.reset()
        else:
            return self.task2_env.reset()

    def render(self, mode='human'):
        if self.current_task == 1:
            return self.task1_env.render(mode)
        else:
            return self.task2_env.render(mode)
