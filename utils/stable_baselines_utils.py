import gym
import numpy as np
from gym.spaces import Box
from stable_baselines3.common.callbacks import BaseCallback
import datetime
from logger.karting_logger import Logger
from utils.enums import AgentEnum
from configs.run_config import config




'''
Gym environment wrapper
'''
class PreprocessedEnv(gym.Env):
    def __init__(self, unity_env):
        self.unity_env = unity_env
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(48,), dtype=np.float32)
        low = [-1, 0, 0]
        high = [1, 1, 1]
        self.action_space = Box(low=np.array(low), high=np.array(high), shape=(3,), dtype=np.float32)
        
        #self.action_space = self.unity_env.action_space
        #print("Action Space: ", self.action_space) # Action Space:  Box([-1. -1. -1.], [1. 1. 1.], (3,), float32)

    def step(self, action):
        state, reward, done, info = self.unity_env.step(action)
        preprocessed_state = state[0]
        return preprocessed_state, float(reward), done, info
    
    def reset(self):
        state = self.unity_env.reset()
        preprocessed_state = state[0]
        return preprocessed_state

'''
Defines Custom Callbacks to be able to track
and log the agent behavior
'''
class CustomCallback(BaseCallback):
    def __init__(self, env, verbose=0):
        super(CustomCallback, self).__init__(verbose)
        self.date_string = f"sb3-{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.env = env
        self.episode_distance = 0
        self.run_distances = []
        self.agent_name = config["agent"]


    def _on_step(self) -> bool:
        done = self.locals["dones"]
        if self.agent_name == AgentEnum.SB_SAC:
            state = self.locals["new_obs"]
        else:
            state = self.locals["obs_tensor"]
        if not done and state[0][1] > 0:
            self.episode_distance += 1
        elif not done and state[0][1] < 0:
            self.episode_distance -= 1
        elif done:
            self.run_distances.append(self.episode_distance)
            self.episode_distance = 0

        return True

    def _on_training_end(self) -> None:
        
        episode_rewards = self.env.get_episode_rewards()
        episode_distance = self.run_distances
        episode_steps = self.env.get_episode_lengths()
        episode_durations = self.env.get_episode_times()
        total_steps = self.env.get_total_steps()
        
        max_reward = max(episode_rewards)
        print("Max Reward: ",max_reward)

        logger = Logger(self.date_string)
        logger.create_run_log_file()
        logger.log_run_results_sb3(episode_rewards, episode_distance, episode_steps, episode_durations, total_steps, max_reward)

        print("Episode Steps: ", episode_steps)
        print("Episode Rewards: ", episode_rewards)
        print("Episode Durations: ", episode_durations)
        print("Total Steps: ", total_steps)
        print("Episode Distances: ", self.run_distances)
