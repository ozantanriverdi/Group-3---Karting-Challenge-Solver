from ast import List
import random
from mlagents_envs.base_env import ActionTuple  
import numpy as np
from agents.agent import Agent
from training.memory import Memory

class RandomAgent(Agent):
    def __init__(self, date_string):
        pass

    def update(self, memory: Memory):
        pass

    def predict_action(self, state):
        return self.define_random_action_tuple(), 0
    
    def save(self):
        pass
    
    # Returns a list of the form [steer_value, accelerate_value, break_value] (e.g. [0, 0.73, 0])
    def define_random_action_tuple(self):
        random_action_tuple = np.zeros(3, dtype=np.float32)
        random_action_tuple[0] = random.uniform(-1, 1)  # steer
        random_action_tuple[1] = random.uniform(0, 1)   # accelerate
        random_action_tuple[2] = random.uniform(0, 1)   # brake
        return random_action_tuple