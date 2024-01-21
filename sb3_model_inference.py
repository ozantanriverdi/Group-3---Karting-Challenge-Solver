import torch
import numpy as np
import os
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.envs.unity_gym_env import UnityToGymWrapper
from stable_baselines3 import A2C, PPO, SAC
from networks.policy_network_beta import PolicyNetwork
from utils.enums import EnvironmentEnum
from configs.run_config import config
from utils.stable_baselines_utils import PreprocessedEnv
    
if __name__ == '__main__':

    # Create the Unity environment
    if config["environment"] == EnvironmentEnum.ORIGINAL:
        env = UnityEnvironment(file_name='./environments/karting-challenge-build.app', seed=config["seed"])
    elif config["environment"] == EnvironmentEnum.ADVANCED:
        env = UnityEnvironment(file_name='./environments/karting-challenge-build-advanced.app', seed=config["seed"])
    elif config["environment"] == EnvironmentEnum.RIGHTTURN:
        env = UnityEnvironment(file_name='./environments/karting-challenge-build-rightturn.app', seed=config["seed"])
    elif config["environment"] == EnvironmentEnum.MIDDLE:
        env = UnityEnvironment(file_name='./environments/karting-challenge-build-middle.app', seed=config["seed"])
    elif config["environment"] == EnvironmentEnum.CURVES:
        env = UnityEnvironment(file_name='./environments/karting-challenge-build-curves.app', seed=config["seed"])
    else:
        raise Exception("No valid Environment specified")
    env_gym = UnityToGymWrapper(env, allow_multiple_obs=True)
    env_sb = PreprocessedEnv(env_gym)


    # Enter the model name below
    model_name = "20230712_191458.009_model_SbPPO"
    loading_path = os.path.join("models/torch/policy_networks", model_name)
    if "A2C" in model_name:
        model = A2C.load(loading_path, env=env_sb, print_system_info=True)
    elif "PPO" in model_name:
        model = PPO.load(loading_path, env=env_sb, print_system_info=True)
    elif "SAC" in model_name:
        model = SAC.load(loading_path, env=env_sb, print_system_info=True)
    
    obs = env_sb.reset()

   # Run inference on observations
    for _ in range(1000):  # Replace 1000 with the desired number of steps
        obs = np.array(obs)
        #obs_tensor = torch.from_numpy(obs_array).float().unsqueeze(0)  # Convert the observation to a tensor
        action, _states = model.predict(obs, deterministic=True)

        # Step through the environment with the chosen actions
        obs, rewards, dones, info = env_sb.step(action)

        # Reset the environment if the episode is done
        if dones:
            obs = env_sb.reset()

    # Close the environment
    env_sb.close()