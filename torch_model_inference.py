import torch
import numpy as np
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.envs.unity_gym_env import UnityToGymWrapper
from networks.policy_network_beta import PolicyNetwork
from utils.enums import EnvironmentEnum
from configs.run_config import config
    
if __name__ == '__main__':

    model = PolicyNetwork(48, 3)
    # Specify the model set for inference mode
    model.load_checkpoint("20230710_190432.671_model")
    model.eval()

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
    obs = env_gym.reset()

   # Run inference on observations
    for _ in range(1000):  # Replace 1000 with the desired number of steps
        obs_array = np.array(obs)
        obs_tensor = torch.from_numpy(obs_array).float().unsqueeze(0)  # Convert the observation to a tensor
        alpha, beta, mean, log_std = model(obs_tensor)

        mean_alpha_beta = (alpha / (alpha + beta)) * 2 - 1

        mean_alpha_beta = mean_alpha_beta.view(1, -1)
        mean = mean.view(1, -1)  # Reshape the mean tensor
        
        # Example action post-processing
        if mean[:, 1] < 0.01:
            mean[:, 1] = 0
        else:
            mean[:, 0] = 0
            print(mean)
        
        actions = torch.cat((mean_alpha_beta, mean), dim=1)
        
        # Convert the actions tensor to a numpy array
        actions_np = actions.squeeze().detach().numpy()
        actions_np = actions_np.reshape((1, 3))

        # Step through the environment with the chosen actions
        obs, reward, done, info = env_gym.step(actions_np)

        # Reset the environment if the episode is done
        if done:
            obs = env_gym.reset()  

    # Close the environment
    env_gym.close()