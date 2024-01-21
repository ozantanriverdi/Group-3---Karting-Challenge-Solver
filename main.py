"""
 This script handles:
- the overall control flow,
- including setting up the environment
- initializing the PPO agent
- running the training loop
- evaluating the agent's performance.
 """
 # https://github.com/Unity-Technologies/ml-agents/blob/develop/docs/Python-LLAPI.md
import argparse
from mlagents_envs.environment import UnityEnvironment 
from configs.run_config import config
from utils.enums import EnvironmentEnum
from utils.enums import AgentEnum

from training.trainer import Trainer


if __name__ == '__main__':
   parser = argparse.ArgumentParser(description='Karting Challenge Solver')
   parser.add_argument('--no-graphics', action='store_true', help='Disable graphics')
   args = parser.parse_args()
   no_graphics = args.no_graphics

   if config["environment"] == EnvironmentEnum.ORIGINAL:
      env = UnityEnvironment(file_name='./environments/karting-challenge-build.app', no_graphics=no_graphics, seed=config["seed"])
   elif config["environment"] == EnvironmentEnum.ADVANCED:
      env = UnityEnvironment(file_name='./environments/karting-challenge-build-advanced.app', no_graphics=no_graphics, seed=config["seed"])
   elif config["environment"] == EnvironmentEnum.RIGHTTURN:
      env = UnityEnvironment(file_name='./environments/karting-challenge-build-rightturn.app', no_graphics=no_graphics, seed=config["seed"])
   elif config["environment"] == EnvironmentEnum.MIDDLE:
      env = UnityEnvironment(file_name='./environments/karting-challenge-build-middle.app', no_graphics=no_graphics, seed=config["seed"])
   elif config["environment"] == EnvironmentEnum.CURVES:
      env = UnityEnvironment(file_name='./environments/karting-challenge-build-curves.app', no_graphics=no_graphics, seed=config["seed"])
   else:
      raise Exception("No valid Environment specified")

   trainer = Trainer(env, interactive_plotting=False if no_graphics else True)
   agent_name = config["agent"]
   if agent_name == AgentEnum.PPO or agent_name == AgentEnum.RANDOM or agent_name == AgentEnum.PPOBETA:
      trainer.run_training()
   elif agent_name == AgentEnum.SB_PPO or agent_name == AgentEnum.SB_A2C or agent_name == AgentEnum.SB_SAC:
      trainer.run_training_sb()
   