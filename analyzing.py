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
import json
import os
from plotter.interactive_training_plotter import InteractiveTrainingPlotter
from plotter.html_plotter import HTMLPlotter
from utils.enums import PlottingSettingEnum

"""
Delets the logs of runs with too few episodes
"""
def delete_runs_with_too_few_episodes():
    log_folder = "logger/logs"
    policy_models_folder = "models/torch/policy_networks"
    value_models_folder = "models/torch/value_networks"
    unity_models_folder = "models/unity"
    for filename in os.listdir(log_folder):
        if filename.endswith(".json"):
            log_file_path = os.path.join(log_folder, filename)
            policy_model_file_path = os.path.join(policy_models_folder, filename[:15]+"_model")
            value_model_file_path = os.path.join(value_models_folder, filename[:15]+"_model")
            unity_model_file_path = os.path.join(unity_models_folder, filename[:15]+".onnx")
            with open(log_file_path) as file:
                data = json.load(file)
                episodes_reward_data = data.get("episodes_reward_data")
                if episodes_reward_data is not None and len(episodes_reward_data) < 50:
                    if(os.path.exists(log_file_path)):
                      os.remove(log_file_path)
                    if(os.path.exists(policy_model_file_path)):
                      os.remove(policy_model_file_path)
                    if(os.path.exists(value_model_file_path)):
                      os.remove(value_model_file_path)
                    if(os.path.exists(unity_model_file_path)):
                      os.remove(unity_model_file_path)
            

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Karting Challenge Solver')
    parser.add_argument('--no-plotting', action='store_true', help='Disable plotting')
    args = parser.parse_args()
    no_plotting = args.no_plotting

    delete_runs_with_too_few_episodes()

    if not no_plotting:
      plotter = InteractiveTrainingPlotter()
      #html_plotter = HTMLPlotter()
      #html_plotter.update_html_file() # Updates the html file

      #plotter.plot_multiple_trainings_boxplots(["20230711_005151.004_training_results", "20230711_005157.395_training_results"], plotting_setting=PlottingSettingEnum.REWARDS)
      plotter.plot_multiple_trainings(["20230705_002621.685_training_results", "20230627_174959_training_results"], plotting_setting=PlottingSettingEnum.REWARDS)
      # plotter.plot_multiple_trainings_with_same_config([['sb3-20230717_135408_training_results', 'sb3-20230717_151730_training_results'], # SB_PPO
      #                                                   ['sb3-20230717_183545_training_results', 'sb3-20230717_214457_training_results'], # SB_A2C
      #                                                   ['sb3-20230717_223907_training_results', 'sb3-20230718_050028_training_results'], # SB_SAC
      #                                                   ['20230718_121912.186_training_results', '20230718_122119.427_training_results'] # PPOBETA
      #                                                   ], PlottingSettingEnum.REWARDS, False)