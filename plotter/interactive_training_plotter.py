import json
from matplotlib import  pyplot as plt
import numpy as np
import pandas as pd
from configs.run_config import config
from configs.hyperparameters import hyperparameters
from utils.enums import PlottingSettingEnum
import mpld3



class InteractiveTrainingPlotter: 
    def __init__(self):
        self.plotting_setting = config["plotting_setting"]

    """
    Initializes the activation of interactive plotting during training
    """
    def init_active_plotting_while_training(self):
        self.fig, self.ax1 = plt.subplots()

        if self.plotting_setting == PlottingSettingEnum.REWARDS_AND_TRACK_DISTANCE:
            self.ax2 = self.ax1.twinx()
            self.ax1.set_title("Episode Stats")
        plt.ion()

    """
    Updates the plot with the passed data for rewards and track distance
    """
    def update_plot(self, reward_in_episodes, track_distance_in_episodes):
        if self.plotting_setting == PlottingSettingEnum.REWARDS or self.plotting_setting == PlottingSettingEnum.REWARDS_AND_TRACK_DISTANCE:
            self.ax1.plot(reward_in_episodes, color='blue')
            self.ax1.set_ylabel('Cumulative Rewards', color='blue')
        else:
            self.ax1.plot(track_distance_in_episodes, color='blue')
            self.ax1.set_ylabel('Track Distance', color='blue')


        self.ax1.tick_params('y', colors='blue')

        if self.plotting_setting == PlottingSettingEnum.REWARDS_AND_TRACK_DISTANCE:
            self.ax2.plot(track_distance_in_episodes, color='red')
            self.ax2.tick_params('y', colors='red')
            self.ax2.set_ylabel('Track Distance', color='red')

        
        plt.draw()
        plt.pause(0.001)

    def stop_plotting(self):
        plt.close()

    """
    Plots for the training are created
    """
    def plot_previous_training(self, training_name, as_html_plot):
        file = open(f"logger/logs/{training_name}.json")
        data = json.load(file)
        if as_html_plot: 
            fig, (self.ax1, self.ax2, self.ax3, self.ax4) = plt.subplots(4, 1)
        else:
            fig, (self.ax1, self.ax2) = plt.subplots(2, 1)
        if not as_html_plot:
            self.ax1.set_title(training_name)

        episodes_reward_data = data["episodes_reward_data"]        
        episodes_reward_data_mean = self.data_windows(episodes_reward_data).mean()
        episodes_reward_data_std = self.data_windows(episodes_reward_data).std()
        self.ax1.fill_between(episodes_reward_data_mean.index, episodes_reward_data_mean - episodes_reward_data_std,
                 episodes_reward_data_mean + episodes_reward_data_std, color='blue', alpha=0.3 if as_html_plot else 0.05)
        self.ax1.plot(episodes_reward_data_mean, color='blue')
        self.ax1.set_ylabel('Cumulative Rewards', color='blue', fontsize=16, fontweight='bold')
        self.ax1.tick_params('y', colors='blue')
        self.ax1.set_xlim([0,hyperparameters["num_episodes"]])
        self.ax1.set_ylim(top=300)

        episodes_track_distance_data = data["episodes_track_distance_data"]
        episodes_track_distance_data_mean = self.data_windows(episodes_track_distance_data).mean()
        episodes_track_distance_data_std = self.data_windows(episodes_track_distance_data).std()
        self.ax2.fill_between(episodes_track_distance_data_mean.index, episodes_track_distance_data_mean - episodes_track_distance_data_std,
                 episodes_track_distance_data_mean + episodes_track_distance_data_std, color='red', alpha=0.3 if as_html_plot else 0.05)
        self.ax2.plot(episodes_track_distance_data_mean, color='red')
        self.ax2.set_xlim([0,hyperparameters["num_episodes"]])
        self.ax2.set_ylim(top=1500)
        self.ax2.tick_params('y', colors='red')
        self.ax2.set_ylabel('Track Distance', color='red', fontsize=16, fontweight='bold')
        
        #Box-Plots are created
        if as_html_plot: 
            self.ax3.boxplot(episodes_reward_data, vert=False, patch_artist=True, boxprops=dict(facecolor="blue"))
            self.ax3.set_xlabel('Cumulative Rewards', color='blue', fontsize=16, fontweight='bold')
            self.ax4.boxplot(episodes_track_distance_data, vert=False, patch_artist=True, boxprops=dict(facecolor="red"))
            self.ax4.set_xlabel('Track Distance', color='red', fontsize=16, fontweight='bold')
            self.ax3.set_yticks([])
            self.ax3.set_xlim([-50, 300])
            self.ax4.set_xlim([0, 1000])
            self.ax4.set_yticks([])
            

        file.close()

        if as_html_plot: 
            fig.set_size_inches(5, 8)
            fig.tight_layout(pad=3.0)
            plt.close()
            return mpld3.fig_to_html(fig)
        
        else:
            plt.show()

    """
    Plots and compares the episode rewards for multiple training sessions.
    """
    def plot_multiple_trainings(self, trainings_to_compare: list, plotting_setting: PlottingSettingEnum):

        if plotting_setting == PlottingSettingEnum.REWARDS_AND_TRACK_DISTANCE:
            raise Exception('Not valid for multiple Training plotting')
        
        for training_name in trainings_to_compare:
            file = open(f"logger/logs/{training_name}.json")
            data = json.load(file)
            training_results = data["episodes_reward_data" if plotting_setting == PlottingSettingEnum.REWARDS else "episodes_track_distance_data"]
            training_results_mean = pd.Series(training_results).rolling(10, min_periods=10).mean()
            training_results_std = pd.Series(training_results).rolling(10, min_periods=10).std()
            plt.fill_between(training_results_mean.index, training_results_mean - training_results_std,
                 training_results_mean + training_results_std, alpha=0.1)
            plt.plot(training_results_mean, label=f"{training_name}")
            file.close()
        plt.xlabel('Episode')
        plt.ylabel('Cumulative Reward' if plotting_setting == PlottingSettingEnum.REWARDS else 'Track Distance')
        plt.xlim([0,300])
        plt.ylim(top=300 if plotting_setting == PlottingSettingEnum.REWARDS else 1500)
        plt.title("Compared Stats")
        plt.legend()
        plt.show()
    
    """
    Compares the episode rewards for multiple training sessions with Box-Plots
    """
    def plot_multiple_trainings_boxplots(self, trainings_to_compare: list, plotting_setting: PlottingSettingEnum):
        

        if plotting_setting == PlottingSettingEnum.REWARDS_AND_TRACK_DISTANCE:
            raise Exception('Not valid for multiple Training plotting')
        
        training_results_list=[]
        for training_name in trainings_to_compare:
            file = open(f"logger/logs/{training_name}.json")
            data = json.load(file)
            training_results = data["episodes_reward_data" if plotting_setting == PlottingSettingEnum.REWARDS else "episodes_track_distance_data"]
            training_results_list.append(training_results)
            file.close()
        plt.ylim(bottom=-50 if plotting_setting == PlottingSettingEnum.REWARDS else 0)
        plt.ylim(top=300 if plotting_setting == PlottingSettingEnum.REWARDS else 1000)
        plt.ylabel('Cumulative Reward' if plotting_setting == PlottingSettingEnum.REWARDS else 'Track Distance')
        plt.title("Compared Stats")
        plt.boxplot(training_results_list, labels=trainings_to_compare)
        plt.show()

    """
    Creates a plot that compares multiple training runs with the same configuration
    """
    def plot_multiple_trainings_with_same_config(self, trainings_to_compare_per_config, plotting_setting: PlottingSettingEnum, as_html_plot: bool):
        fig = plt.figure()
        plt.xlim([0,300])
        plt.ylim(top=300 if plotting_setting == PlottingSettingEnum.REWARDS else 1500)
        config_index = 0
        for trainings_with_same_config in trainings_to_compare_per_config:
            rewards_data = []
            for training_name in trainings_with_same_config:
                file = open(f"logger/logs/{training_name}.json")
                data = json.load(file)
                rewards = data["episodes_reward_data" if plotting_setting == PlottingSettingEnum.REWARDS else "episodes_track_distance_data"]
                rewards_data.append(rewards)

                print(training_name, len(rewards))
            print(' ')
            # Calculate mean and standard deviation of rewards data
            mean_rewards = np.mean(rewards_data, axis=0)
            std_rewards = np.std(rewards_data, axis=0)
            mean_rewards = np.convolve(mean_rewards, np.ones(10) / 10, mode='valid') # smoothing window = 10
            std_rewards = np.convolve(std_rewards, np.ones(10) / 10, mode='valid')

            # Plot the mean and standard deviation

            episodes = range(1, len(mean_rewards) + 1)
            if plotting_setting == PlottingSettingEnum.REWARDS:
                plt.plot(episodes, mean_rewards, label='Mean Rewards (Config: '+str(config_index)+')')
                plt.fill_between(episodes, mean_rewards - std_rewards, mean_rewards + std_rewards, alpha=0.2, label='Standard Deviation (Config: '+str(config_index)+')')
                plt.xlabel('Episodes')
                plt.ylabel('Cumulative Rewards')
            elif plotting_setting == PlottingSettingEnum.TRACK_DISTANCE:
                plt.plot(episodes, mean_rewards, label='Mean Track Distance (Config: '+str(config_index)+')')
                plt.fill_between(episodes, mean_rewards - std_rewards, mean_rewards + std_rewards, alpha=0.2, label='Standard Deviation (Config: '+str(config_index)+')')
                plt.xlabel('Episodes')
                plt.ylabel('Track Distance')
            elif plotting_setting == PlottingSettingEnum.REWARDS_AND_TRACK_DISTANCE:
                plt.plot(episodes, mean_rewards, label='Mean Rewards (Config: '+str(config_index)+')')
                plt.fill_between(episodes, mean_rewards - std_rewards, mean_rewards + std_rewards, alpha=0.2, label='Standard Deviation (Config: '+str(config_index)+')')
                plt.xlabel('Episodes')
                plt.ylabel('Rewards/Track Distance')
            config_index += 1

        plt.legend()

        if as_html_plot == True: 
            fig.set_size_inches(5, 4)
            html_content = mpld3.fig_to_html(plt.gcf())
            plt.close()
            return html_content
        
        else:
            plt.show()

        plt.show()



    def keep_plot_open_after_training(self):
        pass # not implemented yet


    """
    used for defining moving windows 
    """
    def data_windows(self, data):
        return pd.Series(data).rolling(10, min_periods=10)