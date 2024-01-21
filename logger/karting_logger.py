import datetime
import json
from configs.hyperparameters import hyperparameters
from configs.run_config import config

class Logger: 
    def __init__(self, date_string):

        self.date_string = date_string
        self.log_filename = f"{self.date_string}_training_logs.txt"
        self.results_filename = f"{self.date_string}_training_results.json"

    """
     Function is used to create a log file for the training results
    """
    def create_run_log_file(self):
        results = {
                    "hyperparameters": hyperparameters,
                    "run_configs": config,
                    "episodes_reward_data": [],
                    "episodes_track_distance_data": []
                }
        with open("logger/logs/"+self.results_filename, "w") as results_file:
            json.dump(results, results_file,indent=4)
    """
    Data of a single episode are written to logs
    """
    def log_stats_in_episode(self, rewards_in_episode, track_distance_in_episode):
        with open("logger/logs/"+self.results_filename,'r+') as file:
          # First we load existing data into a dict.
            file_data = json.load(file)
            # Join new_data with file_data inside emp_details
            file_data["episodes_reward_data"].append(rewards_in_episode)
            file_data["episodes_track_distance_data"].append(track_distance_in_episode)
            # Sets file's current position at offset.
            file.seek(0)
            # convert back to json.
            json.dump(file_data, file, indent = 4)

    """
    Data of all episodes are written to logs
    """
    def log_run_results(self, episodes_reward_data, episodes_track_distance_data): 
        results = {
                    "hyperparameters": hyperparameters,
                    "run_configs": config,
                    "episodes_reward_data": episodes_reward_data,
                    "episodes_track_distance_data": episodes_track_distance_data
                }
        with open("logger/logs/"+self.results_filename, "w") as results_file:
            json.dump(results, results_file,indent=4)

    def log_run_results_sb3(self, episodes_reward_data, episodes_track_distance_data, 
                        episode_steps, episode_durations, total_steps, max_reward): 
        results = {
                    "hyperparameters": hyperparameters,
                    "run_configs": config,
                    "episodes_reward_data": episodes_reward_data,
                    "episodes_track_distance_data": episodes_track_distance_data,
                    "episodes_step_data": episode_steps,
                    "episodes_duration_data": episode_durations,
                    "run_total_step_data": total_steps,
                    "run_max_reward_data": max_reward
                }
        with open("logger/logs/"+self.results_filename, "w") as results_file:
            json.dump(results, results_file,indent=4)


    def finish_logging(self):
        pass
