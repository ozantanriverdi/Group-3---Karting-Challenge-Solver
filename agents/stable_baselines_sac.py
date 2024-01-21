import os
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import StopTrainingOnMaxEpisodes
from stable_baselines3.common.monitor import Monitor
from configs.hyperparameters_SbSAC import hyperparameters
from agents.agent import Agent
from utils.stable_baselines_utils import CustomCallback


class SbSACAgent(Agent):

    def __init__(self, date_string, environment):
        super().__init__(date_string)

        logs = f"logs_sb3/SAC"
        self.save_path = os.path.join("models/torch/policy_networks", date_string + "_model_SbSAC")
        os.makedirs(logs, exist_ok=True)

        # Monitor wrapper for more tracking capabilities and other hyperparameters of the algorithm
        self.env = Monitor(environment)
        self.learning_rate = hyperparameters["learning_rate"]
        self.buffer_size = hyperparameters["buffer_size"]
        self.learning_starts = hyperparameters["learning_starts"]
        self.batch_size = hyperparameters["batch_size"]
        self.tau = hyperparameters["tau"]
        self.gamma = hyperparameters["gamma"]
        self.train_freq = hyperparameters["train_freq"]
        self.gradient_steps = hyperparameters["gradient_steps"]
        self.action_noise = hyperparameters["action_noise"]
        self.replay_buffer_class = hyperparameters["replay_buffer_class"]
        self.replay_buffer_kwargs = hyperparameters["replay_buffer_kwargs"]
        self.optimize_memory_usage = hyperparameters["optimize_memory_usage"]
        self.ent_coef = hyperparameters["ent_coef"]
        self.target_update_interval = hyperparameters["target_update_interval"]
        self.target_entropy = hyperparameters["target_entropy"]
        self.use_sde = hyperparameters["use_sde"]
        self.sde_sample_freq = hyperparameters["sde_sample_freq"]
        self.use_sde_at_warmup = hyperparameters["use_sde_at_warmup"]
        self.stats_window_size = hyperparameters["stats_window_size"]
        self.tensorboard_log = hyperparameters["tensorboard_log"]
        self.policy_kwargs = hyperparameters["policy_kwargs"]
        self.verbose = hyperparameters["verbose"]
        self.seed = hyperparameters["seed"]
        self.device = hyperparameters["device"]
        # self._init_setup_model = hyperparameters["_init_setup_model"]


        self.sb_sac_agent = SAC("MlpPolicy", self.env, self.learning_rate, self.buffer_size, self.learning_starts,
                                self.batch_size, self.tau, self.gamma, self.train_freq, self.gradient_steps,
                                self.action_noise, self.replay_buffer_class, self.replay_buffer_kwargs, 
                                self.optimize_memory_usage, self.ent_coef, self.target_update_interval, 
                                self.target_entropy, self.use_sde, self.sde_sample_freq, self.use_sde_at_warmup, 
                                self.stats_window_size, self.tensorboard_log, self.policy_kwargs, self.verbose, 
                                self.seed, self.device, True)


    '''
    Trains the model with callbacks enabled
    '''
    def update(self):
        num_episodes = hyperparameters["num_episodes"]
        
        #Stop the training after "num_episodes" episodes
        callback_max_episodes = StopTrainingOnMaxEpisodes(max_episodes=num_episodes, verbose=1)
        custom_callback = CustomCallback(self.env)

        self.sb_sac_agent.learn(99999999, tb_log_name="SAC", progress_bar=True, 
                                callback=[custom_callback, callback_max_episodes])
        self.save()


    '''
    Predicts actions for a (preloaded) model
    '''
    def predict_action(self, state):
        state_arr = np.array(state).flatten()
        action = self.sb_sac_agent.predict(state_arr)
        return np.array(action[0])


    '''
    Saves the trained model
    '''
    def save(self):
        self.sb_sac_agent.save(path=self.save_path)


    '''
    Load a pretrained model
    '''
    def load(self, model_name):
        loading_path = os.path.join("models/torch/policy_networks", model_name)
        self.sb_sac_agent.load(loading_path)