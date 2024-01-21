import os
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import StopTrainingOnMaxEpisodes
from stable_baselines3.common.monitor import Monitor
from configs.hyperparameters_SbPPO import hyperparameters
from agents.agent import Agent
from utils.stable_baselines_utils import CustomCallback


class SbPPOAgent(Agent):

    def __init__(self, date_string, environment):
        super().__init__(date_string)

        logs = f"logs_sb3/PPO"
        self.save_path = os.path.join("models/torch/policy_networks", date_string + "_model_SbPPO")
        os.makedirs(logs, exist_ok=True)

        # Monitor wrapper for more tracking capabilities and other hyperparameters of the algorithm
        self.env = Monitor(environment)
        self.learning_rate = hyperparameters["learning_rate"]
        self.horizon = hyperparameters["horizon"]
        self.batch_size = hyperparameters["batch_size"]
        self.num_epochs = hyperparameters["num_epochs"]
        self.discount_factor = hyperparameters["discount_factor"]
        self.gae_lambda = hyperparameters["gae_lambda"]
        self.epsilon = hyperparameters["epsilon"]
        self.clip_range_vf = hyperparameters["clip_range_vf"]
        self.normalize_advantage = hyperparameters["normalize_advantage"]
        self.ent_coef = hyperparameters["ent_coef"]
        self.vf_coef = hyperparameters["vf_coef"]
        self.max_grad_norm = hyperparameters["max_grad_norm"]
        self.use_sde = hyperparameters["use_sde"]
        self.sde_sample_freq = hyperparameters["sde_sample_freq"]
        self.target_kl = hyperparameters["target_kl"]
        self.stats_window_size = hyperparameters["stats_window_size"]
        self.tensorboard_log = hyperparameters["tensorboard_log"]
        self.policy_kwargs = hyperparameters["policy_kwargs"]
        self.verbose = hyperparameters["verbose"]
        self.seed = hyperparameters["seed"]
        self.device = hyperparameters["device"]
        # self._init_setup_model = hyperparameters["_init_setup_model"]


        self.sb_ppo_agent = PPO("MlpPolicy", self.env, self.learning_rate, self.horizon, self.batch_size,
                                self.num_epochs, self.discount_factor, self.gae_lambda, self.epsilon, self.clip_range_vf,
                                self.normalize_advantage, self.ent_coef, self.vf_coef, self.max_grad_norm, self.use_sde,
                                self.sde_sample_freq, self.target_kl, self.stats_window_size, self.tensorboard_log,
                                self.policy_kwargs, self.verbose, self.seed, self.device, True)


    '''
    Trains the model with callbacks enabled
    '''
    def update(self):
        num_episodes = hyperparameters["num_episodes"]
        
        #Stop the training after "num_episodes" episodes
        callback_max_episodes = StopTrainingOnMaxEpisodes(max_episodes=num_episodes, verbose=1)
        custom_callback = CustomCallback(self.env)

        self.sb_ppo_agent.learn(99999999, tb_log_name="PPO", progress_bar=True, 
                                callback=[custom_callback, callback_max_episodes])
        self.save()


    '''
    Predicts actions for a (preloaded) model
    '''
    def predict_action(self, state):
        state_arr = np.array(state).flatten()
        action = self.sb_ppo_agent.predict(state_arr)
        return np.array(action[0])


    '''
    Saves the trained model
    '''
    def save(self):
        self.sb_ppo_agent.save(path=self.save_path)


    '''
    Load a pretrained model
    '''
    def load(self, model_name):
        loading_path = os.path.join("models/torch/policy_networks", model_name)
        self.sb_ppo_agent.load(loading_path)