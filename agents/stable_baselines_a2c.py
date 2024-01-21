import os
import numpy as np
from stable_baselines3 import A2C
from stable_baselines3.common.callbacks import StopTrainingOnMaxEpisodes
from stable_baselines3.common.monitor import Monitor
from configs.hyperparameters_SbA2C import hyperparameters
from agents.agent import Agent
from utils.stable_baselines_utils import CustomCallback


class SbA2CAgent(Agent):

    def __init__(self, date_string, environment):
        super().__init__(date_string)

        logs = f"logs_sb3/A2C"
        self.save_path = os.path.join("models/torch/policy_networks", date_string + "_model_SbA2C")
        os.makedirs(logs, exist_ok=True)

        # Monitor wrapper for more tracking capabilities and other hyperparameters of the algorithm
        self.env = Monitor(environment)
        self.learning_rate = hyperparameters["learning_rate"]
        self.n_steps = hyperparameters["n_steps"]
        self.discount_factor = hyperparameters["discount_factor"]
        self.gae_lambda = hyperparameters["gae_lambda"]
        self.ent_coef = hyperparameters["ent_coef"]
        self.vf_coef = hyperparameters["vf_coef"]
        self.max_grad_norm = hyperparameters["max_grad_norm"]
        self.rms_prop_eps = hyperparameters["rms_prop_eps"]
        self.use_rms_prop = hyperparameters["use_rms_prop"]
        self.use_sde = hyperparameters["use_sde"]
        self.sde_sample_freq = hyperparameters["sde_sample_freq"]
        self.normalize_advantage = hyperparameters["normalize_advantage"]
        self.stats_window_size = hyperparameters["stats_window_size"]
        self.tensorboard_log = hyperparameters["tensorboard_log"]
        self.policy_kwargs = hyperparameters["policy_kwargs"]
        self.verbose = hyperparameters["verbose"]
        self.seed = hyperparameters["seed"]
        self.device = hyperparameters["device"]
        # self._init_setup_model = hyperparameters["_init_setup_model"]


        self.sb_a2c_agent = A2C("MlpPolicy", self.env, self.learning_rate, self.n_steps, self.discount_factor,
                                self.gae_lambda, self.ent_coef, self.vf_coef, self.max_grad_norm, self.rms_prop_eps,
                                self.use_rms_prop, self.use_sde, self.sde_sample_freq, self.normalize_advantage, 
                                self.stats_window_size, self.tensorboard_log, self.policy_kwargs, self.verbose, 
                                self.seed, self.device, True) 


    '''
    Train the model with callbacks enabled
    '''
    def update(self):
        #tb_log_name = f"A2C-{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        num_episodes = hyperparameters["num_episodes"]
        
        #Stop the training after "num_episodes" episodes
        callback_max_episodes = StopTrainingOnMaxEpisodes(max_episodes=num_episodes, verbose=1)
        custom_callback = CustomCallback(self.env)

        self.sb_a2c_agent.learn(99999999, tb_log_name="A2C", progress_bar=True, 
                                callback=[custom_callback, callback_max_episodes])
        self.save()

    '''
    Predict actions for a (preloaded) model
    '''
    def predict_action(self, state):
        state_arr = np.array(state).flatten()
        action = self.sb_a2c_agent.predict(state_arr)
        return np.array(action[0])


    '''
    Save the trained model
    '''
    def save(self):
        self.sb_a2c_agent.save(path=self.save_path)


    '''
    Load a pretrained model
    '''
    def load(self, model_name):
        loading_path = os.path.join("models/torch/policy_networks", model_name)
        self.sb_a2c_agent.load(loading_path)