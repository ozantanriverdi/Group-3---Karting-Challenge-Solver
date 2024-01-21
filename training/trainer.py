import datetime
from mlagents_envs.envs.unity_gym_env import UnityToGymWrapper
from mlagents_envs.environment import UnityEnvironment
from codecarbon import track_emissions
from utils.enums import AgentEnum
from agents.ppo_agent import PPOAgent
from agents.random_agent import RandomAgent
from configs.run_config import config
from utils.stable_baselines_utils import PreprocessedEnv
from configs.hyperparameters import hyperparameters
from logger.karting_logger import Logger
from plotter.interactive_training_plotter import InteractiveTrainingPlotter
from training.memory import Memory

# import only if Agent is of Stable-Baseline type because imports don't work on cip-pool 
if config["agent"] == AgentEnum.SB_PPO:
    from agents.stable_baselines_ppo import SbPPOAgent
    from stable_baselines3.common.env_checker import check_env

if config["agent"] == AgentEnum.SB_A2C:
    from agents.stable_baselines_a2c import SbA2CAgent
    from stable_baselines3.common.env_checker import check_env

if config["agent"] == AgentEnum.SB_SAC:
    from agents.stable_baselines_sac import SbSACAgent
    from stable_baselines3.common.env_checker import check_env


class Trainer:
    def __init__(self, env: UnityEnvironment, interactive_plotting: bool):
        self.date_string = f"{datetime.datetime.now().strftime('%Y%m%d_%H%M%S.%f')[:-3]}"
        self.env = UnityToGymWrapper(env, allow_multiple_obs=True)
        
        if (config["agent"] == AgentEnum.SB_SAC or config["agent"] == AgentEnum.SB_PPO or config["agent"] == AgentEnum.SB_A2C):
            #Pre-processed environment for sb3
            self.env_sb = PreprocessedEnv(self.env)
            # Check the pre-processed environment if necessary
            #check_env(self.env_sb)
        
        self.interactive_plotting = interactive_plotting
        self.set_agent()
        existing_model = config["use_existing_model"]

        if(existing_model):
            self.agent.load(existing_model)

    """
    Starts the training
    """
    @track_emissions(project_name="karting", output_file="logger/codecarbon/emissions.csv")
    def run_training(self):
        num_episodes = hyperparameters["num_episodes"]
        self.env.reset()
        memory = Memory()
        if self.interactive_plotting:
            interactive_training_plotter = InteractiveTrainingPlotter()
            interactive_training_plotter.init_active_plotting_while_training()
        logger = Logger(self.date_string)
        logger.create_run_log_file()
        episodes_reward_data = []
        episodes_track_distance_data = []
        n_steps = 0
        highest_score = 0

        for episode in range(num_episodes):
            state = self.env.reset()
            episode_reward = 0
            track_distance = 0
            done = False

            while not done:
                action, action_log_prob, value = self.agent.predict_action(state)
                next_state, reward, done, _ = self.env.step(action)
                n_steps += 1
                memory.store_memory(state[0], action, reward, action_log_prob, value, done)

                # state[0][1] is the direction. Either negative or positive
                if state[0][1] > 0: 
                    track_distance += 1
                if state[0][1] < 0:
                    track_distance -= 1

                # Parameter update when horizon is reached
                if n_steps % hyperparameters["horizon"] == 0:
                    self.agent.update(memory, n_steps)
                    print("Parameter Update!")

                episode_reward += reward
                state = next_state

            if episode_reward > highest_score:
                highest_score = episode_reward
                self.agent.save() # save model when episode_reward is higher than highest_score in training
                print("High Score:", highest_score)
                print(f"Achieved {episode_reward} in {n_steps} steps.")

            # Training will restart if no more than 10 rewards have been achieved after 150 episodes
            if config["automatic_restart"] and episode == 150 and highest_score < 10:
                interactive_training_plotter.stop_plotting()
                print("restart training")
                self.restart_training()

            episodes_reward_data.append(episode_reward)
            episodes_track_distance_data.append(track_distance)
            logger.log_stats_in_episode(episode_reward, track_distance)

            
            if self.interactive_plotting:
                interactive_training_plotter.update_plot(episodes_reward_data, episodes_track_distance_data)

        print(f"Run ended with {n_steps}")
        

        if self.interactive_plotting:
            interactive_training_plotter.keep_plot_open_after_training()
     
        logger.log_run_results(episodes_reward_data, episodes_track_distance_data)
        logger.finish_logging()
        
        self.env.close()


    @track_emissions(project_name="karting", output_file="logger/codecarbon/emissions.csv")
    def run_training_sb(self):
        self.agent.update()
        self.env.close()


    """"
    Training restarts
    """
    def restart_training(self):
        self.set_agent()
        self.run_training()

    """
    Any agent can be selected
    """
    def set_agent(self):
        agent_name = config["agent"]
        if agent_name == AgentEnum.PPO or agent_name == AgentEnum.PPOBETA:
            self.agent = PPOAgent(date_string=self.date_string)
        # if agent_name == AgentEnum.PPOBETA:
        #     self.agent = PPOBetaAgent(date_string=self.date_string)

        if agent_name == AgentEnum.RANDOM:
            self.agent = RandomAgent(self.date_string)
        if agent_name == AgentEnum.SB_PPO:
            self.agent = SbPPOAgent(self.date_string, self.env_sb)
        if agent_name == AgentEnum.SB_A2C:
            self.agent = SbA2CAgent(self.date_string, self.env_sb)
        if agent_name == AgentEnum.SB_SAC:
            self.agent = SbSACAgent(self.date_string, self.env_sb)
