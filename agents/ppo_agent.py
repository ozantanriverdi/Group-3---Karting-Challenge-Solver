import torch
import torch.optim as optim
from agents.agent import Agent
from torch.distributions import MultivariateNormal, Beta
import torch.nn as nn
import numpy
from networks.policy_network import PolicyNetwork as PolicyNetwork
from networks.policy_network_beta import PolicyNetwork as PolicyNetworkBeta
from networks.value_network import ValueNetwork
from configs.hyperparameters import hyperparameters
from configs.run_config import config
from training.memory import Memory
from utils.enums import AgentEnum, EntropySettingEnum


class PPOAgent(Agent):
    

    def __init__(self, date_string):
        super().__init__(date_string)
        torch.manual_seed(config["seed"])

        # Initialize hyperparameters
        self.learning_rate = hyperparameters["learning_rate"]
        self.discount_factor = hyperparameters["discount_factor"]
        self.epsilon = hyperparameters["epsilon"]
        self.gae_lambda = hyperparameters["gae_lambda"]
        self.num_epochs = hyperparameters["num_epochs"]
        self.entropy_coefficient = hyperparameters["entropy_coefficient_const"]
        self.entropy_coefficient_init = hyperparameters["entropy_coefficient_init"]
        self.entropy_coefficient_target = hyperparameters["entropy_coefficient_target"]
        self.entropy_decay_steps = hyperparameters["entropy_decay_steps"]

        # Initialize networks and optimizers
        if config['agent'] == AgentEnum.PPO:
            self.policy_network = PolicyNetwork(state_dim=48, action_dim=3)     
        if config['agent'] == AgentEnum.PPOBETA:
            self.policy_network = PolicyNetworkBeta(state_dim=48, action_dim=3)
        self.value_network = ValueNetwork(state_dim=48)
        self.policy_optimizer = optim.Adam(self.policy_network.parameters(), lr=self.learning_rate)
        self.value_optimizer = optim.Adam(self.value_network.parameters(), lr=self.learning_rate)
        self.MseLoss = nn.MSELoss()


    """
    Update the agent's networks using the collected experiences.
    """
    def update(self, memory: Memory, n_steps):

        for _ in range(self.num_epochs):
            state_arr, action_arr, log_prob_arr, values, rewards_arr, dones_arr, batches = memory.generate_batches()
            
            advantages = self.compute_advantages(rewards_arr, values, dones_arr)
        
            values = torch.tensor(values, dtype=torch.float)

            for batch in batches:
                states = torch.tensor(state_arr[batch], dtype=torch.float)
                old_probs = torch.tensor(log_prob_arr[batch], dtype=torch.float)
                actions = torch.tensor(action_arr[batch], dtype=torch.float)

                policy_loss = self.compute_policy_loss(states, actions, old_probs, advantages, batch, n_steps)

                value = self.value_network(states)
                value = torch.squeeze(value)
                returns = advantages[batch] + values[batch]
                value_loss = self.compute_value_loss(value, returns)

                self.optimize_policy(policy_loss)
                self.optimize_value_function(value_loss)

        memory.clear_memory()


    """
    Predict an action given a state.
    Returns the action, its log probability and its value.
    """
    def predict_action(self, state):
        state_arr = numpy.array(state)
        state = torch.tensor(state_arr)
        if config['agent'] == AgentEnum.PPO: 
            mean, log_std = self.policy_network(state)
            std = torch.exp(log_std)
            
            normal = MultivariateNormal(mean, torch.diag_embed(std))
            action = normal.sample()
            if(config['zero_brake'] == True):
                action[0, 2] = 0
            action_log_prob = normal.log_prob(action)

            value = self.value_network(state)
            value = torch.squeeze(value).item()
            return action.detach().cpu().numpy()[0], action_log_prob.detach().cpu().numpy()[0], value
        
        if config['agent'] == AgentEnum.PPOBETA: 
            alpha, beta, mean, log_std = self.policy_network(state)
            std = torch.exp(log_std)
            
            action = torch.Tensor(3,)
            beta_dist = Beta(alpha, beta)
            action[0] = beta_dist.sample().squeeze() # Steering with beta distribution

            normal = MultivariateNormal(mean, torch.diag_embed(std))
            action[1:] = normal.sample().squeeze() # acceleration and brake with gaussian distribution
            action_log_prob = beta_dist.log_prob((action[0]))+normal.log_prob(action[1:])
            action[0] = action[0] * 2 - 1 # To bring it in [-1,1]

            value = self.value_network(state)
            value = torch.squeeze(value).item()
            return action.detach().cpu().numpy(), action_log_prob.detach().cpu().item(), value

    """
    Compute advantages based on the returns and old values.
    """
    def compute_advantages(self, rewards, values, dones) -> torch.Tensor:
        advantages = numpy.zeros(len(rewards), dtype=numpy.float32)
        
        for t in range(len(rewards)-1):
            discount = 1
            a_t = 0
            for k in range(t, len(rewards) - 1):
                a_t += discount * (rewards[k] + self.discount_factor * values[k + 1] * (1 - int(dones[k])) - values[k])
                discount *= self.discount_factor * self.gae_lambda
            advantages[t] = a_t
        advantages = torch.tensor(advantages)
        return advantages
    
    """
    Compute the policy loss based on old states, actions, log probabilities, and advantages.
    """
    def compute_policy_loss(self, old_states: torch.Tensor, old_actions: torch.Tensor, old_log_probs: torch.Tensor, advantages: torch.Tensor, batch, n_steps) -> torch.Tensor:
        if config['agent'] == AgentEnum.PPO: 
            mean, log_std = self.policy_network(old_states)
            normal = MultivariateNormal(mean, torch.diag_embed(log_std.exp()))
            new_log_probs = normal.log_prob(old_actions)
            ratio = (new_log_probs - old_log_probs).exp()
            entropy = normal.entropy().mean()

        if config['agent'] == AgentEnum.PPOBETA: 
            alpha, beta, mean, log_std = self.policy_network(old_states)
            beta_dist = Beta(alpha, beta)
            normal = MultivariateNormal(mean, torch.diag_embed(log_std.exp()))
            # *0.5 + 0.5 to bring it in [0,1]
            new_log_probs = beta_dist.log_prob(old_actions[..., 0] * 0.5 + 0.5) +\
                            normal.log_prob(old_actions[..., 1:])
            entropy = (beta_dist.entropy().mean() + 2 * normal.entropy().mean()) / 3

        # Surrogate loss
        surr1 = ratio * advantages[batch]
        surr2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * advantages[batch]
        policy_loss = -torch.min(surr1, surr2).mean()

        # Entropy regularization
        if config["entropy_setting"] == EntropySettingEnum.FIXED:
            entropy_coefficient = self.entropy_coefficient
        elif config["entropy_setting"] == EntropySettingEnum.SCHEDULED:
            entropy_coefficient = self.entropy_decay(n_steps)
        else:
            raise Exception("No valid Entropy Setting specified")

        policy_loss -= entropy_coefficient * entropy

        return policy_loss


    """
    Optimize the policy network using the policy loss.
    """
    def optimize_policy(self, policy_loss: torch.Tensor):
        self.policy_optimizer.zero_grad() 
        policy_loss.backward(retain_graph=True)  
        self.policy_optimizer.step()  


    """
    Compute the value loss based on old states and returns.
    """
    def compute_value_loss(self, value: torch.Tensor, returns: torch.Tensor) -> torch.Tensor:
        return self.MseLoss(value, returns)


    """
    Optimize the value function using the value loss.
    """
    def optimize_value_function(self, value_loss: torch.Tensor):
        self.value_optimizer.zero_grad()
        value_loss.backward(retain_graph=True)
        self.value_optimizer.step()


    '''
    Calculates the entropy coefficient to be used in Entropy Regularization
    depending on the current number of steps.
    '''
    def entropy_decay(self, n_steps):
        decay_rate = (self.entropy_coefficient_init - self.entropy_coefficient_target) / self.entropy_decay_steps
        entropy_coefficient = max(self.entropy_coefficient_init - n_steps * decay_rate, self.entropy_coefficient_target)
        return entropy_coefficient


    """
    Save model parameters of both networks
    """
    def save(self):
        print('... saving models ...')
        self.policy_network.save_checkpoint(self.date_string)
        self.value_network.save_checkpoint(self.date_string)

    """
    Load an already trained model to be able to train it further
    """    
    def load(self, model):
        print('... loading models ...')
        self.policy_network.load_checkpoint(model_name=model)
        self.value_network.load_checkpoint(model_name=model)