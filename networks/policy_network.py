import torch.nn as nn
import torch
import os

from configs.run_config import config

class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(PolicyNetwork, self).__init__()
        torch.manual_seed(config["seed"])
        
        self.network = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            )
        self.mean_linear = nn.Linear(64, action_dim)
        self.log_std_linear = nn.Linear(64, action_dim)
        self.sigmoid = nn.Sigmoid() # value of range [0,1]
        self.tanh = nn.Tanh() # value of range [-1,1]
    
    """
    Forward pass through the network
    """
    def forward(self, state):
        x = self.network(state)
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = self.tanh(log_std)
    
        mean = torch.cat((self.tanh(mean[:, :1]), self.sigmoid(mean[:, 1:])), dim=1)
        return mean, log_std

    """
    Saves the current state (weights and parameters) of a model
    """
    def save_checkpoint(self, date_string):
        save_path = os.path.join("models/torch/policy_networks", date_string + "_model")
        torch.save(self.state_dict(), save_path)

    """
    Loads the current state (weights and parameters) of a model
    """
    def load_checkpoint(self, model_name):
        loading_path = os.path.join("models/torch/policy_networks", model_name)
        self.load_state_dict(torch.load(loading_path))

