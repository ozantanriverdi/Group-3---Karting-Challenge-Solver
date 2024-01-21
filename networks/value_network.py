import torch
import torch.nn as nn
import os
from configs.run_config import config

class ValueNetwork(nn.Module):
    def __init__(self, state_dim):
        super(ValueNetwork, self).__init__()
        torch.manual_seed(config["seed"])

        self.network = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    
    """
    Forward pass through the network
    """
    def forward(self, state):
        value = self.network(state)
        
        return value
    
    """
    Saves the current state of a model
    """
    def save_checkpoint(self, date_string):
        save_path = os.path.join("models/torch/value_networks", date_string + "_model")
        torch.save(self.state_dict(), save_path)
    """
    Loads the current state of a model
    """
    def load_checkpoint(self, model_name):
        loading_path = os.path.join("models/torch/value_networks", model_name)
        self.load_state_dict(torch.load(loading_path))