import numpy as np
from configs.hyperparameters import hyperparameters


class Memory:
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.values = []
        self.dones = []
        
    """
    Split training data into smaller batches to train the model in steps for efficient processing
    """
    def generate_batches(self):
        batch_size = hyperparameters["batch_size"]
        
        n_states = len(self.states)
        # Find start indices of each batch
        batch_start = np.arange(0, n_states, batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        # Take batch_size chunks of the shuffled indices
        batches = [indices[i:i + batch_size] for i in batch_start]
    
        return np.array(self.states), np.array(self.actions), np.array(self.log_probs), self.values,\
                np.array(self.rewards), np.array(self.dones), batches
    
    """
    Saving training data
    """
    def store_memory(self, state, action, reward, action_log_prob, value, done):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.log_probs.append(action_log_prob)
        self.values.append(value)
        self.dones.append(done)

    """
    Clear the storage
    """
    def clear_memory(self):
        del self.states[:]
        del self.actions[:]
        del self.rewards[:]
        del self.log_probs[:]
        del self.values[:]
        del self.dones[:]