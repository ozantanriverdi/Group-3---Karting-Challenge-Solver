from abc import ABC, abstractmethod

from training.memory import Memory

"""
Base class for agents
"""
class Agent(ABC):
    def __init__(self, date_string):
        self.date_string = date_string

    @abstractmethod
    def update(self, memory: Memory):
        pass

    @abstractmethod
    def predict_action(self, state):
        pass

    @abstractmethod
    def save(self):
        pass

    