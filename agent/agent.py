from env.roroDeck import *

"""
Agent class. Does nothing but being the mother of all events.

Subclasses:
    SARSA   -   implementation of the SARSA algorithm
    TDQ     -   implementation of the Time Difference Q Learning
    DQN     -   implementation of the Deep Q Learning (with Experience Replay and fixed target netowrk)
"""
class Agent:
    def __init__(self):
        self.NUMBER_OF_EPISODES = None
        self.env = None

    def save(self, path):
        raise NotImplementedError

    def load(self,path_to_file):
        raise NotImplementedError

    def execute(self):
        raise NotImplementedError

    def train(self):
        raise NotImplementedError
