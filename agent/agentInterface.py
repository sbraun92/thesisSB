from env.roroDeck import *


class Agent:
    """
    Agent class. Does nothing but being the mother of all events.

    Subclasses:
        SARSA   -   implementation of the SARSA algorithm
        TDQ     -   implementation of the Time Difference Q Learning
        DQN     -   implementation of the Deep Q Learning (with Experience Replay and fixed target network)
    """
    def __init__(self):
        self.NUMBER_OF_EPISODES = None
        self.env = None
        self.strategies = ["Epsilon-greedy"]

    def save(self, path):
        raise NotImplementedError

    def load(self,path_to_file):
        raise NotImplementedError

    def execute(self, env = None):
        if env != None:
            self.env = env
            current_state = self.env.current_state
        else:
            current_state = self.env.reset()
        self.done = False
        #self.epReward = 0

        while not self.done:
            self.action = self.maxAction(current_state)
            current_state, self.reward, self.done, self.info = self.env.step(self.action)
            #self.epReward += self.reward
            if current_state.tobytes() not in self.q_table:
                self.q_table[current_state.tobytes()] = np.zeros(self.actionSpace_length)


    def train(self):
        raise NotImplementedError

    def choose_action(self,possible_actions):
        raise NotImplementedError

