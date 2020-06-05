from env.roroDeck import *
import pickle
import os

class Agent:
    """
    Agent class. Does nothing but being the mother of all events.

    Subclasses:
        SARSA   -   implementation of the SARSA algorithm
        TDQ     -   implementation of the Time Difference Q Learning
        DQN     -   implementation of the Deep Q Learning (with Experience Replay and fixed target network)
    """

    def __init__(self):
        self.number_of_episodes = None
        self.env = None
        self.strategies = ["Epsilon-greedy"]
        self.training_time = 0
        self.action_space_length = None
        self.MAX_REWARD = 24
        self.BENCHMARK_REWARD = 14
        self.eps_history = None

    def save_model(self, path, output_type='pickle'):
        try:
            if output_type == 'pickle':
                pickle.dump(self.q_table, open(path + '.p', "wb"))

            else:
                with open(path + '.csv', 'w') as f:
                    for key in self.q_table.keys():
                        f.write("%s,%s\n" % (key, self.q_table[key]))
        finally:
            if output_type == 'pickle':
                logging.getLogger("log1").error("Could not save pickle file to+ " + path)
            else:
                logging.getLogger("log1").error("Could not save csv file to+ " + path)

        path += '_training_history\\'
        try:
            os.makedirs(path , exist_ok=True)

            pickle.dump(self.total_rewards, open(path + 'rewards.p', "wb"))
            pickle.dump(self.eps_history, open(path + 'eps_history.p', "wb"))
        finally:
            logging.getLogger("log1").error("Could not save training history as pickle file to+ " + path)

    def load_model(self, path_to_file):
        raise NotImplementedError

    def execute(self, env=None):
        if env is not None:
            self.env = env
            current_state = self.env.current_state
        else:
            current_state = self.env.reset()
        done = False

        while not done:
            action = self.max_action(current_state, self.env.possible_actions)
            current_state, reward, done, _ = self.env.step(action)
            if current_state.tobytes() not in self.q_table:
                self.q_table[current_state.tobytes()] = np.zeros(self.action_space_length)

    def train(self):
        raise NotImplementedError

    def choose_action(self, possible_actions):
        raise NotImplementedError
#TODO check this
    def max_action(self, state, possible_actions):
        possible_actions = self.action_ix[possible_actions]
        prediction = self.predict(state, possible_actions)
        positions_of_best_possible_action = np.argmax(prediction)

        return possible_actions[positions_of_best_possible_action]

    # TODO unify tabular and NN
    def predict(self, state, action=None):
        if self.q_table is not None:
            try:
                if action is None:
                    return self.q_table[state.tobytes()]
                else:
                    return self.q_table[state.tobytes()][action]
            except:
                if action is None:
                    return np.zeros(self.env.action_space)
                else:
                    return 0 if isinstance(action, (np.integer, int)) else np.zeros(len(action))

    #TODO delete
    def get_action(self, state, epsilon = None):
        pass