from env.roroDeck import *
import pickle
import os


class Agent:
    """
    The Agent class is providing an interface for all other agent classes

    Subclasses:
        SARSA   -   implementation of the SARSA algorithm
        TDQ     -   implementation of the Time Difference Q Learning
        DQN     -   implementation of the Deep Q Learning (with Experience Replay and fixed target network)
        DDQN    -   wrapper class of Double Deep Q Learning with Duelling Network and prioritised experience replay
                    (core implementation is done by Stable Baselines (2018))
    """

    def __init__(self):
        pass

    def get_info(self):
        info_str = 'Information on Agent:\n'
        for key, value in vars(self).items():
            # if value is dict, numpy array or list print its size
            if isinstance(value, (list, np.ndarray, dict)):
                key += ' size'
                value = len(value)
            # variable tab space for clean printing
            tab_space = ':' + '\t' * (5 - int(len(key) / 8)) if 5 - int(len(key) / 8) > 0 else ':\t'
            info_str += key + tab_space + str(value) + '\n'
        return info_str

    def train(self):
        pass

    def save_model(self, path, name=None, output_type='pickle'):
        if self.additional_info is not None:
            path += '_' + str(self.additional_info)
        try:
            if output_type == 'pickle':
                pickle.dump(self.q_table, open(path + '.p', "wb"))
            else:
                with open(path + '.csv', 'w') as f:
                    for key in self.q_table.keys():
                        f.write("%s,%s\n" % (key, self.q_table[key]))
        except:
            if output_type == 'pickle':
                logging.getLogger(__name__).error("Could not save model as pickle file to+ " + path)
            else:
                logging.getLogger(__name__).error("Could not save model as csv file to+ " + path)

        path += '_training_history\\'
        os.makedirs(path, exist_ok=True)
        try:
            pickle.dump(self.total_rewards, open(path + 'rewards.p', "wb"))
            pickle.dump(self.eps_history, open(path + 'eps_history.p', "wb"))
            pickle.dump(self.loaded_cargo, open(path + 'cargo_loaded_history.p', "wb"))
        except:
            logging.getLogger(__name__).error("Could not save training history as pickle file to" + path)

    def load_model(self, path_to_file):
        raise NotImplementedError

    def execute(self, env=None):
        if env is not None:
            self.env = env
            current_state = self.env.current_state
        else:
            current_state = self.env.reset()
        done = False
        # env.stochastic = False
        while not done:
            action = self.max_action(current_state, self.env.possible_actions)
            current_state, reward, done, _ = self.env.step(action)
            if current_state.tobytes() not in self.q_table:
                self.q_table[current_state.tobytes()] = np.zeros(self.action_space_length)

    def train(self):
        raise NotImplementedError

    def choose_action(self, possible_actions):
        raise NotImplementedError

    # TODO check this
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

    # TODO delete
    def get_action(self, state, epsilon=None):
        pass
