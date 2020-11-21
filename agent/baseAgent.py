import pickle
import os
import numpy as np
import logging
from pathlib import Path


class Agent:
    """
    The Agent class works as an interface for all agent classes
    The implementations provided are for SARSA and Q-Learning to avoid code duplications

    Subclasses:
        SARSA          -   implementation of the SARSA algorithm
        Q-Learning     -   implementation of the Time Difference Q Learning
        DQ-Learning    -   implementation of the Deep Q Learning (with Experience Replay and fixed target network)
    """

    def __init__(self):
        pass

    def get_info(self):
        """Transform variables to info-string """

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
        """ Method to train an agent -> must be implemented"""

        raise NotImplementedError

    def save_model(self, path, info, name=None, output_type='pickle'):
        """Save a model - here specifically Q-table as Pickle-file or csv-file """
        path = str(path)+'_' + info
        if self.additional_info is not None:
            path = path+'_'+ str(self.additional_info)
        try:
            if output_type == 'pickle':
                pickle.dump(self.q_table, open(str(path) + '.p', "wb"))
            else:
                with open(str(path) + '.csv', 'w') as f:
                    for key in self.q_table.keys():
                        f.write("%s,%s\n" % (key, self.q_table[key]))
        except:
            if output_type == 'pickle':
                logging.getLogger(__name__).error("Could not save model as pickle file to+ " + str(path))
            else:
                logging.getLogger(__name__).error("Could not save model as csv file to+ " + str(path))

        path = Path(path+'_training_history/')
        os.makedirs(path, exist_ok=True)
        try:
            pickle.dump(self.total_rewards, open(Path(path).joinpath('rewards_history.p'), "wb"))
            pickle.dump(self.eps_history, open(Path(path).joinpath('eps_history_history.p'), "wb"))
            pickle.dump(self.loaded_cargo, open(Path(path).joinpath('cargo_loaded_history.p'), "wb"))
        except:
            logging.getLogger(__name__).error("Could not save training history as pickle file to" + str(path))

    def load_model(self, path_to_file):
        """ Method to load a RL system -> must be implemented"""
        raise NotImplementedError

    def execute(self, env=None):
        """finish an episode (or do one completely) by picking always the best action"""

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


    def max_action(self, state, possible_actions):
        """Determine best action based on current estimate"""

        possible_actions = self.action_ix[possible_actions]
        prediction = self.predict(state, possible_actions)
        positions_of_best_possible_action = np.argmax(prediction)

        return possible_actions[positions_of_best_possible_action]

    def predict(self, state, action):
        """Predict Q-values for for a state or for one state/action-pair"""

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
