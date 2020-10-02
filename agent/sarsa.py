from agent.tdq import TDQLearning
from env.roroDeck import RoRoDeck
import numpy as np
import matplotlib.pyplot as plt
# import tqdm
import time
import logging
import pickle
import csv

from agent.BasicAgent import Agent

np.random.seed(0)


class SARSA(Agent):
    def __init__(self, env=None, path=None, number_of_episodes=20000, orig=True, GAMMA=0.999, additional_info=None):
        #super().__init__()
        # help only for timing
        self.orig = orig
        logging.info("Initialise SARSA Agent")
        self.number_of_episodes = number_of_episodes
        self.q_table = {}
        self.EPSdec = 0.999995
        self.EPSmin = 0.001
        self.GAMMA = GAMMA
        self.path = path
        self.env = env
        self.eps_history = []
        self.ALPHA = 0.1
        self.EPS = 1.0
        self.MAX_IT = 400
        if self.env.open_ai_structure:
            self.action_space_length = self.env.action_space.n
        else:
            self.action_space_length = len(self.env.action_space)
        self.action_ix = np.arange(self.action_space_length)
        self.epReward = 0
        self.total_rewards = np.zeros(self.number_of_episodes)
        self.state_expansion = np.zeros(self.number_of_episodes)
        self.loaded_cargo = np.zeros(self.number_of_episodes)
        self.additional_info = additional_info


    # TODO Output QTable
    # TODO Load QTable

    def train(self):
        logging.getLogger('log1').info("prepare training...")

        start = time.time()

        observation = self.env.reset()

        self.q_table[observation.tobytes()] = np.zeros(self.action_space_length)

        logging.getLogger('log1').info(self.get_info())


        print("Start Training Process")
        logging.getLogger('log1').info("Start training process")
        for i in range(self.number_of_episodes):
            observation = self.env.reset()
            done = False
            ep_reward = 0
            steps = 0
            current_action = self.max_action(observation, self.env.possible_actions)\
                if np.random.random() < (1 - self.EPS) \
                else self.env.action_space_sample()
            while not done:
                # Show for visualisation the last training epoch

                observation_, reward, done, info = self.env.step(current_action)

                steps += 1

                # Log Loading Sequence
                if i == self.number_of_episodes - 1:
                    logging.getLogger('log2').info(
                        "Current Lane:" + str(observation[-1]) + " Action:" + str(current_action))

                if observation_.tobytes() not in self.q_table:
                    self.q_table[observation_.tobytes()] = np.zeros(self.action_space_length)

                ep_reward += reward

                # SARSA with Epsilon-Greedy
                if not done:
                    action_ = self.max_action(observation_,self.env.possible_actions) if np.random.random() < (1 - self.EPS) \
                        else self.env.action_space_sample()

                    self.q_table[observation.tobytes()][current_action] += self.ALPHA * (
                            reward + self.GAMMA * self.q_table[observation_.tobytes()][action_]
                            - self.q_table[observation.tobytes()][current_action])

                    current_action = action_

                # Value of Terminal State is zero
                else:
                    self.q_table[observation.tobytes()][current_action] += self.ALPHA * (
                            reward - self.q_table[observation.tobytes()][current_action])

                observation = observation_

                #TODO add for other agents
                if i == self.number_of_episodes - 1 and done:
                    logging.getLogger('log1').info('Set environment for the final episode to deterministic')
                    self.env.stochastic = False


                if i == self.number_of_episodes - 1 and done:
                    if i == self.number_of_episodes - 1 and done:
                        logging.getLogger(__name__).info(self.env._get_grid_representations())
                        print("The reward of the last training episode was {} (was deterministic)".format(str(ep_reward)))
                    if self.path != None:
                        self.env.save_stowage_plan(self.path)

            # TODO set to .format and move to Agent Interface
            logging.getLogger('log1').info('It. {:7d} \t'.format(i)
                                           + 'EPS: {} \t'.format(round(self.EPS, 4))
                                           + 'Reward: {}'.format(round(ep_reward, 2)))


            # Epsilon decreases lineary during training TODO 50 is arbitrary
            if 1. - i / (self.number_of_episodes - 50.) > self.EPSmin:
                self.EPS -= 1. / (self.number_of_episodes - 50.)
            else:
                self.EPS = self.EPSmin


            if i == self.number_of_episodes - 2:
                logging.getLogger(__name__).info('Set env to deterministic for last it. {}'.format(i+1)) if self.env.stochastic else None
                self.env.stochastic = False
                self.EPS = 0.0


            self.eps_history.append(self.EPS)

            self.total_rewards[i] = ep_reward
            self.state_expansion[i] = len(self.q_table.keys())
            self.loaded_cargo[i] = steps

            if i % 500 == 0 and i > 0:
                avg_reward = np.mean(self.total_rewards[max(0, i - 100):(i + 1)])
                # std_reward = np.std(self.totalRewards[max(0, i - 100):(i + 1)])
                print('episode ', i, '\tscore %.2f' % ep_reward, '\tavg. score %.2f' % avg_reward)

        self.training_time = time.time() - start
        ttime_minutes = (int(self.training_time / 60), round(self.training_time % 60, 0))

        print('Finished training after {} min {} sec. \n'.format(ttime_minutes[0], ttime_minutes[1]))
        logging.getLogger(__name__).info('Training Time: {} m. {} s. \n\t\t\t--> ({} s.)'.format(ttime_minutes[0],
                                                                                                ttime_minutes[1],
                                                                                                self.training_time))



        if self.path is not None:
            print('Save output to: \n' + self.path + '\n')
            self.env.save_stowage_plan(self.path)

        return self.q_table, self.total_rewards, self.loaded_cargo, np.array(self.eps_history), self.state_expansion


    def load_model(self, path):
        try:
            self.q_table = pickle.load(open(path, "rb"))
        except:
            logging.getLogger("log1").error("Could not load pickle file")

    # TODO work with super method
    def save_model(self, path, file_format='pickle'):
        self.q_table["ModelParam"] = {"Algorithm": "SARSA",
                                      "GAMMA": self.GAMMA,
                                      "ALPHA": self.ALPHA,
                                      "Episodes": self.number_of_episodes,
                                      "EnvLanes:": self.env.lanes,
                                      "EnvRows": self.env.rows,
                                      "VehicleData": self.env.vehicle_data,
                                      "TrainingTime": self.training_time}
        info = "SARSA" + "_L" + str(self.env.lanes) + "-R" + str(self.env.rows)

        super().save_model(path + info)

'''

class SARSA(TDQLearning):
    def __init__(self, env=None, path=None, number_of_episodes=20000, orig=True, GAMMA=0.999, additional_info=None):
        logging.info("Initialise SARSA Agent")
        self.number_of_episodes = number_of_episodes
        self.q_table = {}
        self.EPSdec = 0.999995
        self.EPSmin = 0.001
        self.GAMMA = GAMMA
        self.path = path
        self.env = env
        self.eps_history = []
        self.ALPHA = 0.1
        self.EPS = 1.0
        self.MAX_IT = 400
        if self.env.open_ai_structure:
            self.action_space_length = self.env.action_space.n
        else:
            self.action_space_length = len(self.env.action_space)
        self.action_ix = np.arange(self.action_space_length)
        self.epReward = 0
        self.total_rewards = np.zeros(self.number_of_episodes)
        self.state_expansion = np.zeros(self.number_of_episodes)
        self.loaded_cargo = np.zeros(self.number_of_episodes)

    def train(self):
        logging.getLogger('log1').info("prepare training...")

        start = time.time()

        observation = self.env.reset()

        self.q_table[observation.tobytes()] = np.zeros(self.action_space_length)

        logging.getLogger('log1').info(self.get_info())

        print("Start Training Process")
        logging.getLogger('log1').info("Start training process")
        for i in range(self.number_of_episodes):
            observation = self.env.reset()
            done = False
            ep_reward = 0
            steps = 0
            current_action = self.max_action(observation, self.env.possible_actions) \
                if np.random.random() < (1 - self.EPS) \
                else self.env.action_space_sample()
            while not done:
                # Show for visualisation the last training epoch

                observation_, reward, done, info = self.env.step(current_action)

                steps += 1

                # Log Loading Sequence
                if i == self.number_of_episodes - 1:
                    logging.getLogger('log2').info(
                        "Current Lane:" + str(observation[-1]) + " Action:" + str(current_action))

                if observation_.tobytes() not in self.q_table:
                    self.q_table[observation_.tobytes()] = np.zeros(self.action_space_length)

                ep_reward += reward

                # SARSA with Epsilon-Greedy
                if not done:
                    action_ = self.max_action(observation_, self.env.possible_actions) if np.random.random() < (
                                1 - self.EPS) \
                        else self.env.action_space_sample()

                    self.q_table[observation.tobytes()][current_action] += self.ALPHA * (
                            reward + self.GAMMA * self.q_table[observation_.tobytes()][action_]
                            - self.q_table[observation.tobytes()][current_action])

                    current_action = action_

                # Value of Terminal State is zero
                else:
                    self.q_table[observation.tobytes()][current_action] += self.ALPHA * (
                            reward - self.q_table[observation.tobytes()][current_action])

                observation = observation_

                # TODO add for other agents
                if i == self.number_of_episodes - 2 and done:
                    logging.getLogger('log1').info('Set environment for the final episode to deterministic')
                    self.env.stochastic = False

                if i == self.number_of_episodes - 1 and done:
                    if i == self.number_of_episodes - 1 and done:
                        logging.getLogger(__name__).info(self.env._get_grid_representations())
                        print(
                            "The reward of the last training episode was {} (was deterministic)".format(str(ep_reward)))
                    if self.path != None:
                        self.env.save_stowage_plan(self.path)

            # TODO set to .format and move to Agent Interface
            logging.getLogger('log1').info('It. {:7d} \t'.format(i)
                                           + 'EPS: {} \t'.format(round(self.EPS, 4))
                                           + 'Reward: {}'.format(round(ep_reward, 2)))

            if i == self.number_of_episodes - 2:
                logging.getLogger(__name__).info(
                    'Set env to deterministic for last it. {}'.format(i + 1)) if self.env.stochastic else None
                self.env.stochastic = False

            # Epsilon decreases lineary during training TODO 50 is arbitrary
            if 1. - i / (self.number_of_episodes - 100) > self.EPSmin:
                self.EPS -= 1. / (self.number_of_episodes - 100)
            else:
                self.EPS = self.EPSmin

            self.eps_history.append(self.EPS)

            self.total_rewards[i] = ep_reward
            self.state_expansion[i] = len(self.q_table.keys())
            self.loaded_cargo[i] = steps

            if i % 500 == 0 and i > 0:
                avg_reward = np.mean(self.total_rewards[max(0, i - 100):(i + 1)])
                # std_reward = np.std(self.totalRewards[max(0, i - 100):(i + 1)])
                print('episode ', i, '\tscore %.2f' % ep_reward, '\tavg. score %.2f' % avg_reward)

        self.training_time = time.time() - start
        ttime_minutes = (int(self.training_time / 60), round(self.training_time % 60, 0))

        print('Finished training after {} min {} sec. \n'.format(ttime_minutes[0], ttime_minutes[1]))
        logging.getLogger(__name__).info('Training Time: {} m. {} s. \n\t\t\t--> ({} s.)'.format(ttime_minutes[0],
                                                                                                 ttime_minutes[1],
                                                                                                 self.training_time))

        if self.path is not None:
            print('Save output to: \n' + self.path + '\n')
            self.env.save_stowage_plan(self.path)

        return self.q_table, self.total_rewards, self.loaded_cargo, np.array(self.eps_history), self.state_expansion
'''