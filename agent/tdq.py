from env.roroDeck import RoRoDeck
import numpy as np
import matplotlib.pyplot as plt
# import tqdm
import time
import logging
import pickle
import csv

from agent.BasicAgent import Agent


class TDQLearning(Agent):
    def __init__(self, env=None, path=None, number_of_episodes=20000, orig=True, GAMMA=0.999):
        np.random.seed(0)
        # help only for timing
        self.orig = orig
        logging.info("Initialise TD-Q-Learning Agent")
        self.number_of_episodes = number_of_episodes
        self.q_table = {}
        self.EPSdec = 0.9999
        self.EPSmin = 0.001
        self.GAMMA = GAMMA
        self.env = env
        self.path = path
        self.env = env

        self.ALPHA = 0.1
        self.EPS = 1.0
        self.MAX_IT = 400
        try:
            self.action_space_length = len(self.env.action_space)
        except:
            self.action_space_length = self.env.action_space.n
        self.action_ix = np.arange(self.action_space_length)
        self.total_rewards = np.zeros(self.number_of_episodes)
        self.state_expansion = np.zeros(self.number_of_episodes)
        self.stepsToExit = np.zeros(self.number_of_episodes)
        self.eps_history = []

        # ix_Actions = np.arange(len(env.actionSpace))
        # print(env.actionSpace)
        # self.action_list = []

    # TODO Output QTable
    # TODO Load QTable

    def train(self):
        logging.getLogger('log1').info("prepare training...")

        start = time.time()

        observation = self.env.reset()

        self.q_table[observation.tobytes()] = np.zeros(self.action_space_length)

        logging.getLogger('log1').info("Use param: ALPHA: " + str(self.ALPHA) + " GAMMA: " + str(self.GAMMA))



        print("Start Training Process")
        logging.getLogger('log1').info("Start training process")
        for i in range(self.number_of_episodes):
            done = False
            ep_reward = 0
            observation = self.env.reset()
            steps = 0

            while not done:
                # Show for visualisation the last training epoch

                action = self.max_action(observation, self.env.possible_actions) if np.random.random() < (1 - self.EPS) \
                    else self.env.action_space_sample()
                # self.env.render()
                # TODO delete
                # if i == self.numGames - 1:
                #    print("----")
                #    print(self.q_table[self.observation.tobytes()])
                #    print(self.action)

                observation_, reward, done, _ = self.env.step(action)
                steps += 1

                # Log Loading Sequence
                if i == self.number_of_episodes - 1:
                    logging.getLogger('log2').info(
                        "Current Lane:" + str(observation[-1]) + " Action:" + str(action))

                if observation_.tobytes() not in self.q_table:
                    self.q_table[observation_.tobytes()] = np.zeros(self.action_space_length)

                ep_reward += reward

                # TD-Q-Learning with Epsilon-Greedy
                if not done:
                    action_ = self.max_action(observation_,self.env.possible_actions)

                    self.q_table[observation.tobytes()][action] += self.ALPHA * (
                            reward + self.GAMMA * self.q_table[observation_.tobytes()][action_]
                            - self.q_table[observation.tobytes()][action])

                # Value of Terminal State is zero
                else:
                    self.q_table[observation.tobytes()][action] += self.ALPHA * (
                            reward - self.q_table[observation.tobytes()][action])

                # if not self.done:
                #    self.q_table[self.observation.tobytes()][self.action] = self.reward \
                #                                    + self.GAMMA * self.q_table[self.observation_.tobytes()][self.action_]

                observation = observation_

                if i == self.number_of_episodes - 1 and done:
                    logging.getLogger('log1').info(self.env._get_grid_representations())
                    print("The reward of the last training episode was " + str(ep_reward))
                    print("The Terminal reward was " + str(reward))

                    if self.path is not None:
                        print('Save output to: \n' + self.path + '\n')
                        self.env.save_stowage_plan(self.path)

                # If agent doesnt reach end break here - seems unnessary when there is no switch Lane Option
                #if steps > self.MAX_IT:
                #    break

            logging.getLogger('log1').info("It" + str(i) + " EPS: " + str(self.EPS) + " reward: " + str(ep_reward))
            # Epsilon decreases lineary during training TODO 50 is arbitrary

            if 1. - i / (self.number_of_episodes - 100) > 0:
                self.EPS -= 1. / (self.number_of_episodes - 100)
            else:
                self.EPS = 0

            # if self.EPS > self.EPSmin:
            #    self.EPS *= self.EPSdec
            # else:
            #    self.EPS = self.EPSmin

            self.eps_history.append(self.EPS)

            self.total_rewards[i] = ep_reward
            self.state_expansion[i] = len(self.q_table.keys())
            self.stepsToExit[i] = steps



            if i % 500 == 0 and i > 0:
                avg_reward = np.mean(self.total_rewards[max(0, i - 500):(i + 1)])
                # std_reward = np.std(self.totalRewards[max(0, i - 100):(i + 1)])
                print('episode ', i, 'score %.2f' % ep_reward, '\tavg. score %.2f' % avg_reward)

            # if i%500 == 0:
            #    print(len(self.q_table.keys()))

        logging.getLogger('log1').info("End training process")
        self.training_time = time.time() - start
        print('Finished training after {} min {} sec. \n'
              .format(int(self.training_time/60), round(self.training_time % 60, 0)))
        if self.path is not None:
            print('Save output to: \n' + self.path + '\n')
            self.env.save_stowage_plan(self.path)

        return self.q_table, self.total_rewards, self.stepsToExit, np.array(self.eps_history), self.state_expansion

    # TODO cleanup
    # def maxAction(self, state):
    #    #print(self.env.possibleActions)
    #    possibleActions = self.action_ix[self.env.possible_actions]
    # print(possibleActions)
    # print(Q[state.tobytes()])
    #    positionsOfBestPossibleAction = np.argmax(self.q_table[state.tobytes()][self.env.possible_actions])
    # print(positionsOfBestPossibleAction)
    #    return possibleActions[positionsOfBestPossibleAction]

    # TODO try feather // move AgentIMpl class add csv file format
    def load_model(self, path):
        try:
            self.q_table = pickle.load(open(path + '_qTablePickled.p', "rb"))
        except:
            logging.getLogger("log1").error("Could not load pickle file")

    def save_model(self, path, file_type='pickle'):
        self.q_table["ModelParam"] = {"Algorithm": "Time Difference Q-Learning",
                                      "GAMMA": self.GAMMA,
                                      "ALPHA": self.ALPHA,
                                      "Episodes": self.number_of_episodes,
                                      "EnvLanes:": self.env.lanes,
                                      "EnvRows": self.env.rows,
                                      "VehicleData": self.env.vehicle_data,
                                      "TrainingTime": self.training_time}
        info = "TDQ" + "_L" + str(self.env.lanes) + "_R" + str(self.env.rows) + "_Rf" + \
               str(int(1 in self.env.vehicle_data[5])) + "_A" + str(len(self.env.vehicle_data[0]))

        # path = path + '_qTablePickled.p'

        super().save_model(path + info)

#TODO delete
'''
    def execute(self, human_interaction=False):
        observation = self.env.reset()
        done = False

        while not self.done:
            action = self.max_action(self.q_table, observation, self.env.possible_actions)
            observation, reward, done, _ = self.env.step(self.action)
            self.epReward += reward

        self.env.render()
'''