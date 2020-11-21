import numpy as np
import time
import logging
import pickle
from env.roroDeck import RoRoDeck
from analysis.plotter import Plotter
from analysis.loggingUnit import LoggingBase
from analysis.evaluator import Evaluator
from agent.baseAgent import Agent




class SARSA(Agent):
    def __init__(self, env=None, module_path=None, number_of_episodes=600000, gamma=0.999, alpha=0.1, epsilon=1.,
                 epsilon_min=0.001, additional_info=None):
        """
        Implementation of the SARSA algorithm (cf. Rummery and Niranjan, 1994)
        with modifications specific to the RoRo deck stowage planning problem:
                    - only choose and execute legal actions

        Args:
            env(object):                RoRo-deck environment
            module_path(Path-object):   output path
            number_of_episodes(int):    number of trainings episodes
            gamma(float):               discount factor
            alpha(float):               learning rate
            epsilon(float):             epsilon for epsilon-greedy
            epsilon_min(float):         minimal value of epsilon-greedy
            additional_info(string):    additional information to be appended to model name
        """

        logging.info("Initialise SARSA Agent")

        self.env = env
        self.path = module_path
        self.additional_info = additional_info
        self.number_of_episodes = number_of_episodes

        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.alpha = alpha
        self.gamma = gamma

        self.q_table = {}

        self.action_space_length = len(self.env.action_space)
        self.action_ix = np.arange(self.action_space_length)
        self.training_time = 0

        self.total_rewards = np.zeros(self.number_of_episodes)
        self.state_expansion = np.zeros(self.number_of_episodes)
        self.loaded_cargo = np.zeros(self.number_of_episodes)
        self.eps_history = np.zeros(self.number_of_episodes)

    def train(self):
        """ Trains the agent with the specified hyperparameter according to SARSA"""

        logging.getLogger(__name__).info("Start training process")
        start = time.time()

        # Initial state is initialised with zeros in Q-table
        observation = self.env.reset()
        self.q_table[observation.tobytes()] = np.zeros(self.action_space_length)

        logging.getLogger(__name__).info(self.get_info())
        print("Start Training Process")

        for i in range(self.number_of_episodes):
            observation = self.env.reset()
            done = False
            ep_reward = 0
            steps = 0

            # Choose current action, i.e. cargo unit by epsilon greedy strategy
            current_action = self.max_action(observation, self.env.possible_actions) \
                if np.random.random() < (1 - self.epsilon) \
                else self.env.action_space_sample()

            while not done:
                # Load one cargo unit
                observation_, reward, done, info = self.env.step(current_action)
                ep_reward += reward
                steps += 1

                if observation_.tobytes() not in self.q_table:
                    self.q_table[observation_.tobytes()] = np.zeros(self.action_space_length)


                # SARSA Update rule
                if not done:
                    action_ = self.max_action(observation_, self.env.possible_actions) if np.random.random() < (
                                1 - self.epsilon) \
                        else self.env.action_space_sample()

                    self.q_table[observation.tobytes()][current_action] += self.alpha * (
                            reward + self.gamma * self.q_table[observation_.tobytes()][action_]
                            - self.q_table[observation.tobytes()][current_action])

                    current_action = action_

                # Value of Terminal State is zero
                else:
                    self.q_table[observation.tobytes()][current_action] += self.alpha * (
                            reward - self.q_table[observation.tobytes()][current_action])

                observation = observation_

                if i == self.number_of_episodes - 1 and done:
                    logging.getLogger(__name__).info(self.env._get_grid_representations())
                    print("The reward of the last training episode was {} (was deterministic)".format(str(ep_reward)))
                    if self.path is not None:
                        self.env.save_stowage_plan(str(self.path))

            logging.getLogger(__name__).info('It. {:7d} \t'.format(i)
                                           + 'epsilon: {} \t'.format(round(self.epsilon, 4))
                                           + 'Reward: {}'.format(round(ep_reward, 2)))

            # Epsilon decreases linear during training
            if 1. - i / self.number_of_episodes > self.epsilon_min:
                self.epsilon -= 1. / self.number_of_episodes
            else:
                self.epsilon = self.epsilon_min

            if i == self.number_of_episodes - 2:
                logging.getLogger(__name__).info(
                    'Set env to deterministic for last it. {}'.format(i + 1)) if self.env.stochastic else None
                self.env.stochastic = False
                self.epsilon = 0.0

            self.eps_history[i] = self.epsilon
            self.total_rewards[i] = ep_reward
            self.state_expansion[i] = len(self.q_table.keys())
            self.loaded_cargo[i] = steps

            if i % 500 == 0 and i > 0:
                avg_reward = np.mean(self.total_rewards[max(0, i - 100):(i + 1)])
                print('episode ', i, '\tscore %.2f' % ep_reward, '\tavg. score %.2f' % avg_reward)

        # Print and save training data
        self.training_time = time.time() - start
        ttime_minutes = (int(self.training_time / 60), round(self.training_time % 60, 0))

        print('Finished training after {} min {} sec. \n'.format(ttime_minutes[0], ttime_minutes[1]))
        logging.getLogger(__name__).info('Training Time: {} m. {} s. \n\t\t\t--> ({} s.)'.format(ttime_minutes[0],
                                                                                                 ttime_minutes[1],
                                                                                                 self.training_time))
        if self.path is not None:
            print('Save output to: \n' + str(self.path) + '\n')
            self.env.save_stowage_plan(self.path)
            self.save_model(self.path)

        return self.q_table, self.total_rewards, self.loaded_cargo, np.array(self.eps_history), self.state_expansion

    def load_model(self, path):
        """ Load a Q-table"""

        try:
            self.q_table = pickle.load(open(str(path), "rb"))
        except:
            logging.getLogger(__name__).error("Could not load pickle file!")

    def save_model(self, path):
        """ Save Q-table and input data"""

        self.q_table["ModelParam"] = {"Algorithm": "SARSA",
                                      "GAMMA": self.gamma,
                                      "ALPHA": self.alpha,
                                      "Episodes": self.number_of_episodes,
                                      "EnvLanes:": self.env.lanes,
                                      "EnvRows": self.env.rows,
                                      "VehicleData": self.env.vehicle_data,
                                      "TrainingTime": self.training_time}
        info = "SARSA" + "_L" + str(self.env.lanes) + "-R" + str(self.env.rows)

        super().save_model(path, info)

# Example of usage
if __name__ == '__main__':
    np.random.seed(0)

    loggingBase = LoggingBase()
    module_path = loggingBase.module_path
    env = RoRoDeck(lanes=10, rows=12)
    number_of_episodes = 5000

    agent = SARSA(env=env, module_path=module_path, number_of_episodes=number_of_episodes)

    model, total_rewards, steps_to_exit, eps_history, state_expansion = agent.train()
    plotter = Plotter(module_path, agent.number_of_episodes, show_plot=False)
    plotter.plotRewardPlot(total_rewards)
    plotter.plotStateExp(state_expansion)
    plotter.plotEPSHistory(np.array(eps_history))
    plotter.plot_cargo_units_loaded(np.array(steps_to_exit))

    evaluator = Evaluator(env.vehicle_data, env.grid)
    evaluation = evaluator.evaluate(env.get_stowage_plan())
    print(evaluation)