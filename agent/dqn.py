from keras.layers import Dense, Activation
from keras.models import Sequential, load_model
from keras.regularizers import l2
from keras.optimizers import Adam
import tensorflow as tf

import os
import pickle
import numpy as np
import time
import logging
from env.roroDeck import RoRoDeck
from analysis.plotter import Plotter
from analysis.loggingUnit import LoggingBase
from analysis.evaluator import Evaluator
from agent.BaseAgent import Agent

# Supress tensorflow warnings
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

class DQLearningAgent(Agent):
    def __init__(self, env, module_path, alpha=0.0005, gamma=0.999, epsilon=1., epsilon_dec=0.9996,
                 epsilon_min=0.01, batch_size=32, number_of_episodes=12000, mem_size=1_000_000, layers=[128, 128], activation='relu',
                 regularisation=0.001, model_name=None, bad_moves_cut_off=8,
                 pretraining_duration=10_000, additional_info=None, cut_off=17, update_target=100):
        """
        Implementation of the DQ-Learning algorithm (Minh et.al. 2013)
        based on work done by Tabor (cf. 2020, www.github.com/philtabor/Deep-Q-Learning-Paper-To-Code)
        with modifications specific to the RoRo deck stowage planning problem:
                    - no convolutions
                    - a form of pre-training
                    - smaller network (2 layers)
                    - L2 regularisation
                    - illegal action cut-off
                    - performance cut-off


        Args:
            env(object):                RoRo-deck environment
            module_path(string):        output path
            alpha(float):               learning rate
            gamma(float):               discount factor
            epsilon(float):             initial epsilon for epsilon-greedy
            epsilon_dec(float):         decremental epsilon value (for exponential decay)
            epsilon_min(float):         minimal epsilon
            batch_size(int):            size of minibatch
            number_of_episodes(int):    number of training episodes
            mem_size(int):              size of replay buffer
            layers(int):                number of neurons for each layer of ANN
            activation(string):         activation function for each ANN unit
            regularisation(float):      regularisation factor
            model_name(string):         model name (if None than autogenerated)
            bad_moves_cut_off:          number of illegal actions per episode before cut-off
            pretraining_duration(int):  number of episodes in pretraining
            additional_info(string):    additional information to append to model_name
            cut_off(int):               performance cut-off to avoid instabilites
            update_target(int):         delay (in episodes) for weight update of target network
        """

        self.env = env
        self.module_path = module_path

        name = "DQLearning" + "_L" + str(self.env.lanes) + "-R" + str(self.env.rows)
        if additional_info is not None:
            name += "-L" + str(additional_info)

        self.bad_moves_cut_off = bad_moves_cut_off

        self.number_of_episodes = number_of_episodes

        if model_name is not None:
            self.model_name = model_name
        else:
            self.model_name = name

        self.number_of_actions = len(self.env.vehicle_data[0])
        self.input_dims = np.shape(env.reset())[0]
        self.layers = layers
        self.activation = activation
        self.regularisation = regularisation
        self.additional_info = additional_info

        self.info = "Init DQ-Agent:\tALPHA: {} \n".format(alpha) \
                    + "\t\t\tGAMMA: {}\n".format(gamma) \
                    + "\t\t\tReplay Buffer Memory Size: {}\n".format(mem_size) \
                    + "\t\t\t Model name: {}\n".format(model_name) \
                    + "\t\t\t Epsilon Decrement: {}\n".format(epsilon_dec) \
                    + "\t\t\t Batch Size: {}\n".format(batch_size) \
                    + "\t\t\t Iterations: {}\n".format(number_of_episodes) \
                    + "\t\t\t Pretraining End Episode: {}".format(pretraining_duration)
        logging.getLogger(__name__).info(self.info)

        self.gamma = gamma
        self.action_space = [i for i in range(self.number_of_actions)]
        self.epsilon = epsilon
        self.epsilon_dec = epsilon_dec
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size

        self.training_time = 0

        # If pretraining_duration equals None then don't do pretraining
        if pretraining_duration is None:
            self.pretraining_duration = number_of_episodes + 1
        else:
            self.pretraining_duration = pretraining_duration

        # Cut-off training when performance threshold is reached to avoid instability
        self.cut_off = cut_off

        # Initialise replay buffer
        self.memory = ReplayBuffer(mem_size, self.input_dims, self.number_of_actions)

        # Build DQN and target network
        logging.getLogger(__name__).info("Start building Q Evaluation NN")
        self.info += '\n Information on Q-Network'
        self.q_eval = self.build_ann(alpha, self.number_of_actions, self.input_dims, layers=self.layers)
        logging.getLogger(__name__).info("Start building Q Target NN")
        self.info += '\n Information on target Q-Network (Identical with Q-Network)'
        self.q_target_model = self.build_ann(alpha, self.number_of_actions, self.input_dims, layers=self.layers)
        logging.getLogger(__name__).info("Copy weights of Evaluation NN to Target Network")
        self.q_target_model.set_weights(self.q_eval.get_weights())

        # Set delay for target network
        self.update_target = update_target
        self.target_update_counter = 0

        logging.getLogger(__name__).info("Update the target network every {} learning steps".format(self.update_target))

        # Data collection for plots and metrics
        self.loaded_cargo = []
        self.total_rewards = []
        self.eps_history = []

    def build_ann(self, lr, output_dimension, input_dimension, layers=[128, 128], activation='relu',
                  regularisation=0.001):
        """
        Build an ANN with Tensorflow and Keras
        Args:
            lr(float):                  learning rate for Adam optimiser (upper bound)
            output_dimension(float):    size output layer (number of cargo types)
            input_dimension(float):     size of input layer (size of state representation)
            layers(list of int):        list of integers (how many neurons per layer)
            activation(string):         Activation function
            regularisation(float):      Weight regularisation factor for L2

        Returns:
            compiled Tensorflow/Keras based ANN
        """

        neural_net = Sequential([Dense(layers[0], input_shape=(input_dimension,)),
                                 Activation(activation)])
        for layer in layers[1:]:
            neural_net.add(Sequential([Dense(layer, activity_regularizer=l2(regularisation)), Activation(activation)]))
        neural_net.add(Sequential([Dense(output_dimension, activity_regularizer=l2(regularisation))]))

        logging.getLogger(__name__).info("Compile NN...")
        nn_info = "NN has \n \t\t\t\t\t" + \
                  "{} layers with {} activation \n\t\t\t\t\t".format(len(layers), activation) + \
                  "{} L2-activity regularisation in each layer\n\t\t\t\t\t".format(
                      regularisation) + \
                  "Adam-Optimiser with learning rate {} \n\t\t\t\t\t".format(lr) + \
                  "Mean Squared Error- loss function"

        logging.getLogger(__name__).info(nn_info)
        neural_net.compile(optimizer=Adam(lr=lr), loss='mse')
        neural_net.summary(print_fn=logging.getLogger(__name__).info)
        self.info += '\n' + '*' * 80 + '\n' + nn_info + '\n' + '*' * 80 + '\n'
        logging.getLogger(__name__).info("Compiled NN successfully!")
        return neural_net

    def train(self):
        """Initialise training process"""

        start = time.time()

        for i in range(self.number_of_episodes):
            bad_moves_counter = 0
            done = False
            episode_reward = 0
            observation = self.env.reset()
            steps = 0
            while not done:
                steps += 1

                # Agent can take legal actions only
                if i > self.pretraining_duration:
                    if np.random.random() < self.epsilon:
                        action = np.random.choice(self.env.possible_actions)
                    else:
                        actions = self.q_eval.predict(observation[np.newaxis, :])
                        action_qVal_reduce = actions[0][self.env.possible_actions]
                        action = self.env.possible_actions[np.argmax(action_qVal_reduce)]
                # Agent can take every action in pretraining
                else:
                    if np.random.random() < self.epsilon:
                        action = np.random.choice(self.action_space)
                    else:
                        action = np.argmax(self.q_eval.predict(observation[np.newaxis, :]))

                state_actions = self.env.possible_actions
                if action not in state_actions:
                    bad_moves_counter += 1
                    steps -= 1
                observation_, reward, done, info = self.env.step(action)
                episode_reward += reward
                self.memory.store_transition(observation, action, reward, observation_, done)
                observation = observation_
                self.learn()

                # Break if to many illegal actions to avoid investigating irrelevant areas of state space
                if bad_moves_counter >= self.bad_moves_cut_off:
                    break

            # Decrement epsilon
            self.epsilon = self.epsilon * self.epsilon_dec if self.epsilon > self.epsilon_min else self.epsilon_min

            # Save training process data
            self.eps_history.append(self.epsilon)
            self.total_rewards.append(episode_reward)
            self.loaded_cargo.append(steps)
            avg_reward = np.mean(self.total_rewards[max(0, i - 100):(i + 1)])

            if i > self.number_of_episodes - 50:
                self.env.stochastic = False
                self.epsilon = 0.


            logging.getLogger(__name__).info('episode {}'.format(i)
                                             + 'epsilon: {} '.format(self.epsilon)
                                             + ' score {} '.format(round(episode_reward, 2))
                                             + 'avg. score {} '.format(round(avg_reward, 2))
                                             + 'bad moves {}'.format(bad_moves_counter))
            if i % 10 == 0 and i > 0:
                print('episode ', i, 'score %.2f \t' % episode_reward, 'illegal moves {} \t'.format(bad_moves_counter),
                      'avg. score %.2f' % avg_reward)
                self.save_model(self.module_path)

            #Check if agent reached performance cut-off
            performance_cut_off = True if np.mean(
                self.total_rewards[max(0, i - 100):(i + 1)]) > self.cut_off else False

            if performance_cut_off:
                self.env.stochastic = False
                break

        # Print and save final training results
        if i == self.number_of_episodes - 1 or performance_cut_off:
            if performance_cut_off:
                logging.getLogger(__name__).info("ANN has reached performance cut-off...")

            logging.getLogger(__name__).info(self.env._get_grid_representations())

            if self.module_path is not None:
                self.env.save_stowage_plan(self.module_path)

            self.env.render()

            self.training_time = time.time() - start

            _ = self.env.reset()
            last_reward = self.execute(self.env)
            print("The reward of the last training episode was " + str(last_reward))

            self.env.save_stowage_plan(self.module_path)
            self.save_model(self.module_path)
            evaluator = Evaluator(self.env.vehicle_data, self.env.grid)
            evaluation = evaluator.evaluate(self.env.get_stowage_plan())
            print(evaluation)

            self.training_time = time.time() - start
            ttime_minutes = (int(self.training_time / 60), round(self.training_time % 60, 0))

            print('Lanes:', self.env.lanes, ' rows: ', self.env.rows)
            print('Finished training after {} min {} sec. \n'.format(ttime_minutes[0], ttime_minutes[1]))
            logging.getLogger(__name__).info('Training Time: {} m. {} s. \n\t\t\t--> ({} s.)'.format(ttime_minutes[0],
                                                                                                     ttime_minutes[1],
                                                                                                     self.training_time))
            print('Save output to: \n' + self.module_path + '\n')

        return self.q_eval, self.total_rewards, self.loaded_cargo, self.eps_history, None

    def learn(self):
        """ Update the DQN as outlined in thesis."""

        # Don't sample/learn if the replay buffer size is smaller than minibatch size
        if self.memory.memory_size <= self.batch_size:
            return

        state, action, reward, new_state, done = self.memory.sample(self.batch_size)

        action_values = np.array(self.action_space, dtype=np.int8)
        action_indices = np.dot(action, action_values)

        q_next = self.q_eval.predict(new_state)

        q_target = self.q_target_model.predict(state)

        batch_index = np.arange(self.batch_size, dtype=np.int32)

        # Calculate target (in thesis this corresponds to variable y)
        q_target[batch_index, action_indices] = reward + self.gamma * np.max(q_next, axis=1) * (1 - done)

        # Train on minibatch (minimise loss)
        _ = self.q_eval.train_on_batch(state, q_target)

        # update weights of target network with delay
        if self.target_update_counter > self.update_target:
            self.q_target_model.set_weights(self.q_eval.get_weights())
            self.target_update_counter = 0

        self.target_update_counter += 1

    def save_model(self, path):
        """Save model and trainings history"""

        self.q_eval.save(path + '_' + self.model_name + '.h5')
        path += self.model_name + '_training_history\\'
        os.makedirs(path, exist_ok=True)
        try:
            pickle.dump(self.total_rewards, open(path + 'rewards_history.p', "wb"))
            pickle.dump(self.eps_history, open(path + 'eps_history.p', "wb"))
            pickle.dump(self.loaded_cargo, open(path + 'cargo_loaded_history.p', "wb"))
        except:
            logging.getLogger(__name__).error("Could not save training history as pickle file to " + path)

    def load_model(self, path):
        """Load a saved model"""

        self.q_eval = load_model(path)

    def execute(self, env_=None):
        """finish an episode (or do one completely) by picking always the best action"""

        if env_ is not None:
            self.env = env_
            current_state = self.env.current_state
        else:
            current_state = self.env.reset()
        done = False
        cumulated_reward = 0
        while not done:
            pos_action = np.argmax(self.q_eval.predict(current_state[np.newaxis, :])[0][self.env.possible_actions])
            action = self.env.possible_actions[pos_action]
            current_state, reward, done, info = self.env.step(action)
            cumulated_reward += self.gamma * reward
        return cumulated_reward

    def max_action(self, state, possible_actions):
        """Find best actions of all legal actions"""

        pos_action = np.argmax(self.q_eval.predict(state[np.newaxis, :])[0][possible_actions])
        action = possible_actions[pos_action]
        return action

    def predict(self, state, action):
        """Predict Q values of a given action"""

        source = self.q_eval.predict(state[np.newaxis, :])
        return source[0][action]


class ReplayBuffer(object):
    def __init__(self, max_size, input_shape, n_actions):
        """
        Initialise replay buffer. The implementation of the replay buffer is
        based on work done by Tabor (cf. 2020, www.github.com/philtabor/Deep-Q-Learning-Paper-To-Code/)

        Args:
            max_size(int):          Size of replay buffer
            input_shape(int):       length of state representations
            n_actions(int):         number of actions, i.e. number of cargo types
        """

        np.random.seed(0)

        logging.getLogger(__name__).info("Init Replay Buffer:\n\tMax. Size: " + str(max_size) + "\tInput Shape: "
                                         + str(input_shape) + "\tNumber of actions: " + str(n_actions))

        self.memory_size = max_size
        self.memory_counter = 0
        self.state_memory = np.zeros((self.memory_size, input_shape))
        self.new_state_memory = np.zeros((self.memory_size, input_shape))
        self.reward_memory = np.zeros(self.memory_size)
        self.action_memory = np.zeros((self.memory_size, n_actions), dtype=np.int)
        self.terminal_memory = np.zeros(self.memory_size, dtype=np.float32)

    def store_transition(self, state, action, reward, new_state, done):
        """Store a transition in the replay buffer"""
        index = self.memory_counter % self.memory_size
        if self.memory_size > 0 and index == 0:
            logging.getLogger(__name__).info("Memory of size {} full - start" +
                                             " overwriting old experiences...".format(self.memory_size))

        self.state_memory[index] = state
        self.new_state_memory[index] = new_state
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done

        # Actions as one-hot-encoding
        actions = np.zeros(self.action_memory[1].size)
        actions[action] = 1
        self.action_memory[index] = actions

        self.memory_counter += 1

    def sample(self, batch_size):
        """Sample from replay buffer a minibatch"""

        # Ensure that only valid transitions are sampled
        max_memory = min(self.memory_size, self.memory_counter)
        batch = np.random.choice(max_memory, batch_size)

        states = self.state_memory[batch]
        states_ = self.new_state_memory[batch]
        rewards = self.reward_memory[batch]
        actions = self.action_memory[batch]
        terminal = self.terminal_memory[batch]

        return states, actions, rewards, states_, terminal

# Example of usage
if __name__ == '__main__':
    np.random.seed(0)

    loggingBase = LoggingBase()
    module_path = loggingBase.module_path
    env = RoRoDeck(lanes=10, rows=12)

    agent = DQLearningAgent(env=env, module_path=module_path, number_of_episodes=6000)

    model, total_rewards, steps_to_exit, eps_history, state_expansion = agent.train()
    plotter = Plotter(module_path, agent.number_of_episodes, show_plot=True)
    plotter.plotRewardPlot(total_rewards)
    plotter.plotEPSHistory(np.array(eps_history))
    plotter.plot_cargo_units_loaded(np.array(steps_to_exit))
