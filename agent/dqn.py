from keras.layers import Dense, Activation
from keras.models import Sequential, load_model
from keras.regularizers import l2
from keras.optimizers import Adam
#from keras_radam import RAdam
import tensorflow as tf

# Supress warinings
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
# from tensorflow.random import set_seed
import os
import pickle
import numpy as np
from env.roroDeck import RoRoDeck
from analysis.plotter import Plotter
import logging
from analysis.loggingUnit import LoggingBase
from analysis.evaluator import Evaluator
import time
from agent.BasicAgent import Agent


class DQLearningAgent(Agent):
    """
    Implementation of the DQN algorithm (Minh et.al. 2013) with alternations fitting
    for the RORO deck stowage planning problem:
                    - no convolutions
                    - a form of pre-training
                    - smaller network (2 layers)
                    - L2 regularisation
     """
    def __init__(self, env, module_path, alpha=0.0005, gamma=0.999, epsilon=1, batch_size=32, number_of_episodes=12000,
                 epsilon_dec=0.996,
                 epsilon_min=0.01, mem_size=1_000_000, layers=[128, 128], activation='relu',
                 regularisation=0.001, optimiser='Adam', model_name=None, bad_moves_cut_off = 7,
                 pretraining_duration=10_000, additional_info=None, cut_off = 17):

        self.bad_moves_cut_off = bad_moves_cut_off
        np.random.seed(0)
        #        tf.random.set_seed(0)

        self.module_path = module_path

        # For train method

        self.number_of_episodes = number_of_episodes
        self.env = env
        name = "DQLearning" + "_L" + str(self.env.lanes) + "-R" + str(self.env.rows)
        if additional_info is not None:
            name += "-L"+ str(additional_info)

        if model_name is not None:
            self.model_name = model_name
        else:
            self.model_name = name
        self.cut_off = cut_off
        self.number_of_actions = len(self.env.vehicle_data[0])
        self.input_dims = np.shape(env.reset())[0]
        self.info = "Init DQ-Agent:\tALPHA: {} \n".format(alpha) \
                                       + "\t\t\tGAMMA: {}\n".format(gamma) \
                                       + "\t\t\tReplay Buffer Memory Size: {}\n".format(mem_size) \
                                       + "\t\t\t Model name: {}\n".format(model_name) \
                                       + "\t\t\t Epsilon Decrement: {}\n".format(epsilon_dec) \
                                       + "\t\t\t Batch Size: {}\n".format(batch_size) \
                                       + "\t\t\t Iterations: {}\n".format(number_of_episodes) \
                                       + "\t\t\t Pretraining End Episode: {}".format(pretraining_duration) \
        # TODO schöner
        logging.getLogger(__name__).info(self.info)
        self.action_space = [i for i in range(self.number_of_actions)]
        self.GAMMA = gamma
        self.EPSILON = epsilon
        self.EPSILON_DEC = epsilon_dec
        self.EPSILON_MIN = epsilon_min
        self.batch_size = batch_size
        # This ensures that no pretraining is conducted
        if pretraining_duration is None:
            self.PRETRAINING_DURATION = number_of_episodes + 1
        else:
            self.PRETRAINING_DURATION = pretraining_duration
        self.LAYERS = layers
        self.ACTIVATION = activation
        self.REGULARISATION = regularisation
        self.additional_info = additional_info

        # TODO make a numpy array or make a list in agent interface
        self.total_rewards = []
        self.eps_history = []

        self.memory = ExperienceReplay(mem_size, self.input_dims, self.number_of_actions, discrete=True)

        logging.getLogger(__name__).info("Start building Q Evaluation NN")
        self.info += '\n Information on Q-Network'
        self.q_eval = self.build_ann(alpha, self.number_of_actions, self.input_dims, layers=self.LAYERS)

        # Add target network for stability and update it delayed to q_eval
        logging.getLogger(__name__).info("Start building Q Target NN")
        self.info += '\n Information on target Q-Network (Identical with Q-Network)'
        self.q_target_model = self.build_ann(alpha, self.number_of_actions, self.input_dims, layers=self.LAYERS)
        logging.getLogger(__name__).info("Copy weights of Evaluation NN to Target Network")
        self.q_target_model.set_weights(self.q_eval.get_weights())

        self.target_update_counter = 0

        self.UPDATE_TARGET = 100 # was 20
        logging.getLogger('log1').info("Update the target network every {} learning steps".format(self.UPDATE_TARGET))

        # Plots
        self.loaded_cargo = []

    # TODO citation https://machinelearningmastery.com/how-to-reduce-overfitting-in-deep-learning-with-weight-regularization/
    def build_ann(self, lr, output_dimension, input_dimension, layers=[128, 128], activation='relu',
                  regularisation=0.001):

        neural_net = Sequential([Dense(layers[0], input_shape=(input_dimension,)),
                                 Activation('relu')])
        for layer in layers[1:]:
            neural_net.add(Sequential([Dense(layer, activity_regularizer=l2(regularisation)), Activation(activation)]))
        neural_net.add(Sequential([Dense(output_dimension, activity_regularizer=l2(regularisation))]))

        logging.getLogger(__name__).info("Compile NN...")
        nn_info = "NN has \n \t\t\t\t\t" + \
                     "{} layers with {} activation \n\t\t\t\t\t".format(len(layers), activation) + \
                     "{} l2-activity regularisation in each layer\n\t\t\t\t\t".format(
                         regularisation) + \
                     "Adam-Optimiser with learning rate {} \n\t\t\t\t\t".format(lr) + \
                     "MSE-Lossfunction"

        logging.getLogger(__name__).info(nn_info)
        neural_net.compile(optimizer=Adam(lr=lr), loss='mse')
        # model.compile(RAdam(), loss='mse')
        neural_net.summary(print_fn=logging.getLogger(__name__).info)
        self.info += '\n'+'*'*80+'\n'+ nn_info+'\n'+'*'*80+'\n'
        logging.getLogger(__name__).info("Compiled NN successfully!")
        return neural_net

    def train(self):
        start = time.time()


        for i in range(self.number_of_episodes):
            bad_moves_counter = 0
            done = False
            episode_reward = 0
            observation = self.env.reset()
            steps = 0
            avg_reward = -np.inf
            while not done:
                steps += 1

                if i > self.PRETRAINING_DURATION:
                    action = self.choose_action(observation, self.env.possible_actions)  ## add possible actions here
                else:

                    action = self.choose_action(observation)

                state_actions = self.env.possible_actions
                if action not in state_actions:
                    bad_moves_counter += 1
                    steps -= 1
                observation_, reward, done, info = self.env.step(action)
                new_state_actions = self.env.possible_actions
                episode_reward += reward
                self.remember(observation, action, reward, observation_, done, state_actions, new_state_actions)
                observation = observation_
                self.learn()

                # Break if to many illegal actions to avoid investigating unintresting areas
                if bad_moves_counter > self.bad_moves_cut_off:
                    break
            self.EPSILON = self.EPSILON * self.EPSILON_DEC if self.EPSILON > self.EPSILON_MIN else self.EPSILON_MIN

            # Save training process data
            self.eps_history.append(self.EPSILON)
            self.total_rewards.append(episode_reward)
            self.loaded_cargo.append(steps)
            avg_reward = np.mean(self.total_rewards[max(0, i - 100):(i + 1)])

            if i > self.number_of_episodes-50:
                self.env.stochastic = False
                self.EPSILON = 0.


            # TODO plot to logger
            logging.getLogger(__name__).info('episode {}'.format(i)
                                           + 'EPS: {} '.format(self.EPSILON)
                                           + ' score {} '.format(round(episode_reward, 2))
                                           + 'avg. score {} '.format(round(avg_reward, 2))
                                           + 'bad moves {}'.format(bad_moves_counter))
            if i % 10 == 0 and i > 0:
                print('episode ', i, 'score %.2f \t' % episode_reward, 'illegal moves {} \t'.format(bad_moves_counter), 'avg. score %.2f' % avg_reward)
                self.save_model(self.module_path)

            converged = True if np.mean(
                self.total_rewards[max(0, i - 100):(i + 1)]) > self.cut_off else False

            if converged:
                self.env.stoachstic = False
                break

        if i == self.number_of_episodes - 1 or converged:
            if converged:
                logging.getLogger(__name__).info("ANN has converged...")


            logging.getLogger(__name__).info(self.env._get_grid_representations())
            print(self.module_path)
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

            print('Lanes:',self.env.lanes,' rows: ',self.env.rows)
            print('Finished training after {} min {} sec. \n'.format(ttime_minutes[0], ttime_minutes[1]))
            logging.getLogger(__name__).info('Training Time: {} m. {} s. \n\t\t\t--> ({} s.)'.format(ttime_minutes[0],
                                                                                                     ttime_minutes[1],
                                                                                                     self.training_time))
            print('Save output to: \n' + self.module_path + '\n')

        return self.q_eval, self.total_rewards, self.loaded_cargo, self.eps_history, None

    # TODO lösche diese Methode
    def remember(self, state, action, reward, new_state, done, possible_Actions_state, possible_Actions_new_state):
        self.memory.store_transition(state, action, reward, new_state, done, possible_Actions_state,
                                     possible_Actions_new_state)

    def choose_action(self, state, possible_actions=None):
        state = state[np.newaxis, :]
        rand = np.random.random()
        if rand < self.EPSILON:
            if possible_actions is None:
                action = np.random.choice(self.action_space)
            else:
                action = np.random.choice(possible_actions)
        else:
            actions = self.q_eval.predict(state)
            if possible_actions is None:
                action = np.argmax(actions)
            else:
                action_qVal_red = actions[0][possible_actions]
                action = possible_actions[np.argmax(action_qVal_red)]
        return action

    def learn(self):
        # Don't sample/learn if the replay buffer size is smaller than minibatch size
        if self.memory.memory_size <= self.batch_size:
            return

        # logging.getLogger('log1').info("Learning Step - sample from replay buffer")

        state, action, reward, new_state, done, possible_actions_state, possible_actions_new_state = \
            self.memory.sample(self.batch_size)

        action_values = np.array(self.action_space, dtype=np.int8)
        action_indices = np.dot(action, action_values)

        'TODO delete'
        #if state.ndim == 1:
        #    state = np.array([state])
        #if new_state.ndim == 1:
        #    new_state = np.array([new_state])

        #q = self.q_eval.predict(state)
        q_next = self.q_eval.predict(new_state)

        # q_target = q_eval.copy()
        # q_target = q_eval[:]

        q_target = self.q_target_model.predict(state)
        # q_target = self.q_target_model.predict(new_state)

        # q_target[possible_actions_state] = 0.
        # q_next[possible_actions_new_state] = -5.

        # print(q_target[possible_actions_state])

        # self.q_target_model.set_weights(self.q_eval.get_weights())

        batch_index = np.arange(self.batch_size, dtype=np.int32)

        # print(q_target[batch_index,action_indices])
        # DQN
        q_target[batch_index, action_indices] = reward + \
                                                self.GAMMA * np.max(q_next, axis=1) * (1 - done)




        # Double DQN  //TODO
        # q_target[batch_index, action_indices] = reward + \
        #                                        self.GAMMA * np.max(q_next, axis=1) * (1 - done)

        _ = self.q_eval.train_on_batch(state, q_target) #TODO how many epochs default was; 1 change from fit()

        # self.q_eval.train_on_batch(state, q_target)


        if self.target_update_counter > self.UPDATE_TARGET:
            self.q_target_model.set_weights(self.q_eval.get_weights())
            self.target_update_counter = 0
            # logging.getLogger('log1').info("Copy weights of Evaluation NN to Target Network")

        self.target_update_counter += 1

    def save_model(self, path):
        ##file path
        self.q_eval.save(path + '_' + self.model_name + '.h5')
        path += self.model_name + '_training_history\\'
        os.makedirs(path, exist_ok=True)
        try:
            # os.makedirs(path, exist_ok=True)
            pickle.dump(self.total_rewards, open(path + 'rewards_history.p', "wb"))
            pickle.dump(self.eps_history, open(path + 'eps_history.p', "wb"))
            pickle.dump(self.loaded_cargo, open(path + 'cargo_loaded_history.p', "wb"))
        except:
            logging.getLogger("log1").error("Could not save training history as pickle file to " + path)

    def load_model(self, path):
        self.q_eval = load_model(path)

    # TODO return best reward
    def execute(self, env=None):
        if env is not None:
            self.env = env
            current_state = self.env.current_state
        else:
            current_state = self.env.reset()
        #current_state = env.current_state
        done = False
        total_rewards = 0
        while not done:
            pos_action = np.argmax(self.q_eval.predict(current_state[np.newaxis, :])[0][self.env.possible_actions])
            action = self.env.possible_actions[pos_action]
            current_state, reward, done, info = self.env.step(action)
            total_rewards += self.GAMMA*reward
        return total_rewards

    # Overwritten max_action
    def max_action(self, state, possible_actions):
        # prediction = self.q_eval.predict(state[np.newaxis, :])
        pos_action = np.argmax(self.q_eval.predict(state[np.newaxis, :])[0][possible_actions])
        action = possible_actions[pos_action]

        return action

    def predict(self, state, action):
        source = self.q_eval.predict(state[np.newaxis, :])
        return source[0][action]


# Replay Buffer
class ExperienceReplay(object):
    """
        The implementation of the Replay Buffer is based on Tabor (2019) https://github.com/philtabor

    """
    def __init__(self, max_size, input_shape, n_actions, discrete=False):
        np.random.seed(0)

        logging.getLogger('log1').info("Init Replay Buffer: Max. Size: " + str(max_size) + " Input Shape: "
                                       + str(input_shape) + " Number of actions: "
                                       + str(n_actions) + "Discrete Action Space: " + str(discrete))

        self.memory_size = max_size
        self.memory_counter = 0
        self.discrete = discrete
        self.state_memory = np.zeros((self.memory_size, input_shape))
        self.new_state_memory = np.zeros((self.memory_size, input_shape))
        #dtype = np.int8 if self.discrete else np.float32
        self.action_memory = np.zeros((self.memory_size, n_actions), dtype=np.int)
        self.reward_memory = np.zeros(self.memory_size)
        self.terminal_memory = np.zeros(self.memory_size, dtype=np.float32)

        # TODO added this to eliminate illigal actions
        self.possibleActions_state = np.zeros((self.memory_size, n_actions), dtype='bool')
        self.possibleActions_new_state = np.zeros((self.memory_size, n_actions), dtype='bool')

    def store_transition(self, state, action, reward, state_, done, possible_Actions_state, possible_Actions_new_state):
        index = self.memory_counter % self.memory_size
        if self.memory_size > 0 and index == 0:
            logging.getLogger('log1').info("Memory of size %d full - start overwriting old experiences",
                                           self.memory_size)

        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done

        # TODO possible actions -> delete discrete than??

        if self.discrete:
            actions = np.zeros(self.action_memory[1].size)
            actions[action] = 1.0
            self.action_memory[index] = actions

            # ToDO added this to avoid estimating  illgeal actions
            self.possibleActions_state[index][possible_Actions_state] = True
            if not done:
                self.possibleActions_new_state[index][possible_Actions_new_state] = True
        else:
            self.action_memory[index] = action
        self.memory_counter += 1

    def sample(self, batch_size):
        max_memory = min(self.memory_size, self.memory_counter)
        batch = np.random.choice(max_memory, batch_size)

        states = self.state_memory[batch]
        states_ = self.new_state_memory[batch]
        rewards = self.reward_memory[batch]
        actions = self.action_memory[batch]
        terminal = self.terminal_memory[batch]

        possible_actions_state = self.possibleActions_state[batch]
        possible_actions_new_state = self.possibleActions_new_state[batch]

        return states, actions, rewards, states_, terminal, possible_actions_state, possible_actions_new_state


#    def clear(self):
#        logging.getLogger('log1').info("Clear Replay Buffer: Max. Size: " + str(max_size) + " Input Shape: "
#                                       + str(input_shape) + " Number of actions: "
#                                       + str(n_actions) + "Discrete Action Space: " + str(discrete))

if __name__ == '__main__':
    np.random.seed(0)
    tf.random.set_seed(0)

    loggingBase = LoggingBase()
    module_path = loggingBase.module_path
    env = RoRoDeck(lanes=10, rows=12, stochastic=False)
    #    env.vehicle_Data[4][env.mandatory_cargo_mask]+=4
    #    env.vehicle_Data[4][4] = 2 #reefer

    number_of_episodes = 12000

    agent = DQLearningAgent(env=env, module_path=module_path, gamma=0.999, number_of_episodes=number_of_episodes, epsilon=1.0,
                            alpha=0.0005,
                            mem_size=1000_000,
                            batch_size=32, epsilon_min=0.01, epsilon_dec=0.99999, layers=[128,128]) #layers=[550, 450, 450, 550]

    model, total_rewards, steps_to_exit, eps_history, state_expansion = agent.train()
    plotter = Plotter(module_path, number_of_episodes)
    plotter.plotRewardPlot(total_rewards)
    plotter.plotEPSHistory(np.array(eps_history))
    plotter.plot_cargo_units_loaded(np.array(steps_to_exit))
