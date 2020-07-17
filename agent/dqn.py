from keras.layers import Dense, Activation
from keras.models import Sequential, load_model
from keras.regularizers import l2, l1
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
from valuation.evaluator import Evaluator
import time
from agent.BasicAgent import Agent


class DQNAgent(Agent):
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
                 epsilon_min=0.02, mem_size=1_000_000, layers=[450, 450, 450], activation='relu',
                 regularisation=0.001, optimiser='Adam', model_name=None,
                 pretraining_duration=10_000):

        np.random.seed(0)
        #        tf.random.set_seed(0)

        self.module_path = module_path

        # For train method

        self.number_of_episodes = number_of_episodes
        self.env = env
        name = "DQN" + "_L" + str(self.env.lanes) + "_R" + str(self.env.rows) + "_Rf" + \
               str(int(1 in self.env.vehicle_data[5])) + "_A" + str(len(self.env.vehicle_data[0]))

        if model_name is not None:
            self.model_name = model_name
        else:
            self.model_name = name

        self.number_of_actions = len(self.env.vehicle_data[0])
        self.input_dims = np.shape(env.reset())[0]

        # TODO schöner
        logging.getLogger('log1').info("Init DQN-Agent:\tALPHA: {} \n".format(alpha)
                                       + "\t\t\tGAMMA: {}\n".format(gamma)
                                       + "\t\t\tReplay Buffer Memory Size: {}\n".format(mem_size)
                                       + "\t\t\t Model name: {}\n".format(model_name)
                                       + "\t\t\t Epsilon Decrement: {}\n".format(epsilon_dec)
                                       + "\t\t\t Batch Size: {}\n".format(batch_size)
                                       + "\t\t\t Iterations: {}\n".format(number_of_episodes)
                                       +"\t\t\t Pretraining End Episode: {}".format(pretraining_duration))
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

        # TODO make a numpy array or make a list in agent interface
        self.total_rewards = []
        self.eps_history = []

        self.memory = ExperienceReplay(mem_size, self.input_dims, self.number_of_actions, discrete=True)

        logging.getLogger('log1').info("Start building Q Evaluation NN")
        self.q_eval = self.build_ann(alpha, self.number_of_actions, self.input_dims, layers=self.LAYERS)

        # Add target network for stability and update it delayed to q_eval
        logging.getLogger('log1').info("Start building Q Target NN")
        self.q_target_model = self.build_ann(alpha, self.number_of_actions, self.input_dims, layers=self.LAYERS)
        logging.getLogger('log1').info("Copy weights of Evaluation NN to Target Network")
        self.q_target_model.set_weights(self.q_eval.get_weights())

        self.target_update_counter = 0

        self.UPDATE_TARGET = 100 # was 20
        logging.getLogger('log1').info("Update the target network every {} learning steps".format(self.UPDATE_TARGET))

        # Plots
        self.steps_to_exit = np.zeros(self.number_of_episodes)

    # TODO citation https://machinelearningmastery.com/how-to-reduce-overfitting-in-deep-learning-with-weight-regularization/
    def build_ann(self, lr, output_dimension, input_dimension, layers=[450, 450, 450, 450], activation='relu',
                  regularisation=0.001):

        neural_net = Sequential([Dense(layers[0], input_shape=(input_dimension,)),
                                 Activation('relu')])
        for layer in layers[1:]:
            neural_net.add(Sequential([Dense(layer, activity_regularizer=l2(regularisation)), Activation(activation)]))
        neural_net.add(Sequential([Dense(output_dimension, activity_regularizer=l2(regularisation))]))

        logging.getLogger(__name__).info("Compile NN...")
        logging.getLogger(__name__).info("NN has \n \t\t\t\t\t" + \
                                         "{} layers with {} activation \n\t\t\t\t\t".format(len(layers), activation) + \
                                         "{} l2-activity regularisation in each layer\n\t\t\t\t\t".format(
                                             regularisation) + \
                                         "Adam-Optimiser with learning rate {} \n\t\t\t\t\t".format(lr) + \
                                         "MSE-Lossfunction")
        neural_net.compile(optimizer=Adam(lr=lr), loss='mse')
        # model.compile(RAdam(), loss='mse')
        neural_net.summary(print_fn=logging.getLogger(__name__).info)
        logging.getLogger(__name__).info("Compiled NN successfully!")
        return neural_net

    def train(self):
        start = time.time()
        self.steps_to_exit = np.zeros(self.number_of_episodes)

        for i in range(self.number_of_episodes):
            bad_moves_counter = 0
            done = False
            episode_reward = 0
            observation = self.env.reset()
            steps = 0
            avg_reward = -np.inf
            while not done:
                steps += 1
                # possible_actions = env.possibleActions
                # if i<n_games*(3./4.): #TODO

                # Increase Batchsize to stabilise Training TODO:Paper: DON’TDECAY THE LEARNINGRATE,INCREASE THE BATCHSIZE
                #if i == self.PRETRAINING_DURATION or i == self.PRETRAINING_DURATION + 500:
                #    self.batch_size += 16

                if i > self.PRETRAINING_DURATION:
                    action = self.choose_action(observation, self.env.possible_actions)  ## add possible actions here
                else:
                    action = self.choose_action(observation)  ## add possible actions here

                state_actions = self.env.possible_actions
                if action not in state_actions:
                    bad_moves_counter += 1
                observation_, reward, done, info = self.env.step(action)
                new_state_actions = self.env.possible_actions
                episode_reward += reward
                self.remember(observation, action, reward, observation_, done, state_actions, new_state_actions)
                observation = observation_
                self.learn()

                # Break if to many illegal actions to avoid investigating unintresting areas
                if bad_moves_counter > 5:
                    break

            self.eps_history.append(self.EPSILON)
            self.total_rewards.append(episode_reward)
            self.steps_to_exit[i] = steps
            avg_reward = np.mean(self.total_rewards[max(0, i - 100):(i + 1)])

            # TODO plot to logger
            logging.getLogger('log1').info('episode {}'.format(i)
                                           + ' score {}'.format(round(episode_reward, 2))
                                           + 'avg. score {} '.format(round(avg_reward, 2))
                                           + 'bad moves {}'.format(bad_moves_counter))
            if i % 10 == 0 and i > 0:
                print('episode ', i, 'score %.2f' % episode_reward, 'avg. score %.2f' % avg_reward, 'bad moves {}'.format(bad_moves_counter))
                self.save_model(self.module_path)

            converged = True if np.mean(
                self.total_rewards[max(0, i - 100):(i + 1)]) > 13 else False  # TODO erbe init von general Agent class
            # np.median(total_rewards[max(0, i - 100):(i + 1)]) > 17 and \

            if converged:
                break

        if i == self.number_of_episodes - 1 or converged:
            if converged:
                logging.getLogger('log1').info("ANN has converged...")

            logging.getLogger('log1').info(self.env._get_grid_representations())
            print("The reward of the last training episode was " + str(episode_reward))
            print("The Terminal reward was " + str(reward))
            print(self.module_path)
            if self.module_path is not None:
                self.env.save_stowage_plan(self.module_path)
            self.training_time = time.time() - start

            _ = self.env.reset()
            self.execute(self.env)
            self.env.render()
            self.env.save_stowage_plan(self.module_path)
            self.save_model(self.module_path)
            evaluator = Evaluator(self.env.vehicle_data, self.env.grid)
            evaluation = evaluator.evaluate(self.env.get_stowage_plan())

            print(evaluation)

            logging.getLogger('log1').info("\nEnd training process after {} sec".format(self.training_time))
            print('Finished training after {} min {} sec. \n'
                  .format(int(self.training_time / 60), round(self.training_time % 60, 0)))
            print('Save output to: \n' + self.module_path + '\n')

        return self.q_eval, self.total_rewards, self.steps_to_exit, self.eps_history, None

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

        q = self.q_eval.predict(state)
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

        _ = self.q_eval.fit(state, q_target, verbose=0)

        # self.q_eval.train_on_batch(state, q_target)

        self.EPSILON = self.EPSILON * self.EPSILON_DEC if self.EPSILON > self.EPSILON_MIN else self.EPSILON_MIN

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
            pickle.dump(self.total_rewards, open(path + 'rewards.p', "wb"))
            pickle.dump(self.eps_history, open(path + 'eps_history.p', "wb"))
        finally:
            logging.getLogger("log1").error("Could not save training history as pickle file to+ " + path)

    def load_model(self, path):
        self.q_eval = load_model(path)

    # TODO return best reward
    def execute(self, env):
        state = env.current_state
        done = False
        while not done:
            pos_action = np.argmax(self.q_eval.predict(state[np.newaxis, :])[0][env.possible_actions])
            action = env.possible_actions[pos_action]
            state, reward, done, info = env.step(action)

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
    env = RoRoDeck(False, lanes=10, rows=12, stochastic=False)
    #    env.vehicle_Data[4][env.mandatory_cargo_mask]+=4
    #    env.vehicle_Data[4][4] = 2 #reefer

    number_of_episodes = 12000

    agent = DQNAgent(env=env, module_path=module_path, gamma=0.999, number_of_episodes=number_of_episodes, epsilon=1.0,
                     alpha=0.0005,
                     mem_size=1000_000,
                     batch_size=32, epsilon_min=0.01, epsilon_dec=0.9999925, layers=[550, 450, 450, 550])

    model, total_rewards, steps_to_exit, eps_history, state_expansion = agent.train()
    plotter = Plotter(module_path, number_of_episodes)
    plotter.plotRewardPlot(total_rewards)
    plotter.plotEPSHistory(np.array(eps_history))
