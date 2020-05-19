from keras.layers import Dense, Activation, Conv1D, Conv2D
from keras.models import Sequential, load_model
from keras.regularizers import l2, l1
from keras.optimizers import Adam
from keras_radam import RAdam
import tensorflow as tf
# from tensorflow.random import set_seed
import numpy as np
import logging
# from agent.agent import Agent
# from agent.dqn import DQNAgent
import numpy as np
from env.roroDeck import RoRoDeck
from analysis.Plotter import Plotter
import logging
from analysis.loggingUnit import LoggingBase
from valuation.evaluator import Evaluator
import time
from agent.agentInterface import Agent






class DQNAgent(Agent):
    def __init__(self, env, alpha, gamma, module_path, epsilon=1, batch_size=32, number_of_episodes=12000, epsilon_dec=0.996,
                 epsilon_end=0.01, mem_size=1000_000, layers=[450,450,450], activation= 'relu', regularisation=0.001, optimiser='Adam',model_name='20200427stochasticdqn_model.h5'):

        if model_name is not None:
            self.model_name = model_name
        else:
            pass #TODO


        np.random.seed(0)
        tf.random.set_seed(0)

        self.module_path = module_path

        # For train method

        self.number_of_episodes = number_of_episodes
        self.env = env
        self.number_of_actions = len(self.env.vehicle_data[0])
        self.input_dims = np.shape(env.reset())[0]

        # TODO schöner
        logging.getLogger('log1').info("Init DQN-Agent: ALPHA: {}".format(alpha)
                                       + " GAMMA: {}".format(gamma)
                                       + " Replay Buffer Memory Size: {}".format(mem_size)
                                       + " Model name: {}".format(model_name)
                                       + " Epsilon Decrement: {}".format(epsilon_dec)
                                       + " Batch Size: {}".format(batch_size))
        self.action_space = [i for i in range(self.number_of_actions)]
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_dec = epsilon_dec
        self.epsilon_min = epsilon_end
        self.batch_size = batch_size

        self.layers = layers
        self.activation = activation
        self.regularisation = regularisation

        self.memory = ExperienceReplay(mem_size, self.input_dims, self.number_of_actions, discrete=True)

        logging.getLogger('log1').info("Start building Q Evaluation NN")
        self.q_eval = self.build_ANN(alpha, self.number_of_actions, self.input_dims, layers= [450,450,450,450])

        # Add target network for stability and update it delayed to q_eval
        logging.getLogger('log1').info("Start building Q Target NN")
        self.q_target_model = self.build_ANN(alpha, self.number_of_actions, self.input_dims, layers= [450,450,450,450])
        logging.getLogger('log1').info("Copy weights of Evaluation NN to Target Network")
        self.q_target_model.set_weights(self.q_eval.get_weights())

        self.target_update_counter = 0

        self.UPDATE_TARGET = 20


        #Plots
        self.steps_to_exit = np.zeros(self.number_of_episodes)

    def build_ANN(self, lr, output_dimension, input_dimsion, layers= [450,450,450,450], activation='relu', regularisation=0.001):
        #model = Sequential([Dense(layers[0], input_shape=(input_dimsion,)),
        #                    Activation(activation),
         #                   Dense(layers[1], activity_regularizer=l2(regularisation)),
          #                  Activation(activation),
           #                 Dense(layers[2], activity_regularizer=l2(regularisation)),
            #                Activation(activation),
             #               Dense(layers[3], activity_regularizer=l2(regularisation)),
              #              Activation(activation),
               #             Dense(output_dimension, activity_regularizer=l1(regularisation))])

        model = Sequential([Dense(layers[0], input_shape=(input_dimsion,)),
                            Activation('relu')])
        for layer in layers[1:]:
            model.add(Sequential([Dense(layer, activity_regularizer=l2(regularisation)), Activation(activation)]))
        model.add(Sequential([Dense(output_dimension, activity_regularizer=l1(regularisation))]))


        logging.getLogger(__name__).info("Compile NN")
        model.compile(optimizer=Adam(lr=lr), loss='mse')
        # model.compile(RAdam(), loss='mse')
        model.summary(print_fn=logging.getLogger(__name__).info)
        logging.getLogger(__name__).info("Finish build NN")
        return model

    def train(self):
        total_rewards = []
        eps_history = []
        start = time.time()
        self.steps_to_exit = np.zeros(self.number_of_episodes)

        for i in range(self.number_of_episodes):
            bad_moves_counter = 0
            done = False
            episode_reward = 0
            observation = self.env.reset()
            steps = 0
            while not done:
                steps += 1
                # possible_actions = env.possibleActions
                # if i<n_games*(3./4.): #TODO
                if i == 10000:
                    self.epsilon = 1
                    self.epsilon_dec = 0.99995
                if i > 10000:
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
                if bad_moves_counter > 6:
                    break

            eps_history.append(self.epsilon)
            total_rewards.append(episode_reward)
            self.steps_to_exit[i] = steps
            avg_reward = np.mean(total_rewards[max(0, i - 100):(i + 1)])

            # TODO plot to logger
            logging.getLogger('log1').info('episode {}'.format(i)
                                           + ' score {}'.format(round(episode_reward, 2))
                                           + 'avg. score {}'.format(round(avg_reward, 2)))
            if i % 10 == 0 and i > 0:
                print('episode ', i, 'score %.2f' % episode_reward, 'avg. score %.2f' % avg_reward)
                self.save_model(self.module_path)

        if i == self.number_of_episodes - 1 and done:
            logging.getLogger('log1').info(self.env._get_grid_representations())
            print("The reward of the last training episode was " + str(episode_reward))
            print("The Terminal reward was " + str(reward))
            print(self.module_path)
            if self.module_path != None:
                self.env.save_stowage_plan(self.module_path)
            self.training_time = time.time() - start

            _ = self.env.reset()
            self.execute(self.env)
            env.render()
            env.save_stowage_plan(self.module_path)
            self.save_model(self.module_path)
            evaluator = Evaluator(env.vehicle_data, env.grid)
            evaluation = evaluator.evaluate(env.get_stowage_plan())

            print(evaluation)

            logging.getLogger('log1').info("\nEnd training process after %d sec".format(self.training_time))
        return self.q_eval, total_rewards, self.steps_to_exit, eps_history, None

    def remember(self, state, action, reward, new_state, done, possible_Actions_state, possible_Actions_new_state):
        self.memory.store_transition(state, action, reward, new_state, done, possible_Actions_state,
                                     possible_Actions_new_state)

    def choose_action(self, state, possibleactions=None):
        state = state[np.newaxis, :]
        rand = np.random.random()
        if rand < self.epsilon:
            if possibleactions is None:
                action = np.random.choice(self.action_space)
            else:
                action = np.random.choice(possibleactions)
        else:
            actions = self.q_eval.predict(state)
            if possibleactions is None:
                action = np.argmax(actions)
            else:
                action_qVal_red = actions[0][possibleactions]
                action = possibleactions[np.argmax(action_qVal_red)]
        return action

    def learn(self):
        if self.memory.memory_size <= self.batch_size:
            return

        # logging.getLogger('log1').info("Learning Step - sample from replay buffer")

        state, action, reward, new_state, done, possible_actions_state, possible_actions_new_state = \
            self.memory.sample(self.batch_size)

        action_values = np.array(self.action_space, dtype=np.int8)
        action_indices = np.dot(action, action_values)

        if (state.ndim == 1):
            state = np.array([state])
        if (new_state.ndim == 1):
            new_state = np.array([new_state])

        q = self.q_eval.predict(state)
        q_next = self.q_eval.predict(new_state)

        # q_target = q_eval.copy()
        # q_target = q_eval[:]

        q_target = self.q_target_model.predict(state)

        # q_target[possible_actions_state] = 0.
        # q_next[possible_actions_new_state] = -5.

        # print(q_target[possible_actions_state])

        # self.q_target_model.set_weights(self.q_eval.get_weights())

        batch_index = np.arange(self.batch_size, dtype=np.int32)

        # print(q_target[batch_index,action_indices])

        q_target[batch_index, action_indices] = reward + \
                                                self.gamma * np.max(q_next, axis=1) * (1 - done)

        _ = self.q_eval.fit(state, q_target, verbose=0)

        # self.q_eval.train_on_batch(state, q_target)

        self.epsilon = self.epsilon * self.epsilon_dec if self.epsilon > self.epsilon_min else self.epsilon_min

        if self.target_update_counter > self.UPDATE_TARGET:
            self.q_target_model.set_weights(self.q_eval.get_weights())
            self.target_update_counter = 0
            # logging.getLogger('log1').info("Copy weights of Evaluation NN to Target Network")

        self.target_update_counter += 1

    def save_model(self, path):
        ##file path
        self.q_eval.save(path + self.model_name)

    def load_model(self, path):
        self.q_eval = load_model(path)

    #TODO return best reward
    def execute(self, env):
        # if env != None:
        #    self.env = env
        # observation = self.env.reset()
        done = False
        state = env.current_state
        done = False
        while not done:
            pos_action = np.argmax(self.q_eval.predict(state[np.newaxis, :])[0][env.possible_actions])
            action = env.possible_actions[pos_action]
            state, reward, done, info = env.step(action)

    #Overwritten max_action
    def max_action(self, state, possible_actions):
        # prediction = self.q_eval.predict(state[np.newaxis, :])
        pos_action = np.argmax(self.q_eval.predict(state[np.newaxis, :])[0][possible_actions])
        action = possible_actions[pos_action]

        return action


# Replay Buffer
class ExperienceReplay(object):
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
        dtype = np.int8 if self.discrete else np.float32
        self.action_memory = np.zeros((self.memory_size, n_actions), dtype=dtype)
        self.reward_memory = np.zeros(self.memory_size)
        self.terminal_memory = np.zeros(self.memory_size, dtype=np.float32)

        # TODO added this to eliminate illigal actions
        self.possibleActions_state = np.zeros((self.memory_size, n_actions), dtype='bool')
        self.possibleActions_new_state = np.zeros((self.memory_size, n_actions), dtype='bool')

    def store_transition(self, state, action, reward, state_, done, possible_Actions_state, possible_Actions_new_state):
        index = self.memory_counter % self.memory_size
        if self.memory_size > 0 and index == 0:
            logging.getLogger('log1').info("Memory of size %d full - start overwriting old experiences", self.memory_size)

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


if __name__ == '__main__':
    np.random.seed(0)
    tf.random.set_seed(0)

    loggingBase = LoggingBase()
    module_path = loggingBase.module_path
    env = RoRoDeck(False, lanes=10, rows=12, stochastic=False)
    #    env.vehicle_Data[4][env.mandatory_cargo_mask]+=4
    #    env.vehicle_Data[4][4] = 2 #reefer

    number_of_episodes = 12000

    agent = DQNAgent(env=env, module_path=module_path, gamma=0.999, number_of_episodes=number_of_episodes, epsilon=1.0, alpha=0.0005,
                     mem_size=1000_000,
                     batch_size=32, epsilon_end=0.01, epsilon_dec=0.9999925, layers=[550,450,450,550])

    model, total_rewards, steps_to_exit, eps_history, state_expansion = agent.train()
    plotter = Plotter(module_path, number_of_episodes)
    plotter.plotRewardPlot(total_rewards)
    plotter.plotEPSHistory(np.array(eps_history))

