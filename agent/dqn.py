from keras.layers import Dense, Activation, Conv1D
from keras.models import Sequential, load_model
from keras.regularizers import l2,l1
from keras.optimizers import Adam
import tensorflow as tf
#from tensorflow.random import set_seed
import numpy as np
import logging
#from agent.agent import Agent



#Replay Buffer
class ReplayBuffer(object):
    def __init__(self,max_size, input_shape, n_actions, discrete=False):
        np.random.seed(0)

        logging.getLogger('log1').info("Init Replay Buffer: Max. Size: " + str(max_size)+ " Input Shape: "+str(input_shape) + " Number of actions: "+ str(n_actions)+"Discrete Action Space: "+str(discrete))

        self.mem_size = max_size
        self.mem_cntr = 0
        self.discrete = discrete
        self.state_memory = np.zeros((self.mem_size,input_shape))
        self.new_state_memory = np.zeros((self.mem_size, input_shape))
        dtype = np.int8 if self.discrete else np.float32
        self.action_memory = np.zeros((self.mem_size, n_actions),dtype=dtype)
        self.reward_memory = np.zeros((self.mem_size))
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.float32)


        #TODO added this to eliminate illigal actions
        self.possibleActions_state = np.zeros((self.mem_size, n_actions),dtype='bool')
        self.possibleActions_new_state = np.zeros((self.mem_size, n_actions), dtype='bool')



    def store_transition(self, state, action, reward, state_, done, possible_Actions_state, possible_Actions_new_state):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done

        #TODO possible actions -> delete discrete than??

        if self.discrete:
            actions = np.zeros(self.action_memory[1].size)
            actions[action] = 1.0
            self.action_memory[index] = actions


            #ToDO added this to avoid estimating  illgeal actions
            self.possibleActions_state[index][possible_Actions_state] = True
            if not done:
                self.possibleActions_new_state[index][possible_Actions_new_state] = True
        else:
            self.action_memory[index] = action
        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size)

        states = self.state_memory[batch]
        states_ = self.new_state_memory[batch]
        rewards = self.reward_memory[batch]
        actions = self.action_memory[batch]
        terminal = self.terminal_memory[batch]

        possible_actions_state = self.possibleActions_state[batch]
        possible_actions_new_state = self.possibleActions_new_state[batch]

        return states, actions, rewards, states_, terminal, possible_actions_state, possible_actions_new_state

def build_dqn(lr,n_actions, input_dims, fcl_dims, fc2_dims):
    logging.getLogger('log1').info("Build a NN:  Input Shape: " + str(input_dims) + " Output Shape: " + str(n_actions) + "Layer: 3 Optimiser: Adam")
    logging.getLogger('log1').info("1. Layer No Neurons:"+ str(fcl_dims))
    logging.getLogger('log1').info("1. Layer Activation: Relu")

    logging.getLogger('log1').info("2. Layer No Neurons:" + str(fc2_dims))
    logging.getLogger('log1').info("2. Layer Activation: Relu")

    logging.getLogger('log1').info("3. Layer No Neurons:" + str(fcl_dims))
    logging.getLogger('log1').info("1. Layer Activation: Relu")

    logging.getLogger('log1').info("Loss function: Mean Square Error")


    model = Sequential([Dense(fcl_dims, input_shape=(input_dims, )),
                        Activation('relu'),
                        Dense(fc2_dims, activity_regularizer=l2(0.001)),
                        Activation('relu'),
                        Dense(fcl_dims,activity_regularizer= l2(0.001)),
                        Activation('relu'),
                        Dense(n_actions,activity_regularizer= l1(0.001))])

    logging.getLogger('log1').info("Compile NN")
    model.compile(optimizer=Adam(lr=lr), loss='mse')
    logging.getLogger('log1').info("Finish build NN")
    return model


class DQNAgent(object):
    def __init__(self,alpha, gamma, n_actions, epsilon, batch_size, input_dims,epsilon_dec=0.996, epsilon_end=0.01, mem_size=1000_000, fname='dqn_model.h5.22032020'):
        np.random.seed(0)
        tf.random.set_seed(0)


        logging.getLogger('log1').info("Init DQN-Agent: ALPHA: " + str(alpha) + " GAMMA: " + str(gamma)
                                       +" Replay Buffer Memory Size: "+str(mem_size)+" Model name: "+fname+
                                       " Epsilon Decrement: "+str(epsilon_dec)+" Batch Size: "+str(batch_size))
        self.action_space = [i for i in range(n_actions)]
        self.n_actions = n_actions
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_dec = epsilon_dec
        self.epsilon_min = epsilon_end
        self.batch_size = batch_size
        self.model_file = fname

        self.memory = ReplayBuffer(mem_size,input_dims,n_actions,discrete=True)

        logging.getLogger('log1').info("Start building Q Evaluation NN")
        self.q_eval = build_dqn(alpha, n_actions, input_dims, 350,400)



        #Add target network for stability and update it delayed to q_eval
        logging.getLogger('log1').info("Start building Q Target NN")
        self.q_target_model = build_dqn(alpha, n_actions, input_dims, 350,400)
        logging.getLogger('log1').info("Copy weights of Evaluation NN to Target Network")
        self.q_target_model.set_weights(self.q_eval.get_weights())

        self.target_update_counter = 0

        self.UPDATE_TARGET = 20


    def remember(self,state, action, reward, new_state, done,possible_Actions_state,possible_Actions_new_state):
        self.memory.store_transition(state,action,reward,new_state,done,possible_Actions_state,possible_Actions_new_state)

    def choose_action(self,state,possibleactions):
        state = state[np.newaxis, :]
        rand = np.random.random()
        if rand < self.epsilon:
            action = np.random.choice(self.action_space)
            #action = np.random.choice(possibleactions)
            #action = actionSample
        else:
            actions = self.q_eval.predict(state)
            #action_space_red = np.array(self.action_space.copy())[possibleactions]
            #action_qVal_red = actions[0][possibleactions]
            #action = action_space_red[np.argmax(action_qVal_red)]
            action = np.argmax(actions)
        return action

    def learn(self):
        if self.memory.mem_cntr <= self.batch_size:
            return

        logging.getLogger('log1').info("Learning Step - sample from replay buffer")

        state, action, reward, new_state, done, possible_actions_state, possible_actions_new_state = \
                                        self.memory.sample_buffer(self.batch_size)


        action_values = np.array(self.action_space, dtype=np.int8)
        action_indices = np.dot(action, action_values)


        if (state.ndim == 1):
            state = np.array([state])
        if (new_state.ndim == 1):
            new_state = np.array([new_state])

        q_eval = self.q_eval.predict(state)
        q_next = self.q_eval.predict(new_state)


        #q_target = q_eval.copy()
        #q_target = q_eval[:]

        q_target = self.q_target_model.predict(state)

        #q_target[possible_actions_state] = 0.
        #q_next[possible_actions_new_state] = -5.



        #print(q_target[possible_actions_state])


        #self.q_target_model.set_weights(self.q_eval.get_weights())

        batch_index = np.arange(self.batch_size, dtype=np.int32)


        #print(q_target[batch_index,action_indices])
        try:
            q_target[batch_index,action_indices] = reward + \
                self.gamma*np.max(q_next, axis=1)*(1-done)


            _ = self.q_eval.fit(state, q_target, verbose=0)
        except:
            print("BUUUH")

        #self.q_eval.train_on_batch(state, q_target)

        self.epsilon = self.epsilon*self.epsilon_dec if self.epsilon > self.epsilon_min else self.epsilon_min

        if self.target_update_counter > self.UPDATE_TARGET:
            self.q_target_model.set_weights(self.q_eval.get_weights())
            self.target_update_counter = 0
            logging.getLogger('log1').info("Copy weights of Evaluation NN to Target Network")

        self.target_update_counter += 1

    def save_model(self):
        ##file path
        self.q_eval.save(self.model_file)

    def load_model(self):
        self.q_eval = load_model(self.model_file)

