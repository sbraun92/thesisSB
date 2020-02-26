from env.RoRoDeck import RoRoDeck
import numpy as np
import matplotlib.pyplot as plt
import tqdm
import time
import logging
import pickle
import csv

class TDQLearning(object):
    def __init__(self, path, numGames=20000, help= True, GAMMA = 0.999):
        #help only for timing
        self.help = help
        logging.info("Initilise TD-Q-Learning Agent")
        self.numGames=numGames
        self.q_table = {}

        self.GAMMA = GAMMA
        #self.env = env
        self.path = path


    #TODO Output QTable
    #TODO Load QTable


    def train(self,env):
        logging.getLogger('log1').info("prepare training...")
        initState = env.reset()

        self.actionSpace_length = len(env.actionSpace)
        # ix_Actions = np.arange(len(env.actionSpace))
        # print(env.actionSpace)
        self.action_list = []

        # for ix, i in enumerate(env.actionSpace.keys()):
        #    ix_Actions[i] = ix
        #   action_list += [i]

        #logging.getLogger('log1').info("initilise Q table")
        self.q_table[initState.tobytes()] = np.zeros(self.actionSpace_length)


        self.ALPHA = 0.1
        self.GAMMA = 0.999
        self.EPS = 1.0
        self.MAX_IT = 400

        logging.getLogger('log1').info("Use param: ALPHA: "+ str(self.ALPHA)+" GAMMA: "+str(self.GAMMA))


        self.totalRewards = np.zeros(self.numGames)
        self.stateExpantion = np.zeros(self.numGames)
        self.stepsToExit = np.zeros(self.numGames)

        print("Start Training Process")
        logging.getLogger('log1').info("Start training process")
        for i in tqdm.tqdm(range(self.numGames)):
            self.done = False
            self.epReward = 0
            self.observation = env.reset()
            self.steps = 0

            while not self.done:
                # Show for visualisation the last training epoch
                self.rand = np.random.random()
                self.action = self.maxAction(self.q_table, self.observation, env.possibleActions) if self.rand < (1 - self.EPS) \
                    else env.actionSpaceSample()

                self.observation_, self.reward, self.done, self.info = env.step(self.action)
                self.steps += 1

                #Log Loading Sequence
                if i == self.numGames-1:
                   logging.getLogger('log2').info("Current Lane:"+str(self.observation[-1])+" Action:"+str(self.action))

                if self.observation_.tobytes() not in self.q_table:
                    self.q_table[self.observation_.tobytes()] = np.zeros(self.actionSpace_length)

                self.epReward += self.reward

                self.action_ = self.maxAction(self.q_table, self.observation_, env.possibleActions)

                # TD-Q-Learning with Epsilon-Greedy
                if not self.done:
                    self.q_table[self.observation.tobytes()][self.action] += self.ALPHA * (
                                self.reward + self.GAMMA * self.q_table[self.observation_.tobytes()][self.action_]
                                - self.q_table[self.observation.tobytes()][self.action])

                #Value of Terminal State is zero
                else:
                    self.q_table[self.observation.tobytes()][self.action] += self.ALPHA * (
                                self.reward - self.q_table[self.observation.tobytes()][self.action])

                #if not self.done:
                #    self.q_table[self.observation.tobytes()][self.action] = self.reward \
                #                                    + self.GAMMA * self.q_table[self.observation_.tobytes()][self.action_]

                self.observation = self.observation_

                if i == self.numGames - 1 and self.done == True:
                    env.render()
                    logging.getLogger('log1').info(env.render())
                    print("The reward of the last training episode was "+str(self.epReward))
                    print("The Terminal reward was "+ str(self.reward))
                    env.saveStowagePlan(self.path)


                #If agent doesnt reach end break here - seems unnessary when there is no switch Lane Option
                if self.steps > self.MAX_IT:
                    break

            logging.getLogger('log1').info("It" + str(i) + " EPS: " + str(self.EPS) + " reward: " + str(self.epReward))
            # Epsilon decreases lineary during training TODO 50 is arbitrary
            if 1. - i / (self.numGames - 50) > 0:
                self.EPS -= 1. / (self.numGames - 50)
            else:
                self.EPS = 0

            self.totalRewards[i] = self.epReward
            self.stateExpantion[i] = len(self.q_table.keys())
            self.stepsToExit[i] = self.steps
        logging.getLogger('log1').info("End training process")
        return self.q_table, self.totalRewards, self.stateExpantion, self.stepsToExit



    #TODO Warum so kompliziert... and slow
    def maxAction(self, Q, state, actions):
        if self.help == True:
            argSorted_qValues = np.flipud(np.argsort(Q[state.tobytes()]))
            if np.size(np.nonzero(Q[state.tobytes()]))== 0:
                if np.size(actions)==0:
                    return None
                elif np.size(actions)==1:
                    return actions[0]
                else:
                    return np.random.choice(actions)
            for ix_q_values in argSorted_qValues:
                if ix_q_values in actions:
                    #print(ix_q_values)
                    return ix_q_values

        #TODO check why this is not working -> possible actions -> redunadant when going to use Accept / reject logic
        else:
            return np.argmax(Q[state.tobytes()])

    #TODO try feather
    def load(self,path):
        try:
            self.q_table = pickle.load(open(path, "rb"))
        except:
            logging.getLogger("log1").error("Could not load pickle file")

    def save(self,path,type='pickle'):
        #path = path + '_qTablePickled.p'
        print(path)
        try:
            if type == pickle:
                pickle.dump(self.q_table, open(path+'_qTablePickled.p', "wb"))
            else:
                with open(path+'_qTable.csv', 'w') as f:
                    for key in self.q_table.keys():
                        f.write("%s,%s\n" % (key,  self.q_table[key]))
        except:
            if type==pickle:
                logging.getLogger("log1").error("Could not save pickle file to+ " + path)
            else:
                logging.getLogger("log1").error("Could not save csv file to+ " + path)

