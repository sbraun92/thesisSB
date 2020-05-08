from env.roroDeck import RoRoDeck
import numpy as np
import matplotlib.pyplot as plt
#import tqdm
import time
import logging
import pickle
import csv

from agent.agentInterface import Agent



class TDQLearning(Agent):
    def __init__(self, env=None, path=None,  numGames=20000, orig= True, GAMMA = 0.999):
        np.random.seed(0)
        #help only for timing
        self.orig = orig
        logging.info("Initialise TD-Q-Learning Agent")
        self.numGames=numGames
        self.q_table = {}
        self.EPSdec = 0.9999
        self.EPSmin = 0.001
        self.GAMMA = GAMMA
        #self.env = env
        self.path = path
        self.env = env
        self.eps_history = []
        self.ALPHA = 0.1
        self.EPS = 1.0
        self.MAX_IT = 400
        self.actionSpace_length = len(self.env.actionSpace)
        self.action_ix = np.arange(self.actionSpace_length)
        # ix_Actions = np.arange(len(env.actionSpace))
        # print(env.actionSpace)
        #self.action_list = []

    #TODO Output QTable
    #TODO Load QTable


    def train(self):
        logging.getLogger('log1').info("prepare training...")

        start = time.time()


        initState = self.env.reset()

        # for ix, i in enumerate(env.actionSpace.keys()):
        #    ix_Actions[i] = ix
        #   action_list += [i]

        #logging.getLogger('log1').info("initilise Q table")
        self.q_table[initState.tobytes()] = np.zeros(self.actionSpace_length)

        logging.getLogger('log1').info("Use param: ALPHA: "+ str(self.ALPHA)+" GAMMA: "+str(self.GAMMA))


        self.totalRewards = np.zeros(self.numGames)
        self.stateExpantion = np.zeros(self.numGames)
        self.stepsToExit = np.zeros(self.numGames)



        print("Start Training Process")
        logging.getLogger('log1').info("Start training process")
        for i in range(self.numGames):
            self.done = False
            self.epReward = 0
            self.observation = self.env.reset()
            self.steps = 0

            while not self.done:
                # Show for visualisation the last training epoch
                self.rand = np.random.random()
                self.action = self.maxAction(self.observation) if self.rand < (1 - self.EPS) \
                    else self.env.action_space_sample()
                #self.env.render()
                #TODO delete
                #if i == self.numGames - 1:
                #    print("----")
                #    print(self.q_table[self.observation.tobytes()])
                #    print(self.action)

                self.observation_, self.reward, self.done, self.info = self.env.step(self.action)
                self.steps += 1

                #Log Loading Sequence
                if i == self.numGames-1:
                   logging.getLogger('log2').info("Current Lane:"+str(self.observation[-1])+" Action:"+str(self.action))

                if self.observation_.tobytes() not in self.q_table:
                    self.q_table[self.observation_.tobytes()] = np.zeros(self.actionSpace_length)

                self.epReward += self.reward



                # TD-Q-Learning with Epsilon-Greedy
                if not self.done:
                    self.action_ = self.maxAction(self.observation_)

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
                    self.env.render()
                    logging.getLogger('log1').info(self.env.render())
                    print("The reward of the last training episode was "+str(self.epReward))
                    print("The Terminal reward was "+ str(self.reward))
                    print(self.path)
                    if self.path!=None:
                        self.env.saveStowagePlan(self.path)


                #If agent doesnt reach end break here - seems unnessary when there is no switch Lane Option
                if self.steps > self.MAX_IT:
                    break

            logging.getLogger('log1').info("It" + str(i) + " EPS: " + str(self.EPS) + " reward: " + str(self.epReward))
            # Epsilon decreases lineary during training TODO 50 is arbitrary




            if 1. - i / (self.numGames - 100) > 0:
                self.EPS -= 1. / (self.numGames - 100)
            else:
                self.EPS = 0

            #if self.EPS > self.EPSmin:
            #    self.EPS *= self.EPSdec
            #else:
            #    self.EPS = self.EPSmin


            self.eps_history.append(self.EPS)


            self.totalRewards[i] = self.epReward
            self.stateExpantion[i] = len(self.q_table.keys())
            self.stepsToExit[i] = self.steps

            avg_reward = np.mean(self.totalRewards[max(0, i - 100):(i + 1)])
            std_reward = np.std(self.totalRewards[max(0, i - 100):(i + 1)])
            if i % 500 == 0 and i > 0:
                print('episode ', i, 'score %.2f' % self.epReward, '\tavg. score %.2f' % avg_reward, '\tstd of score %.2f' % std_reward)

            #if i%500 == 0:
            #    print(len(self.q_table.keys()))

        logging.getLogger('log1').info("End training process")
        self.training_time = time.time()-start
        return self.q_table, self.totalRewards, self.stateExpantion, self.stepsToExit, np.array(self.eps_history)

    # TODO cleanup
    #def maxAction(self, state):
    #    #print(self.env.possibleActions)
    #    possibleActions = self.action_ix[self.env.possible_actions]
        #print(possibleActions)
        #print(Q[state.tobytes()])
    #    positionsOfBestPossibleAction = np.argmax(self.q_table[state.tobytes()][self.env.possible_actions])
        #print(positionsOfBestPossibleAction)
    #    return possibleActions[positionsOfBestPossibleAction]


    #TODO try feather
    def load_model(self,path):
        try:
            self.q_table = pickle.load(open(path+'_qTablePickled.p', "rb"))
        except:
            logging.getLogger("log1").error("Could not load pickle file")

    def save_model(self,path,type='pickle'):
        self.q_table["ModelParam"] = {"Algorithm": "Time Difference Q-Learning",
                                      "GAMMA": self.GAMMA,
                                      "ALPHA": self.ALPHA,
                                      "Episodes": self.numGames,
                                      "EnvLanes:": self.env.lanes,
                                      "EnvRows": self.env.rows,
                                      "VehicleData": self.env.vehicle_data,
                                      "TrainingTime":self.training_time}
        info = "_TDQ" + "_L" + str(self.env.lanes) + "_R" + str(self.env.rows) + "_Rf" + \
               str(int(1 in self.env.vehicle_data[5])) + "_A" + str(len(self.env.vehicle_data[0]))

        #path = path + '_qTablePickled.p'

        super().save_model(path+info)


    def execute(self, humanInteraction = False):
        self.observation = self.env.reset()
        self.done = False

        while not self.done:
            self.action = self.maxAction(self.q_table, self.observation, self.env.possible_actions)
            self.observation, self.reward, self.done, self.info = self.env.step(self.action)
            self.epReward += self.reward


        self.env.render()