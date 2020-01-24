from env.RoRoDeck import RoRoDeck
import numpy as np
import matplotlib.pyplot as plt
import tqdm
import time



class TDQLearning(object):
    def __init__(self, numGames=20000, GAMMA = 0.999):
        self.numGames=numGames
        self.GAMMA = GAMMA
        #self.env = env



    def train(self,env):
        initState = env.reset()

        self.actionSpace_length = len(env.actionSpace)
        # ix_Actions = np.arange(len(env.actionSpace))
        # print(env.actionSpace)
        self.action_list = []

        # for ix, i in enumerate(env.actionSpace.keys()):
        #    ix_Actions[i] = ix
        #   action_list += [i]

        self.q_table = {initState.tobytes(): np.zeros(self.actionSpace_length)}

        self.ALPHA = 0.1
        self.GAMMA = 0.999
        self.EPS = 1.0
        self.MAX_IT = 400

        self.totalRewards = np.zeros(self.numGames)
        self.stateExpantion = np.zeros(self.numGames)
        self.stepsToExit = np.zeros(self.numGames)

        print("Start Training Process")
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

                if self.observation_.tobytes() not in self.q_table:
                    self.q_table[self.observation_.tobytes()] = np.zeros(self.actionSpace_length)

                self.epReward += self.reward

                self.action_ = self.maxAction(self.q_table, self.observation_, env.possibleActions)

                # TD-Q-Learning with Epsilon-Greedy
                '''if not done:
                    q_table[observation.tobytes()][action] += ALPHA * (
                                reward + GAMMA * q_table[observation_.tobytes()][action_]
                                - q_table[observation.tobytes()][action])
                '''
                if not self.done:
                    self.q_table[self.observation.tobytes()][self.action] = self.reward + self.GAMMA * self.q_table[self.observation_.tobytes()][self.action_]

                self.observation = self.observation_

                if i == self.numGames - 1:
                    print(self.action)
                    env.render()
                    print(self.epReward)

                if self.steps > self.MAX_IT:
                    # print("That took too long")
                    break
            # Epsilon decreases lineary during training
            if 1. - i / (self.numGames - 50) > 0:
                self.EPS -= 1. / (self.numGames - 50)
            else:
                self.EPS = 0

            self.totalRewards[i] = self.epReward

            # if max(totalRewards)-1<=epReward:
            #   print("New Best")
            #   print(i)
            #  print(epReward)
            #  env.render()
            self.stateExpantion[i] = len(self.q_table.keys())
            self.stepsToExit[i] = self.steps
        return self.q_table, self.totalRewards, self.stateExpantion, self.stepsToExit



    def maxAction(self, Q, state, actions):
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



'''
if __name__ == '__main__':

    env = RoRoDeck(10,15)
    initState = env.reset()

    actionSpace_length = len(env.actionSpace)
    #ix_Actions = np.arange(len(env.actionSpace))
    #print(env.actionSpace)
    action_list = []

    # for ix, i in enumerate(env.actionSpace.keys()):
    #    ix_Actions[i] = ix
    #   action_list += [i]

    q_table = {initState.tobytes(): np.zeros(actionSpace_length)}

    ALPHA = 0.1
    GAMMA = 0.999
    EPS = 1.0
    MAX_IT = 400


    numGames = 50000
    totalRewards = np.zeros(numGames)
    stateExpantion = np.zeros(numGames)
    stepsToExit = np.zeros(numGames)
    #env.render()

    for i in tqdm.tqdm(range(numGames)):
        #print(EPS)
        #if i % 2000 == 0:
            #print('learning process epoch:', i)

        done = False
        epReward = 0
        observation = env.reset()
        steps = 0

        while not done:
            # Show for visualisation the last training epoch

            rand = np.random.random()
            action = maxAction(q_table, observation, env.possibleActions) if rand < (1 - EPS) \
                else env.actionSpaceSample()

            observation_, reward, done, info = env.step(action)
            steps +=1


            if observation_.tobytes() not in q_table:
                q_table[observation_.tobytes()] = np.zeros(actionSpace_length)

            epReward += reward

            action_ = maxAction(q_table, observation_, env.possibleActions)

            # TD-Q-Learning with Epsilon-Greedy
            #if not done:
             #   q_table[observation.tobytes()][action] += ALPHA * (
              #              reward + GAMMA * q_table[observation_.tobytes()][action_]
               #             - q_table[observation.tobytes()][action])
       
            if not done:
                q_table[observation.tobytes()][action] = reward + GAMMA * q_table[observation_.tobytes()][action_]

            observation = observation_

            if i == numGames - 1:
                print(action)
                env.render()
                print(epReward)

            if steps > MAX_IT:
                #print("That took too long")
                break
        # Epsilon decreases lineary during training
        if 1. - i / (numGames-50) > 0:
            EPS -= 1. / (numGames-50)
        else:
            EPS = 0

        totalRewards[i] = epReward

        #if max(totalRewards)-1<=epReward:
         #   print("New Best")
         #   print(i)
          #  print(epReward)
          #  env.render()
        stateExpantion[i] = len(q_table.keys())
        stepsToExit[i] = steps



'''