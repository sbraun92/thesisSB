# RORO-Terminal Enviroment based on a GridWorld; is OpenAI Gym complying
import numpy as np
import matplotlib.pyplot as plt


class RoRoDeck(object):
    def __init__(self, lanes=10, rows=12):
        self.lanes = lanes
        self.rows = rows
        self.sequence_no = 1
        self.grid = self.createGrid()
        self.endOfLanes = self.getEndOfLane(self.grid)
        self.capacity = self.getFreeCapacity(self.grid)
        self.currentLane = self.getMinimalLanes()[0]
        self.frontier = self.getFrontier()

        # State-Repräsentation Frontier, BackLook, CurrentLane
        self.currentState = self.getCurrentState()

        #Test without switching
        self.actionSpace_names = {0: 'Type1', 1: 'Type2'}
        self.actionSpace = np.array([0,1])
        self.action2vehicleLength = np.array([2, 3])

        #self.actionSpace_names = {0: 'Switch', 1: 'Type1', 2: 'Type2'}
        #self.actionSpace = np.array([0,1,2])
        #self.action2vehicleLength = np.array([0, 2, 3])
        self.minimalPackage = np.min(self.action2vehicleLength[np.where(self.action2vehicleLength > 0)])
        self.possibleActions = self.possibleActionsOfState()
        self.maxSteps = 0
        self.TerminalStateCounter = 0

    def reset(self):
        self.sequence_no = 1
        self.grid = self.createGrid()
        self.endOfLanes = self.getEndOfLane(self.grid)
        self.capacity = self.getFreeCapacity(self.grid)
        self.currentLane = self.getMinimalLanes()[0]
        self.frontier = self.getFrontier()
        self.currentState = np.hstack((self.frontier, self.endOfLanes, self.currentLane))
        self.possibleActions = self.possibleActionsOfState()

        self.maxSteps = 0
        self.TerminalStateCounter = 0

        return self.getCurrentState()

    def render(self):
        print('---------------------------------------------------------------------------')
        for row in self.grid:
            for col in row:
                if col == -1:
                    print('X', end='\t')
                elif col == 0:
                    print('-', end='\t')
                else:
                    print(str(int(col)), end='\t')
            print('\n')
            print('---------------------------------------------------------------------------')

    def actionSpaceSample(self):
        return np.random.choice(self.possibleActions)

    # Done
    # Return a grid world with a arrow shaped ships hule
    def createGrid(self):
        grid = np.zeros((self.rows, self.lanes))
        for i in range(4):
            t = 4 - i
            grid[i] += np.hstack([-np.ones(t), np.zeros(self.lanes - t)])
            grid[i] += np.hstack([np.zeros(self.lanes - t), -np.ones(t)])
        return grid

    # Return an Array with the Indicies of the last free space
    # Find indcies of last free slot in lane (if full set -1)
    def getEndOfLane(self, grid):
        endOfLanes = np.zeros(len(grid.T), dtype=np.int32)
        for idx, lane in enumerate(grid.T):
            emptySpaceInLanes = np.argwhere(lane != 0)
            if emptySpaceInLanes.size != 0:
                endOfLanes[idx] = emptySpaceInLanes[-1] + 1

            if grid.T[idx][-1] != 0:
                endOfLanes[idx] = -1
        return endOfLanes

    # Return Array of which indicates how much space is free in each lane
    def getFreeCapacity(self, grid):
        capacity = np.ones(len(grid.T)) * len(grid)
        capacity -= np.count_nonzero(grid, axis=0)
        return capacity

    def getFrontier(self):
        return np.max(self.endOfLanes)

    def findCurrentLane(self, endOfLanes):
        return np.argmin(self.endOfLanes)

    def getCurrentLaneAfterPut(self):
        return self.getMinimalLanes()[0]

    def switchCurrentLane(self):
        minimalLanes = self.getMinimalLanes()
        if minimalLanes.size == 1:
            return minimalLanes[0]
        else:
            position_in_currentLanes = int(np.argwhere(minimalLanes==self.currentLane))
            newLane = minimalLanes[(position_in_currentLanes +1)%len(minimalLanes)]
            return newLane

    def getMinimalLanes(self):
        return np.argwhere(self.endOfLanes == np.min(self.endOfLanes)).flatten()

    def isActionLegal(self, action):
        if self.endOfLanes[self.currentLane] + self.action2vehicleLength[action] <= self.rows:
            return True
        else:
            return False

    # return an array such as [0,2] - possible lengths ordered
    def possibleActionsOfState(self):
        possibleActions = []
        for action in range(len(self.actionSpace)):
            if not self.isActionLegal(action):
                break
            else:
                possibleActions += [action]
        return np.array(possibleActions)

    def isVesselFull(self):
        return np.size(np.where(self.endOfLanes + (np.ones(self.lanes) * self.minimalPackage) <= self.rows)) == 0

    def isTerminalState(self):
        if self.frontier + self.minimalPackage < self.rows:
            return False
        if self.isVesselFull() or np.size(self.possibleActions)==0: #or (self.TerminalStateCounter > np.size(self.getMinimalLanes()) * 5) or (self.maxSteps > 500):
            return True
        else:
            return False

    def getCurrentState(self):
        return np.hstack((self.frontier, self.endOfLanes, self.currentLane)).astype(np.int32)

    def step(self, action):
        # Must return new State, reward, if it is a TerminalState

        # 1. Check if action was legal
        # 2. Calculate reward
        # 3. Update grid
        # 4. Update EndOfLane
        # 5. Update Frontier
        # 6. Update CurrentLane
        # 7. Update SequenceNumber

        #self.maxSteps += 0.1
        if not self.isActionLegal(action):
            print("Action was not Legal. There is an error in the legal action machine")
            return self.getCurrentState(), -1, self.isTerminalState(), None
        else:
            reward = 0
            #Remove Switching-Option
            if self.actionSpace[action] == -1:
                self.currentLane = self.switchCurrentLane()
                #reward = -1
                #self.TerminalStateCounter += 1
            else:
                slot = self.endOfLanes[self.currentLane]
                self.endOfLanes[self.currentLane] += self.action2vehicleLength[action]
                for i in range(self.action2vehicleLength[action]):
                    self.grid.T[self.currentLane][slot + i] = self.sequence_no
                    reward += 0.1+i*0.6

                self.frontier = self.getFrontier()
                self.sequence_no += 1
                self.currentLane = self.getMinimalLanes()[0]  # better name updateCurrentLane

                # if we put a car we reset the TerminalStateCounter
                self.TerminalStateCounter = 0

            self.possibleActions = self.possibleActionsOfState()

            if self.isTerminalState():
                reward = -0.6*np.sum(-self.endOfLanes + np.ones(self.lanes) * (self.rows))
                #print(1*np.sum(-self.endOfLanes + np.ones(self.lanes) * (self.rows)))
            return self.getCurrentState(), reward, self.isTerminalState(), None

    def actionSpaceSample(self):
        return np.random.choice(self.possibleActions)