# RORO-Terminal Enviroment based on a GridWorld; is OpenAI Gym complying
import numpy as np
import matplotlib.pyplot as plt
import logging

np.random.seed(0)
class RoRoDeck(object):
    def __init__(self, help, lanes=8, rows=10):
        #just to compare runtimes
        #TODO delete helper from
        self.help = help


        logging.getLogger('log1').info('Initilise Enviroment')
        self.lanes = lanes
        self.rows = rows
        self.sequence_no = 1
        self.grid = self.createGrid()
        self.gridDestination = self.createGrid()
        self.endOfLanes = self.getEndOfLane(self.grid)
        self.currentLane = self.getMinimalLanes()[0]

        #Vehicle Data stores vehicle id, destination, if it is mandatory cargo, length and how many to be loaded max
        self.vehicleData = np.array([[0, 1, 2, 3], #vehicle id
                                     [1, 2, 1, 2], #destination
                                     [1, 1, 0, 0], #madatory
                                     [2, 3, 2, 3], #length
                                     [5, 5,-1,-1]]) #number of vehicles on yard (-1 denotes there are infinite vehicles of that type

        self.capacity = self.getFreeCapacity(self.grid)
        self.frontier = self.getFrontier()
        self.numberOfVehicles = self.vehicleData[4].copy()
        #for shifts TODO not a good name
        self.shiftHelper = self.endOfLanes.copy()
        #self.prevVeh = self.endOfLanes.copy()

        #mandatory cargo, must be loaded
        self.mandatoryCargo = self.vehicleData[4][self.vehicleData[2]==1]

        # State-Repräsentation Frontier, BackLook,mandatory cargo, CurrentLane
        self.currentState = self.getCurrentState()

        #Test without switching
        self.actionSpace_names = {0: 'Type1', 1: 'Type2'}
        self.actionSpace = np.array([0,1])
        self.action2vehicleLength = np.array([2, 3])
        self.action2destination = np.array([1,2])

        #self.actionSpace_names = {0: 'Switch', 1: 'Type1', 2: 'Type2'}
        #self.actionSpace = np.array([0,1,2])
        #self.action2vehicleLength = np.array([0, 2, 3])
        #TODO unnoetig??
        #self.minimalPackage = np.min(self.action2vehicleLength[np.where(self.action2vehicleLength > 0)])


        self.minimalPackage = np.min(self.vehicleData[3])
        self.possibleActions = self.possibleActionsOfState()
        self.maxSteps = 0
        self.TerminalStateCounter = 0

    def reset(self):
        logging.getLogger('log1').info('Reset Environment')

        self.sequence_no = 1
        self.grid = self.createGrid()
        self.gridDestination = self.createGrid()
        self.endOfLanes = self.getEndOfLane(self.grid)
        self.numberOfVehicles = self.vehicleData[4].copy()

        self.capacity = self.getFreeCapacity(self.grid)
        self.currentLane = self.getMinimalLanes()[0]
        self.frontier = self.getFrontier()
        #self.currentState = np.hstack((self.frontier, self.endOfLanes, self.currentLane))
        self.currentState = self.getCurrentState()
        self.possibleActions = self.possibleActionsOfState()


        self.maxSteps = 0
        self.TerminalStateCounter = 0

        self.shiftHelper = self.endOfLanes.copy()

        self.mandatoryCargo = 5*np.ones(2)

        return self.getCurrentState()

    def render(self):
        print('-----------Loading Sequence----------------------------------------------------------------')
        for row in self.grid:
            #Loading Sequence
            for col in row:
                if col == -1:
                    print('X', end='\t')
                elif col == 0:
                    print('-', end='\t')
                else:
                    print(str(int(col)), end='\t')
            print('\n')
        print('-------Destination--------------------------------------------------------------------')
        for row in self.gridDestination:
            for col in row:
                if col == -1:
                    print('X', end='\t')
                elif col == 0:
                    print('-', end='\t')
                else:
                    print(str(int(col)), end='\t')
            print('\n')


    def actionSpaceSample(self):
        return np.random.choice(self.possibleActions)

    # Done
    # Return a grid world with a arrow shaped ships hule
    def createGrid(self):
        grid = np.zeros((self.rows, self.lanes),dtype=np.int32)
        for i in range(4):
            t = 4 - i
            grid[i] += np.hstack([-np.ones(t,dtype=np.int32), np.zeros(self.lanes - t,dtype=np.int32)])
            grid[i] += np.hstack([np.zeros(self.lanes - t,dtype=np.int32), -np.ones(t,dtype=np.int32)])
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
        if self.endOfLanes[self.currentLane] + self.vehicleData[3][action] <= self.rows:
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
        return np.hstack((self.frontier, self.endOfLanes, self.mandatoryCargo ,self.currentLane)).astype(np.int32)

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
            reward = 0#self.calculateReward()
            numberOfShifts = self.getNumberOfShifts(action)
            reward -= numberOfShifts * 6  # +self.action2vehicleLength[action]*0.6
            #Remove Switching-Option
            if self.actionSpace[action] == -1:
                self.currentLane = self.switchCurrentLane()
                #reward = -1
                #self.TerminalStateCounter += 1
            else:
                slot = self.endOfLanes[self.currentLane]
                self.endOfLanes[self.currentLane] += self.action2vehicleLength[action]



                if self.numberOfVehicles[action] > 0 and self.vehicleData[2][action]== 1:
                    self.numberOfVehicles[action]-=1
                    reward+=2

                #if self.mandatoryCargo[action] > 0:
                 #   self.mandatoryCargo[action]-=1
                  #  reward+=2

                for i in range(self.vehicleData[3][action]):
                    self.grid.T[self.currentLane][slot + i] = self.sequence_no
                    self.gridDestination.T[self.currentLane][slot + i] = self.vehicleData[1][action]




                self.frontier = self.getFrontier()
                self.sequence_no += 1
                self.currentLane = self.getMinimalLanes()[0]  # better name updateCurrentLane

                # if we put a car we reset the TerminalStateCounter
                self.TerminalStateCounter = 0

            self.possibleActions = self.possibleActionsOfState()

            if self.isTerminalState():
                reward = -2*np.sum(-self.endOfLanes + np.ones(self.lanes) * (self.rows))
                reward -= np.sum(self.numberOfVehicles[self.vehicleData[2]==1])*20
            return self.getCurrentState(), reward, self.isTerminalState(), None

    def actionSpaceSample(self):
        return np.random.choice(self.possibleActions)

    def getNumberOfShifts(self, action):
        destination = self.vehicleData[1][action]

        noVehDest1 = len(np.argwhere(self.gridDestination.T[self.currentLane] == 1))

        if destination == 2 and noVehDest1 !=0:
            return 1
        else:
            return 0




    def heuristic(self):
        # 1. Destination
        # 2. Mandatory
        # 3. Length
        pass



    #TODO Test, make sure this is not envoked during simulations
    def add_CargoType(self, destination, mandatory, length, number):
        assert destination == 1 or destination == 2
        assert mandatory == 0 or mandatory == 1
        assert length > 0 and length < len(self.grid[0])-4
        assert number == -1 or number > 0

        typeNo = self.vehicleData[0][-1] + 1
        newCargo = np.array([typeNo,destination,mandatory,length,number])
        self.vehicleData = np.vstack((self.vehicleData.T,newCargo)).T


    def saveStowagePlan(self,path):
        stowagePlan = open(path+"_StowagePlan.txt", 'w')
        stowagePlan.write('Stowage Plan and Loading Sequence \n')
        stowagePlan.write('-----------Loading Sequence---------------------------------------------------------------- \n')
        for row in self.grid:
            # Loading Sequence
            for col in row:
                if col == -1:
                    stowagePlan.write('X \t')
                elif col == 0:
                    stowagePlan.write('- \t')
                else:
                    stowagePlan.write(str(int(col))+' \t')
            stowagePlan.write(' \n\n')
        stowagePlan.write('-------Destination-------------------------------------------------------------------- \n')
        for row in self.gridDestination:
            for col in row:
                if col == -1:
                    stowagePlan.write('X \t')
                elif col == 0:
                    stowagePlan.write('- \t')
                else:
                    stowagePlan.write(str(int(col))+' \t')
            stowagePlan.write('\n\n')

        # Close the file
        stowagePlan.close()