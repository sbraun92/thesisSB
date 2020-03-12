# RORO-Terminal Enviroment based on a GridWorld; is OpenAI Gym complying
import numpy as np
import matplotlib.pyplot as plt
import logging

np.random.seed(0)


class RoRoDeck(object):
    """
    Enviroment-Class of a RORO-Deck

    public attributes:


    Methods:
        reset()
            Reset method
        step()
            does one simulation ste
        render()
            Representation of the current state of the RORO-Deck
        actionSpaceSample()
            returns an random possible action
        possibleActionsOfState()
            returns all possible actions of the current state
        add_CargoType()
            adds a new Cargo type


    """

    def __init__(self, help, lanes=8, rows=10, rewardSystem = None):
        # just to compare runtimes
        # TODO delete helper from
        self.help = help

        logging.getLogger('log1').info('Initilise Enviroment')
        self.lanes = lanes
        self.rows = rows
        self.sequence_no = 1
        self.grid = self._createGrid()
        self.gridDestination = self._createGrid()
        self.gridVehicleType = self._createGrid()-1

        self.rewardSystem = np.array([2,        #simple Loading
                                      -6,       #caused shifts
                                      -2,       #terminal: Space left unsed
                                      -20])     #terminal: mand. cargo not loaded

        self.endOfLanes = self._getEndOfLane(self.grid)
        self.currentLane = self._getMinimalLanes()[0]

        # Vehicle Data stores vehicle id, destination, if it is mandatory cargo, length and how many to be loaded max
        self.vehicleData = np.array([[0, 1, 2, 3],  # vehicle id
                                     [1, 2, 1, 2],  # destination
                                     [1, 1, 0, 0],  # mandatory
                                     [2, 3, 2, 3],  # length
                                     [5, 5, -1,-1]])  # number of vehicles on yard
                                                      # (-1 denotes there are infinite vehicles of that type)
        self.mandatoryCargoMask = self.vehicleData[2] == 1
        self.loadedVehicles = -np.ones((self.lanes, np.min(self.vehicleData[3]) * self.rows), dtype=np.int16)
        self.vehicleCounter = np.zeros(self.lanes,dtype=np.int16)


        self.capacity = self._getFreeCapacity(self.grid)
        self.frontier = self._getFrontier()
        self.numberOfVehiclesLoaded = np.zeros(len(self.vehicleData[0]),dtype=np.int16)
        # for shifts TODO not a good name
        self.shiftHelper = self.endOfLanes.copy()
        # self.prevVeh = self.endOfLanes.copy()

        # mandatory cargo, must be loaded
        #self.mandatoryCargo = self.vehicleData[4][self.vehicleData[2] == 1]

        # State-Repräsentation Frontier, BackLook,mandatory cargo, CurrentLane
        self.currentState = self._getCurrentState()

        # Test without switching
        self.actionSpace_names = {0: 'Type1', 1: 'Type2'}
        self.actionSpace = self.vehicleData[0]
        self.action2vehicleLength = np.array([2, 3])
        self.action2destination = np.array([1, 2])

        # self.actionSpace_names = {0: 'Switch', 1: 'Type1', 2: 'Type2'}
        # self.actionSpace = np.array([0,1,2])
        # self.action2vehicleLength = np.array([0, 2, 3])
        # TODO unnoetig??
        # self.minimalPackage = np.min(self.action2vehicleLength[np.where(self.action2vehicleLength > 0)])

        self.minimalPackage = np.min(self.vehicleData[3])
        self.possibleActions = self.possibleActionsOfState()
        self.maxSteps = 0
        self.TerminalStateCounter = 0

    def reset(self):
        logging.getLogger('log1').info('Reset Environment')

        self.sequence_no = 1
        self.grid = self._createGrid()
        self.gridDestination = self._createGrid()
        self.gridVehicleType = self._createGrid()-1

        self.endOfLanes = self._getEndOfLane(self.grid)
        self.numberOfVehiclesLoaded = np.zeros(len(self.vehicleData[0]),dtype=np.int16)

        self.capacity = self._getFreeCapacity(self.grid)
        self.currentLane = self._getMinimalLanes()[0]
        self.frontier = self._getFrontier()
        # self.currentState = np.hstack((self.frontier, self.endOfLanes, self.currentLane))
        self.currentState = self._getCurrentState()
        self.possibleActions = self.possibleActionsOfState()

        self.maxSteps = 0
        self.TerminalStateCounter = 0

        self.shiftHelper = self.endOfLanes.copy()

        #self.mandatoryCargo = self.vehicleData[4][self.vehicleData[2] == 1]


        self.loadedVehicles = -np.ones((self.lanes, np.min(self.vehicleData[3]) * self.rows), dtype=np.int16)
        self.vehicleCounter = np.zeros(self.lanes,dtype=np.int16)


        return self._getCurrentState()

    def render(self):
        print('-----------Loading Sequence----------------------------------------------------------------')
        for row in self.grid:
            # Loading Sequence
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
    def _createGrid(self):
        grid = np.zeros((self.rows, self.lanes), dtype=np.int32)
        for i in range(4):
            t = 4 - i
            grid[i] += np.hstack([-np.ones(t, dtype=np.int32), np.zeros(self.lanes - t, dtype=np.int32)])
            grid[i] += np.hstack([np.zeros(self.lanes - t, dtype=np.int32), -np.ones(t, dtype=np.int32)])
        return grid

    # Return an Array with the Indicies of the last free space
    # Find indcies of last free slot in lane (if full set -1)
    def _getEndOfLane(self, grid):
        endOfLanes = np.zeros(len(grid.T), dtype=np.int32)
        for idx, lane in enumerate(grid.T):
            emptySpaceInLanes = np.argwhere(lane != 0)
            if emptySpaceInLanes.size != 0:
                endOfLanes[idx] = emptySpaceInLanes[-1] + 1

            if grid.T[idx][-1] != 0:
                endOfLanes[idx] = -1
        return endOfLanes

    # Return Array of which indicates how much space is free in each lane
    def _getFreeCapacity(self, grid):
        capacity = np.ones(len(grid.T)) * len(grid)
        capacity -= np.count_nonzero(grid, axis=0)
        return capacity

    def _getFrontier(self):
        return np.max(self.endOfLanes)

    def _findCurrentLane(self, endOfLanes):
        return np.argmin(self.endOfLanes)

    def _getCurrentLaneAfterPut(self):
        return self._getMinimalLanes()[0]

    def _switchCurrentLane(self):
        minimalLanes = self._getMinimalLanes()
        if minimalLanes.size == 1:
            return minimalLanes[0]
        else:
            position_in_currentLanes = int(np.argwhere(minimalLanes == self.currentLane))
            newLane = minimalLanes[(position_in_currentLanes + 1) % len(minimalLanes)]
            return newLane

    def _getMinimalLanes(self):
        return np.argwhere(self.endOfLanes == np.min(self.endOfLanes)).flatten()

    def _isActionLegal(self, action):
        if self.endOfLanes[self.currentLane] + self.vehicleData[3][action] <= self.rows:
            if self.vehicleData[4][action] == -1: # infinite Vehicles in parkinglot
                return True
            elif self.numberOfVehiclesLoaded[action]< self.vehicleData[4][action]: #enough vehicles in parking lot
                return True
            else:
                return False
        else:
            return False

    # return an array such as [0,2] - possible lengths ordered
    def possibleActionsOfState(self):
        possibleActions = []
        for action in range(len(self.actionSpace)):
            if self._isActionLegal(action):
                possibleActions += [action]
        return np.array(possibleActions)

    def _isVesselFull(self):
        return np.size(np.where(self.endOfLanes + (np.ones(self.lanes) * self.minimalPackage) <= self.rows)) == 0

    def _isTerminalState(self):
        # Check if the smallest Element still fits after the frontier element and
        # if there are still vehicles in the parking lot to be loaded
        if self.frontier + self.minimalPackage < self.rows and np.size(self.possibleActions) != 0:
            return False

        if (self._isVesselFull() or \
                 np.all((self.vehicleData[4]- self.numberOfVehiclesLoaded)==0)) or \
                np.size(self.possibleActions) == 0:
            return True
        else:
            return False
        #TODO Check if _isVesselFull method is redundant
        #if self._isVesselFull() or np.size(
        #        self.possibleActions) == 0:  # or (self.TerminalStateCounter > np.size(self.getMinimalLanes()) * 5) or (self.maxSteps > 500):
        #    return True
        #else:
        #    return False

    def _getCurrentState(self):
        return np.hstack((self.frontier, self.endOfLanes, self.numberOfVehiclesLoaded[self.mandatoryCargoMask], self.currentLane)).astype(np.int32)

    def step(self, action):
        # Must return new State, reward, if it is a TerminalState

        # 1. Check if action was legal
        # 2. Calculate reward
        # 3. Update grid
        # 4. Update EndOfLane
        # 5. Update Frontier
        # 6. Update CurrentLane
        # 7. Update SequenceNumber

        # self.maxSteps += 0.1
        if not self._isActionLegal(action):
            print("Action was not Legal. There is an error in the legal action machine")
            return self._getCurrentState(), -1, self._isTerminalState(), None
        else:
            reward = 0  # self.calculateReward()
            numberOfShifts = self._getNumberOfShifts(action)
            reward -= numberOfShifts * self.rewardSystem[1]  # +self.action2vehicleLength[action]*0.6
            # Remove Switching-Option
            if self.actionSpace[action] == -1:
                self.currentLane = self._switchCurrentLane()
                # reward = -1
                # self.TerminalStateCounter += 1
            else:
                slot = self.endOfLanes[self.currentLane]
                self.endOfLanes[self.currentLane] += self.vehicleData[3][action]

               # if self.numberOfVehiclesLoaded[action] > 0 and self.vehicleData[2][action] == 1:
                #    self.numberOfVehiclesLoaded[action] -= 1
                 #   reward += 2


                if self.vehicleData[4][action] == -1: #infinite vehicles on car park
                    self.numberOfVehiclesLoaded[action] += 1
                    reward += self.rewardSystem[0]
                elif self.numberOfVehiclesLoaded[action] < self.vehicleData[4][action]:
                    self.numberOfVehiclesLoaded[action] += 1
                    reward += self.rewardSystem[0]

                # if self.mandatoryCargo[action] > 0:
                #   self.mandatoryCargo[action]-=1
                #  reward+=2


                self.loadedVehicles[self.currentLane][self.vehicleCounter[self.currentLane]] = action
                self.vehicleCounter[self.currentLane]+= 1



                for i in range(self.vehicleData[3][action]):
                    self.grid.T[self.currentLane][slot + i] = self.sequence_no
                    self.gridDestination.T[self.currentLane][slot + i] = self.vehicleData[1][action]
                    self.gridVehicleType.T[self.currentLane][slot+i] = self.vehicleData[0][action]

                self.frontier = self._getFrontier()
                self.sequence_no += 1
                self.currentLane = self._getMinimalLanes()[0]  # better name updateCurrentLane

                # if we put a car we reset the TerminalStateCounter
                self.TerminalStateCounter = 0

            self.possibleActions = self.possibleActionsOfState()

            if self._isTerminalState():
                #Space Utilisation
                #reward += self.rewardSystem[2] * np.sum(-self.endOfLanes + np.ones(self.lanes) * self.rows)
                freeSpaces = np.sum(-self.endOfLanes + np.ones(self.lanes) * self.rows)
                reward += self.rewardSystem[2] * freeSpaces
                #Mandatory Vehicles Loaded?
                #TODO seperate method for this
                #mandatoryVehiclesLeft2Load = self.vehicleData[4][self.mandatoryCargoMask]\
                #                          - self.numberOfVehiclesLoaded[self.mandatoryCargoMask]
                mandatoryVehiclesLeft2Load = np.sum(self.vehicleData[4][self.mandatoryCargoMask] \
                                                    - self.numberOfVehiclesLoaded[self.mandatoryCargoMask])

                #reward += np.sum(mandatoryVehiclesLeft2Load) * self.rewardSystem[3]
                reward += mandatoryVehiclesLeft2Load * self.rewardSystem[3]
            return self._getCurrentState(), reward, self._isTerminalState(), None

    def actionSpaceSample(self):
        return np.random.choice(self.possibleActions)

    def _getNumberOfShifts(self, action):
        destination = self.vehicleData[1][action]

        noVehDest1 = len(np.argwhere(self.gridDestination.T[self.currentLane] == 1))

        if destination == 2 and noVehDest1 != 0:
            return 1
        else:
            return 0

    # Return a action recommendation for the currentLane
    def _heuristic(self):
        points = np.ones(len(self.vehicleData[0]))
        # check which action is possible

        for vehicle in self.vehicleData[0]:
            # 1. Destination
            # points[vehicle] +=
            # 2. Mandatory
            points[vehicle] += self.vehicleData[2][vehicle] * self.numberOfVehiclesLoaded[vehicle] * 10
            # 3. Length
            points[vehicle] += self.vehicleData[3] * 5
        pass

    # TODO

    # TODO Test, make sure this is not envoked during simulations
    def add_CargoType(self, destination, mandatory, length, number):
        assert destination == 1 or destination == 2
        assert mandatory == 0 or mandatory == 1
        assert length > 0 and length < len(self.grid[0]) - 4
        assert number == -1 or number > 0

        typeNo = self.vehicleData[0][-1] + 1
        newCargo = np.array([typeNo, destination, mandatory, length, number])
        self.vehicleData = np.vstack((self.vehicleData.T, newCargo)).T

    def saveStowagePlan(self, path):
        stowagePlan = open(path + "_StowagePlan.txt", 'w')
        stowagePlan.write('Stowage Plan and Loading Sequence \n')
        stowagePlan.write(
            '-----------Loading Sequence---------------------------------------------------------------- \n')
        for row in self.grid:
            # Loading Sequence
            for col in row:
                if col == -1:
                    stowagePlan.write('X \t')
                elif col == 0:
                    stowagePlan.write('- \t')
                else:
                    stowagePlan.write(str(int(col)) + ' \t')
            stowagePlan.write(' \n\n')
        stowagePlan.write('-------Destination-------------------------------------------------------------------- \n')
        for row in self.gridDestination:
            for col in row:
                if col == -1:
                    stowagePlan.write('X \t')
                elif col == 0:
                    stowagePlan.write('- \t')
                else:
                    stowagePlan.write(str(int(col)) + ' \t')
            stowagePlan.write('\n\n')

        # Close the file
        stowagePlan.close()


    def getStowagePlan(self):
        return (self.grid, self.loadedVehicles)