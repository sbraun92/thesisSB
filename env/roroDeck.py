# RORO-Terminal Enviroment based on a GridWorld; is OpenAI Gym complying
import numpy as np
import matplotlib.pyplot as plt
import logging
#import gym
np.random.seed(0)


class RoRoDeck(object):

    """
    Enviroment-Class of a RORO-Deck

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




    def __init__(self, help, lanes=8, rows=10, reward_System = None):
        """
        Initilise environment

        Args:
            lanes: Number of Lanes on RORO-Deck
            rows: Number of rows on RORO-Deck
            vehicle_data: Input data of vehicles (which vehicles have to be loaded and what are their features)
            reward_system: weights to calculate reward of an action
        """
        # just to compare runtimes
        # TODO delete helper from
        self.help = help

        logging.getLogger('log1').info('Initilise RORO-Deck enviroment: Lanes'+str(lanes)+" Rows: "+ str(rows))
        self.lanes = lanes
        self.rows = rows
        self.sequence_no = 1
        self.grid = self._createGrid()
        self.grid_Destination = self._createGrid()
        self.grid_Vehicle_Type = self._createGrid() - 1

        if reward_System == None:
            self.reward_System = np.array([0.2,  #simple Loading
                                           -8,  #caused shifts
                                           -2,  #terminal: Space left unsed
                                           -40])     #terminal: mand. cargo not loaded
        else:
            self.reward_System = reward_System
        logging.getLogger('log1').info('Initilise Reward System with parameters: \n' +
                                        'Time step reward: ' + str(self.reward_System[0]) + "\n" +
                                        'Reward for caused shift: ' + str(self.reward_System[1]) + "\n" +
                                        '@Terminal - reward for Space Utilisation: ' + str(self.reward_System[2]) + "\n" +
                                        '@Terminal - reward for mandatory cargo not loaded: ' + str(self.reward_System[3]) + "\n" + "done...")

        self.end_of_Lanes = self._getEndOfLane(self.grid)
        self.current_Lane = self._getMinimalLanes()[0]

        # Vehicle Data stores vehicle id, destination, if it is mandatory cargo, length and how many to be loaded max
        self.vehicle_Data = np.array([[0, 1, 2, 3],  # vehicle id
                                      [1, 2, 1, 2],  # destination
                                      [1, 1, 0, 0],  # mandatory
                                      [2, 3, 2, 3],  # length
                                      [5, 5, -1,-1]])  # number of vehicles on yard
                                                      # (-1 denotes there are infinite vehicles of that type)

        logging.getLogger('log1').info('Initilise Input Vehicle Data...')
        for vehicleId in self.vehicle_Data[0]:
            logging.getLogger('log1').info('Vehicle id ' + str(vehicleId)
                                           +" Destination: " + str(self.vehicle_Data[1][vehicleId])
                                           + ", is mandatory: " + str(bool(self.vehicle_Data[2][vehicleId]))
                                           +", Length: " + str(self.vehicle_Data[3][vehicleId])
                                           + ", Number on yard: " + str(self.vehicle_Data[4][vehicleId] if self.vehicle_Data[4][vehicleId] != -1 else "inf"))


        self.mandatory_Cargo_Mask = self.vehicle_Data[2] == 1
        #Todo dele np.min(self.vehleData
        self.loaded_Vehicles = -np.ones((self.lanes, self.rows), dtype=np.int16)
        self.vehicle_Counter = np.zeros(self.lanes, dtype=np.int16)


        self.capacity = self._getFreeCapacity(self.grid)
        self.frontier = self._getFrontier()
        self.number_of_vehicles_loaded = np.zeros(len(self.vehicle_Data[0]), dtype=np.int16)
        # for shifts TODO not a good name
        self.shift_helper = self.end_of_Lanes.copy()
        # self.prevVeh = self.endOfLanes.copy()

        # mandatory cargo, must be loaded
        #self.mandatoryCargo = self.vehicleData[4][self.vehicleData[2] == 1]

        # State representation Frontier, BackLook,mandatory cargo, CurrentLane
        self.current_state = self._getCurrentState()

        # Test without switching
        self.actionSpace_names = {0: 'Type1', 1: 'Type2'}
        self.actionSpace = self.vehicle_Data[0]
        self.action2vehicleLength = np.array([2, 3])
        self.action2destination = np.array([1, 2])

        self.minimalPackage = np.min(self.vehicle_Data[3])
        self.possibleActions = self.possibleActionsOfState()
        self.maxSteps = 0
        self.TerminalStateCounter = 0

    def reset(self):
        """
        Reset environment

        Args:
            lanes: Number of Lanes on RORO-Deck
            rows: Number of rows on RORO-Deck
            vehicle_data: Input data of vehicles (which vehicles have to be loaded and what are their features)
            reward_system: weights to calculate reward of an action

        Returns:
            inital state

        Raises:
            KeyError: Raises an exception.
        """
        logging.getLogger('log1').info('Reset Environment')

        self.sequence_no = 1
        self.grid = self._createGrid()
        self.grid_Destination = self._createGrid()
        self.grid_Vehicle_Type = self._createGrid() - 1

        self.end_of_Lanes = self._getEndOfLane(self.grid)
        self.number_of_vehicles_loaded = np.zeros(len(self.vehicle_Data[0]), dtype=np.int16)

        self.capacity = self._getFreeCapacity(self.grid)
        self.current_Lane = self._getMinimalLanes()[0]
        self.frontier = self._getFrontier()
        # self.currentState = np.hstack((self.frontier, self.endOfLanes, self.currentLane))
        self.current_state = self._getCurrentState()
        self.possibleActions = self.possibleActionsOfState()

        self.maxSteps = 0
        self.TerminalStateCounter = 0

        self.shift_helper = self.end_of_Lanes.copy()

        #self.mandatoryCargo = self.vehicleData[4][self.vehicleData[2] == 1]


        self.loaded_Vehicles = -np.ones((self.lanes, self.rows), dtype=np.int16)
        self.vehicle_Counter = np.zeros(self.lanes, dtype=np.int16)


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

        print('-----------VehicleType----------------------------------------------------------------')
        for row in self.grid_Vehicle_Type:
            # Loading Sequence
            for col in row:
                if col == -2:
                    print('X', end='\t')
                elif col == -1:
                    print('-', end='\t')
                else:
                    print(str(int(col)), end='\t')
            print('\n')

        print('-------Destination--------------------------------------------------------------------')
        for row in self.grid_Destination:
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
        return np.max(self.end_of_Lanes)

    def _findCurrentLane(self, endOfLanes):
        return np.argmin(self.end_of_Lanes)

    def _getCurrentLaneAfterPut(self):
        return self._getMinimalLanes()[0]

    def _switchCurrentLane(self):
        minimalLanes = self._getMinimalLanes()
        if minimalLanes.size == 1:
            return minimalLanes[0]
        else:
            position_in_currentLanes = int(np.argwhere(minimalLanes == self.current_Lane))
            newLane = minimalLanes[(position_in_currentLanes + 1) % len(minimalLanes)]
            return newLane

    def _getMinimalLanes(self):
        return np.argwhere(self.end_of_Lanes == np.min(self.end_of_Lanes)).flatten()

    def _isActionLegal(self, action):
        if self.end_of_Lanes[self.current_Lane] + self.vehicle_Data[3][action] <= self.rows:
            if self.vehicle_Data[4][action] == -1: # infinite Vehicles in parkinglot
                return True
            elif self.number_of_vehicles_loaded[action]< self.vehicle_Data[4][action]: #enough vehicles in parking lot
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
        return np.size(np.where(self.end_of_Lanes + (np.ones(self.lanes) * self.minimalPackage) <= self.rows)) == 0

    def _isTerminalState(self):
        # Check if the smallest Element still fits after the frontier element and
        # if there are still vehicles in the parking lot to be loaded
        if self.frontier + self.minimalPackage < self.rows and np.size(self.possibleActions) != 0:
            return False

        if (self._isVesselFull() or \
                 np.all((self.vehicle_Data[4] - self.number_of_vehicles_loaded) == 0)) or \
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
        if self.help:
            return np.hstack((self.frontier, self.end_of_Lanes, self.number_of_vehicles_loaded[self.mandatory_Cargo_Mask], self.current_Lane)).astype(np.int32)
        else:
            return np.hstack((self.loaded_Vehicles.flatten(), self.number_of_vehicles_loaded[self.mandatory_Cargo_Mask], self.current_Lane))

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
            reward += numberOfShifts * self.reward_System[1]  # +self.action2vehicleLength[action]*0.6
            # Remove Switching-Option
            if self.actionSpace[action] == -1:
                self.current_Lane = self._switchCurrentLane()
                # reward = -1
                # self.TerminalStateCounter += 1
            else:
                slot = self.end_of_Lanes[self.current_Lane]
                self.end_of_Lanes[self.current_Lane] += self.vehicle_Data[3][action]

               # if self.numberOfVehiclesLoaded[action] > 0 and self.vehicleData[2][action] == 1:
                #    self.numberOfVehiclesLoaded[action] -= 1
                 #   reward += 2


                if self.vehicle_Data[4][action] == -1: #infinite vehicles on car park
                    self.number_of_vehicles_loaded[action] += 1
                    reward += self.reward_System[0]
                elif self.number_of_vehicles_loaded[action] < self.vehicle_Data[4][action]:
                    self.number_of_vehicles_loaded[action] += 1
                    reward += self.reward_System[0]

                # if self.mandatoryCargo[action] > 0:
                #   self.mandatoryCargo[action]-=1
                #  reward+=2


                self.loaded_Vehicles[self.current_Lane][self.vehicle_Counter[self.current_Lane]] = action
                self.vehicle_Counter[self.current_Lane]+= 1



                for i in range(self.vehicle_Data[3][action]):
                    self.grid.T[self.current_Lane][slot + i] = self.sequence_no
                    self.grid_Destination.T[self.current_Lane][slot + i] = self.vehicle_Data[1][action]
                    self.grid_Vehicle_Type.T[self.current_Lane][slot + i] = self.vehicle_Data[0][action]

                self.frontier = self._getFrontier()
                self.sequence_no += 1
                self.current_Lane = self._getMinimalLanes()[0]  # better name updateCurrentLane

                # if we put a car we reset the TerminalStateCounter
                self.TerminalStateCounter = 0

            self.possibleActions = self.possibleActionsOfState()

            if self._isTerminalState():
                #Space Utilisation
                #reward += self.rewardSystem[2] * np.sum(-self.endOfLanes + np.ones(self.lanes) * self.rows)
                freeSpaces = np.sum(-self.end_of_Lanes + np.ones(self.lanes) * self.rows)
                reward += self.reward_System[2] * freeSpaces
                #Mandatory Vehicles Loaded?
                #TODO seperate method for this
                #mandatoryVehiclesLeft2Load = self.vehicleData[4][self.mandatoryCargoMask]\
                #                          - self.numberOfVehiclesLoaded[self.mandatoryCargoMask]
                mandatoryVehiclesLeft2Load = np.sum(self.vehicle_Data[4][self.mandatory_Cargo_Mask] \
                                                    - self.number_of_vehicles_loaded[self.mandatory_Cargo_Mask])

                #reward += np.sum(mandatoryVehiclesLeft2Load) * self.rewardSystem[3]
                reward += mandatoryVehiclesLeft2Load * self.reward_System[3]
            return self._getCurrentState(), reward, self._isTerminalState(), None

    def actionSpaceSample(self):
        return np.random.choice(self.possibleActions)

    def _getNumberOfShifts(self, action):
        destination = self.vehicle_Data[1][action]

        noVehDest1 = len(np.argwhere(self.grid_Destination.T[self.current_Lane] == 1))

        if destination == 2 and noVehDest1 != 0:
            return 1
        else:
            return 0

    # Return a action recommendation for the currentLane
    def _heuristic(self):
        points = np.ones(len(self.vehicle_Data[0]))
        # check which action is possible

        for vehicle in self.vehicle_Data[0]:
            # 1. Destination
            # points[vehicle] +=
            # 2. Mandatory
            points[vehicle] += self.vehicle_Data[2][vehicle] * self.number_of_vehicles_loaded[vehicle] * 10
            # 3. Length
            points[vehicle] += self.vehicle_Data[3] * 5
        pass

    # TODO

    # TODO Test, make sure this is not envoked during simulations
    def add_CargoType(self, destination, mandatory, length, number):
        assert destination == 1 or destination == 2
        assert mandatory == 0 or mandatory == 1
        assert length > 0 and length < len(self.grid[0]) - 4
        assert number == -1 or number > 0

        typeNo = self.vehicle_Data[0][-1] + 1
        newCargo = np.array([typeNo, destination, mandatory, length, number])
        self.vehicle_Data = np.vstack((self.vehicle_Data.T, newCargo)).T

    def saveStowagePlan(self, path):
        stowagePlan = open(path + "_StowagePlan.txt", 'w')
        stowagePlan.write('Stowage Plan and Loading Sequence \n')
        stowagePlan.write('-------Vehicle Type-------------------------------------------------------------------- \n')
        for row in self.grid_Vehicle_Type:
            for col in row:
                if col == -1:
                    stowagePlan.write('X \t')
                elif col == 0:
                    stowagePlan.write('- \t')
                else:
                    stowagePlan.write(str(int(col)) + ' \t')
            stowagePlan.write('\n\n')
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
        for row in self.grid_Destination:
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
        return (self.grid, self.loaded_Vehicles)