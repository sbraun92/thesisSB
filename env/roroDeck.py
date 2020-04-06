# RORO-Terminal Enviroment based on a GridWorld; is OpenAI Gym complying
import numpy as np
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

    def __init__(self, help, lanes=8, rows=10, reward_system=None):
        """
        Initialise environment

        Args:
            lanes: Number of Lanes on RORO-Deck
            rows: Number of rows on RORO-Deck
            vehicle_data: Input data of vehicles (which vehicles have to be loaded and what are their features)
            reward_system: weights to calculate reward of an action
        """
        # just to compare runtime
        # TODO delete helper from
        self.help = help

        logging.getLogger('log1').info('Initilise RORO-Deck enviroment: Lanes'+str(lanes)+" Rows: "+ str(rows))
        self.lanes = lanes
        self.rows = rows
        self.sequence_no = 1
        self.grid = self._create_grid()
        self.grid_destination = self._create_grid()
        self.grid_vehicle_type = self._create_grid() - 1

        if reward_system == None:
            self.reward_system = np.array([0.2,  #simple Loading
                                           -8,  #caused shifts
                                           -2,  #terminal: Space left unsed
                                           -40])     #terminal: mand. cargo not loaded
        else:
            self.reward_system = reward_system
        logging.getLogger('log1').info('Initilise Reward System with parameters: \n' +
                                        'Time step reward: ' + str(self.reward_system[0]) + "\n" +
                                        'Reward for caused shift: ' + str(self.reward_system[1]) + "\n" +
                                        '@Terminal - reward for Space Utilisation: ' + str(self.reward_system[2]) + "\n" +
                                        '@Terminal - reward for mandatory cargo not loaded: ' + str(self.reward_system[3]) + "\n" + "done...")

        self.end_of_lanes = self._get_end_of_lane(self.grid)
        self.current_Lane = self._get_minimal_lanes()[0]

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


        self.mandatory_cargo_mask = self.vehicle_Data[2] == 1
        #Todo dele np.min(self.vehleData
        self.loaded_Vehicles = -np.ones((self.lanes, self.rows), dtype=np.int16)
        self.vehicle_Counter = np.zeros(self.lanes, dtype=np.int16)


        self.capacity = self._get_free_capacity(self.grid)
        self.frontier = self._get_frontier()
        self.number_of_vehicles_loaded = np.zeros(len(self.vehicle_Data[0]), dtype=np.int16)
        # for shifts TODO not a good name
        self.shift_helper = self.end_of_lanes.copy()
        # self.prevVeh = self.endOfLanes.copy()

        # mandatory cargo, must be loaded
        #self.mandatoryCargo = self.vehicleData[4][self.vehicleData[2] == 1]

        # State representation Frontier, BackLook,mandatory cargo, CurrentLane
        self.current_state = self._get_current_state()

        # Test without switching
        self.actionSpace_names = {0: 'Type1', 1: 'Type2'}
        self.actionSpace = self.vehicle_Data[0]
        self.action2vehicleLength = np.array([2, 3])
        self.action2destination = np.array([1, 2])

        self.minimal_package = np.min(self.vehicle_Data[3])
        self.possible_actions = self.possible_actions_of_state()
        self.maxSteps = 0
        self.TerminalStateCounter = 0

    def reset(self):
        """
        Reset environment

        Returns:
            inital state

        Raises:
            KeyError: Raises an exception.
        """
        logging.getLogger('log1').info('Reset Environment')

        self.sequence_no = 1
        self.grid = self._create_grid()
        self.grid_destination = self._create_grid()
        self.grid_vehicle_type = self._create_grid() - 1

        self.end_of_lanes = self._get_end_of_lane(self.grid)
        self.number_of_vehicles_loaded = np.zeros(len(self.vehicle_Data[0]), dtype=np.int16)

        self.capacity = self._get_free_capacity(self.grid)
        self.current_Lane = self._get_minimal_lanes()[0]
        self.frontier = self._get_frontier()
        # self.currentState = np.hstack((self.frontier, self.endOfLanes, self.currentLane))
        self.current_state = self._get_current_state()
        self.possible_actions = self.possible_actions_of_state()

        self.maxSteps = 0
        self.TerminalStateCounter = 0

        self.shift_helper = self.end_of_lanes.copy()

        #self.mandatoryCargo = self.vehicleData[4][self.vehicleData[2] == 1]


        self.loaded_Vehicles = -np.ones((self.lanes, self.rows), dtype=np.int16)
        self.vehicle_Counter = np.zeros(self.lanes, dtype=np.int16)


        return self._get_current_state()

    def render(self):
        """
        Print a representation of an environment (grids of loading sequence, vehicle type and destination)
        Returns
        -------
        None
        """
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
        for row in self.grid_vehicle_type:
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
        for row in self.grid_destination:
            for col in row:
                if col == -1:
                    print('X', end='\t')
                elif col == 0:
                    print('-', end='\t')
                else:
                    print(str(int(col)), end='\t')
            print('\n')

    def actionSpaceSample(self):
        """
        Randomly picks a possible action
        Returns
        -------
        an action
        """
        return np.random.choice(self.possible_actions)

    def _create_grid(self):
        """
        Creates a grid representation of a RORO deck with vessel hull
        0:  empty space
        -1: unusable space

        Returns
        -------
        a numpy array:  size of lanes times rows
        """
        grid = np.zeros((self.rows, self.lanes), dtype=np.int32)
        for i in range(4):
            t = 4 - i
            grid[i] += np.hstack([-np.ones(t, dtype=np.int32), np.zeros(self.lanes - t, dtype=np.int32)])
            grid[i] += np.hstack([np.zeros(self.lanes - t, dtype=np.int32), -np.ones(t, dtype=np.int32)])
        return grid

    # Return an Array with the Indicies of the last free space
    # Find indcies of last free slot in lane (if full set -1)
    def _get_end_of_lane(self, grid):
        """
        Returns the first free row number of each lane

        Parameters
        ----------
        grid:   A grid of loading sequence representation

        Returns
        -------
        numpy array (length: lanes)
        """
        end_of_lanes = np.zeros(len(grid.T), dtype=np.int32)
        for idx, lane in enumerate(grid.T):
            empty_space_in_lanes = np.argwhere(lane != 0)
            if empty_space_in_lanes.size != 0:
                end_of_lanes[idx] = empty_space_in_lanes[-1] + 1

            if grid.T[idx][-1] != 0:
                end_of_lanes[idx] = -1
        return end_of_lanes

    # Return Array of which indicates how much space is free in each lane
    def _get_free_capacity(self, grid):
        """
        get free capacity of each lane
        Parameters
        ----------
        grid:   A grid of loading sequence representation

        Returns
        -------
        numpy array (length: lanes)
        """
        capacity = np.ones(len(grid.T)) * len(grid)
        capacity -= np.count_nonzero(grid, axis=0)
        return capacity

    def _get_frontier(self):
        """
        get the highest row number of the end_of_lanes array
        Returns
        -------

        """
        return np.max(self.end_of_lanes)

    def _find_current_lane(self):
        return np.argmin(self.end_of_lanes)

    def _get_current_lane_after_put(self):
        return self._get_minimal_lanes()[0]

    def _switch_current_lane(self):
        minimal_lanes = self._get_minimal_lanes()
        if minimal_lanes.size == 1:
            return minimal_lanes[0]
        else:
            position_in_current_lanes = int(np.argwhere(minimal_lanes == self.current_Lane))
            new_Lane = minimal_lanes[(position_in_current_lanes + 1) % len(minimal_lanes)]
            return new_Lane

    def _get_minimal_lanes(self):
        return np.argwhere(self.end_of_lanes == np.min(self.end_of_lanes)).flatten()

    def _is_action_legal(self, action):
        if self.end_of_lanes[self.current_Lane] + self.vehicle_Data[3][action] <= self.rows:
            if self.vehicle_Data[4][action] == -1: # infinite Vehicles in parking lot
                return True
            elif self.number_of_vehicles_loaded[action]< self.vehicle_Data[4][action]: #enough vehicles in parking lot
                return True
            else:
                return False
        else:
            return False

    # return an array such as [0,2] - possible lengths ordered
    def possible_actions_of_state(self):
        possible_actions = []
        for action in range(len(self.actionSpace)):
            if self._is_action_legal(action):
                possible_actions += [action]
        return np.array(possible_actions)

    def _is_vessel_full(self):
        return np.size(np.where(self.end_of_lanes + (np.ones(self.lanes) * self.minimal_package) <= self.rows)) == 0

    def _is_terminal_state(self):
        # Check if the smallest Element still fits after the frontier element and
        # if there are still vehicles in the parking lot to be loaded
        if self.frontier + self.minimal_package < self.rows and np.size(self.possible_actions) != 0:
            return False

        if (self._is_vessel_full() or \
            np.all((self.vehicle_Data[4] - self.number_of_vehicles_loaded) == 0)) or \
                np.size(self.possible_actions) == 0:
            return True
        else:
            return False
        #TODO Check if _isVesselFull method is redundant
        #if self._isVesselFull() or np.size(
        #        self.possibleActions) == 0:  # or (self.TerminalStateCounter > np.size(self.getMinimalLanes()) * 5) or (self.maxSteps > 500):
        #    return True
        #else:
        #    return False

    def _get_current_state(self):
        if self.help:
            return np.hstack((self.frontier, self.end_of_lanes, self.number_of_vehicles_loaded[self.mandatory_cargo_mask], self.current_Lane)).astype(np.int32)
        else:
            return np.hstack((self.loaded_Vehicles.flatten(), self.number_of_vehicles_loaded[self.mandatory_cargo_mask], self.current_Lane))

    def step(self, action):
        """
        Must return new State, reward, if it is a TerminalState

        1. Check if action was legal
        2. Calculate reward
        3. Update grid
        4. Update EndOfLane
        5. Update Frontier
        6. Update CurrentLane
        7. Update SequenceNumber

        Parameters
        ----------
        action:    one action

        Returns
        -------
        next state:     the next state after one simulation step
        reward:         reward of taking this action
        is_Terminal:    true if the ship is fully loaded or there is no possible action left to take
        None:           additional info for debugging - not implemented in this version
        """




        # self.maxSteps += 0.1
        if not self._is_action_legal(action):
            print("Action was not Legal. There is an error in the legal action machine")
            return self._get_current_state(), -1, self._is_terminal_state(), None
        else:
            reward = 0  # self.calculateReward()
            number_of_shifts = self._get_number_of_shifts(action)
            reward += number_of_shifts * self.reward_system[1]  # +self.action2vehicleLength[action]*0.6
            # Remove Switching-Option
            if self.actionSpace[action] == -1:
                self.current_Lane = self._switch_current_lane()
                # reward = -1
                # self.TerminalStateCounter += 1
            else:
                slot = self.end_of_lanes[self.current_Lane]
                self.end_of_lanes[self.current_Lane] += self.vehicle_Data[3][action]

               # if self.numberOfVehiclesLoaded[action] > 0 and self.vehicleData[2][action] == 1:
                #    self.numberOfVehiclesLoaded[action] -= 1
                 #   reward += 2


                if self.vehicle_Data[4][action] == -1: #infinite vehicles on car park
                    self.number_of_vehicles_loaded[action] += 1
                    reward += self.reward_system[0]
                elif self.number_of_vehicles_loaded[action] < self.vehicle_Data[4][action]:
                    self.number_of_vehicles_loaded[action] += 1
                    reward += self.reward_system[0]

                # if self.mandatoryCargo[action] > 0:
                #   self.mandatoryCargo[action]-=1
                #  reward+=2


                self.loaded_Vehicles[self.current_Lane][self.vehicle_Counter[self.current_Lane]] = action
                self.vehicle_Counter[self.current_Lane]+= 1



                for i in range(self.vehicle_Data[3][action]):
                    self.grid.T[self.current_Lane][slot + i] = self.sequence_no
                    self.grid_destination.T[self.current_Lane][slot + i] = self.vehicle_Data[1][action]
                    self.grid_vehicle_type.T[self.current_Lane][slot + i] = self.vehicle_Data[0][action]

                self.frontier = self._get_frontier()
                self.sequence_no += 1
                self.current_Lane = self._get_minimal_lanes()[0]  # better name updateCurrentLane

                # if we put a car we reset the TerminalStateCounter
                self.TerminalStateCounter = 0

            self.possible_actions = self.possible_actions_of_state()

            if self._is_terminal_state():
                #Space Utilisation
                #reward += self.rewardSystem[2] * np.sum(-self.endOfLanes + np.ones(self.lanes) * self.rows)
                freeSpaces = np.sum(-self.end_of_lanes + np.ones(self.lanes) * self.rows)
                reward += self.reward_system[2] * freeSpaces
                #Mandatory Vehicles Loaded?
                #TODO seperate method for this
                #mandatoryVehiclesLeft2Load = self.vehicleData[4][self.mandatoryCargoMask]\
                #                          - self.numberOfVehiclesLoaded[self.mandatoryCargoMask]
                mandatoryVehiclesLeft2Load = np.sum(self.vehicle_Data[4][self.mandatory_cargo_mask] \
                                                    - self.number_of_vehicles_loaded[self.mandatory_cargo_mask])

                #reward += np.sum(mandatoryVehiclesLeft2Load) * self.rewardSystem[3]
                reward += mandatoryVehiclesLeft2Load * self.reward_system[3]
            return self._get_current_state(), reward, self._is_terminal_state(), None

    def action_space_sample(self):
        """
        sample a random action

        Returns
        -------

        an action
        """
        return np.random.choice(self.possible_actions)

    def _get_number_of_shifts(self, action):
        """
        Calculate how many shifts are caused by action - only implemented for two different destinations
        Parameters
        ----------
        action:     action taken by agent

        Returns
        -------
        integer:    0 or 1 in the two destination case
        """
        destination = self.vehicle_Data[1][action]

        noVehDest1 = len(np.argwhere(self.grid_destination.T[self.current_Lane] == 1))

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
        for row in self.grid_vehicle_type:
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
        for row in self.grid_destination:
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