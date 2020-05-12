# RORO-Terminal Enviroment based on a GridWorld; is OpenAI Gym complying
import numpy as np
import logging
import gym

np.random.seed(0)


class RoRoDeck(gym.Env):
    """
    Environment-Class of a RORO-Deck

    Methods:
        reset()
            Reset method
        step()
            does one simulation ste
        render()
            Representation of the current state of the RORO-Deck
        actionSpaceSample()
            returns a random possible action
        possibleActionsOfState()
            returns all possible actions of the current state



    """

    def __init__(self, help=False, lanes=8, rows=10, vehicle_data=None, reward_system=None, stochastic=False):
        """
        Initialise environment

        Args:
            lanes:          Number of Lanes on RORO-Deck
            rows:           Number of rows on RORO-Deck
            vehicle_data:   Input data of vehicles (which vehicles have to be loaded and what are their features)
            reward_system:  rewards for given predefined conditions of a state - which will be cumulated
                            at each simulation step
        """
        logging.getLogger('log1').info('Initialise RORO-Deck environment: \tLanes: {}\tRows: {}'.format(lanes, rows))

        # just to compare runtime
        # TODO delete helper from
        # if stochastic is True the environment will use with probability p the action chosen by the agent and with
        # 1-p a random action from the possible actions
        self.stochastic = stochastic
        self.p = 0.95
        #TODO remove help
        self.help = help

        self.loading_sequence = None

        self.lanes = lanes
        self.rows = rows
        self.sequence_no = 1
        self.grid = self._create_grid()
        # Reefer TODo
        self.grid_reefer = self._create_grid()
        self.grid_reefer.T[0][4:rows] = 1
        self.grid_destination = self._create_grid()
        self.grid_vehicle_type = self._create_grid() - 1
        self.end_of_lanes = self._get_end_of_lane(self.grid)
        self.initial_end_of_lanes = self.end_of_lanes.copy()
        self.current_Lane = self._get_minimal_lanes()[0]

        if reward_system is None:
            self.reward_system = np.array([0.2,  # simple Loading
                                           -12,  # caused shifts was -8
                                           -2,  # terminal: Space left unused
                                           -50])  # terminal: mand. cargo not loaded   \was 40
        else:
            self.reward_system = reward_system
        # TODO string formatting

        # Vehicle Data stores vehicle id, destination, if it is mandatory cargo, length and how many to be loaded max
        if vehicle_data is None:
            self.vehicle_data = np.array([[0, 1, 2, 3, 4],  # vehicle id
                                          [1, 2, 1, 2, 2],  # destination
                                          [1, 1, 0, 0, 1],  # mandatory
                                          [2, 3, 2, 3, 2],  # length
                                          [5, 5, -1, -1, 2],  # number of vehicles on yard (-1 denotes there are
                                          # infinite vehicles of that type)
                                          [0, 0, 0, 0, 1]])  # Reefer

        self.mandatory_cargo_mask = self.vehicle_data[2] == 1
        # Todo dele np.min(self.vehleData
        self.loaded_Vehicles = -np.ones((self.lanes, self.rows), dtype=np.int16)
        self.vehicle_Counter = np.zeros(self.lanes, dtype=np.int16)
        self.capacity = self._get_free_capacity()
        # TODO Delete Frontier as it is redundant information
        self.frontier = self._get_frontier()
        self.number_of_vehicles_loaded = np.zeros(len(self.vehicle_data[0]), dtype=np.int16)
        # for shifts TODO not a good name
        self.shift_helper = self.end_of_lanes.copy()
        self.minimal_package = np.min(self.vehicle_data[3])
        self.maximal_package = np.max(self.vehicle_data[3])
        # mandatory cargo, must be loaded
        # self.mandatoryCargo = self.vehicleData[4][self.vehicleData[2] == 1]

        # Test without switching
        self.action_space = self.vehicle_data[0]
        self.possible_actions = self.get_possible_actions_of_state()
        self.TerminalStateCounter = 0
        self.lowest_destination = np.ones(self.lanes) * 8  # TODO
        # self.maximal_shifts = np.sum(self.vehicle_Data[1][np.where(self.vehicle_Data[1]>1)])
        # print(self.maximal_shifts)
        # State representation Frontier, BackLook,mandatory cargo, CurrentLane
        self.current_state = self._get_current_state()

        logging.getLogger('log1').info('Initialise Reward System with parameters: \n' +
                                       '\t\t\t\t Time step reward: \t\t\t\t\t {}\n'.format(self.reward_system[0]) +
                                       '\t\t\t\t Reward for caused shift:\t\t\t\t{}\n'.format(self.reward_system[1]) +
                                       '\t\t\t\t @Terminal - reward for Space Utilisation: \t\t{}\n'
                                       .format(self.reward_system[2]) +
                                       '\t\t\t\t @Terminal - reward for mandatory cargo not loaded:\t{}'
                                       .format(self.reward_system[3]))
        for vehicle_id in self.vehicle_data[0]:
            logging.getLogger('log1').info('Vehicle id {}\t'.format(vehicle_id)
                                           + "Destination: {}\t".format(self.vehicle_data[1][vehicle_id])
                                           + "Mandatory?: {}\t".format(bool(self.vehicle_data[2][vehicle_id]))
                                           + "Length: {}\t".format(self.vehicle_data[3][vehicle_id])
                                           + "Number on yard: " + str(
                self.vehicle_data[4][vehicle_id] if self.vehicle_data[4][vehicle_id] != -1 else "inf")
                                           + "\tReefer?: {}\t".format(bool(self.vehicle_data[5][vehicle_id])))

        logging.getLogger('log1').info("Environment initialised...")
        # TODO print input data in method

    def reset(self):
        """
        Reset environment

        Returns:
            inital state

        Raises:
            KeyError: Raises an exception.
        """
        self.loading_sequence = "Loading Sequence of RORO-Deck (Lanes: {} Rows: {})\n\n".format(self.lanes, self.rows)

        self.sequence_no = 1
        self.grid = self._create_grid()
        self.grid_destination = self._create_grid()
        self.grid_vehicle_type = self._create_grid() - 1
        self.grid_reefer = self._create_grid()
        self.grid_reefer.T[0][4:(self.rows)] = 1
        self.action_space = self.vehicle_data[0]
        self.end_of_lanes = self._get_end_of_lane(self.grid)
        self.initial_end_of_lanes = self.end_of_lanes.copy()
        self.number_of_vehicles_loaded = np.zeros(len(self.vehicle_data[0]), dtype=np.int16)
        self.capacity = self._get_free_capacity()
        self.current_Lane = self._get_minimal_lanes()[0]
        self.frontier = self._get_frontier()
        self.possible_actions = self.get_possible_actions_of_state()
        self.TerminalStateCounter = 0
        self.shift_helper = self.end_of_lanes.copy()
        self.loaded_Vehicles = -np.ones((self.lanes, self.rows), dtype=np.int16)
        self.vehicle_Counter = np.zeros(self.lanes, dtype=np.int16)
        self.lowest_destination = np.ones(self.lanes) * 8  # TODO
        self.mandatory_cargo_mask = self.vehicle_data[2] == 1
        self.current_state = self._get_current_state()
        self.minimal_package = np.min(self.vehicle_data[3])
        self.maximal_package = np.max(self.vehicle_data[3])

        return self.current_state

    def render(self, mode='human'):
        """
        Print a representation of an environment (grids of loading sequence, vehicle type and destination)
        Returns
        -------
        None
        """
        print(self._get_grid_representations())

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

    # TODO check this method -> does not use "self"
    def _get_end_of_lane(self, grid):
        """
        Returns the indices of the first free row number of each lane.
        If a lane is full set it to -1.

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

    def _get_free_capacity(self):
        """
        Return Array of which indicates how much space is free in each lane.

        Returns
        -------
        numpy array (length: lanes)
        """
        capacity = np.ones(len(self.grid.T)) * len(self.grid)
        capacity -= np.count_nonzero(self.grid, axis=0)
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
            new_lane = minimal_lanes[(position_in_current_lanes + 1) % len(minimal_lanes)]
            return new_lane

    def _get_minimal_lanes(self):
        return np.argwhere(self.end_of_lanes == np.min(self.end_of_lanes)).flatten()

    def _is_action_legal(self, action):
        loading_position = self.end_of_lanes[self.current_Lane]
        length_of_vehicle = self.vehicle_data[3][action]

        if loading_position + length_of_vehicle <= self.rows:
            if self.number_of_vehicles_loaded[action] < self.vehicle_data[4][action] or \
                    self.vehicle_data[4][action] == -1:  # enough or infinite Vehicles in parking lot
                if self.vehicle_data[5][action] == 1:  # check reefer position
                    designated_loading_area = self.grid_reefer.T[self.current_Lane][
                                              loading_position:(loading_position + length_of_vehicle)]
                    return np.all(designated_loading_area == 1)
                else:
                    return True
            else:
                return False
        else:
            return False

    # return an array such as [0,2] - possible lengths ordered
    def get_possible_actions_of_state(self):
        possible_actions = []
        for action in self.vehicle_data[0]:
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

        if (self._is_vessel_full() or np.all((self.vehicle_data[4] - self.number_of_vehicles_loaded) == 0)) or \
                np.size(self.possible_actions) == 0:
            return True
        else:
            return False
        # TODO Check if _isVesselFull method is redundant
        # if self._isVesselFull() or np.size(
        #        self.possibleActions) == 0:  # or (self.TerminalStateCounter > np.size(self.getMinimalLanes()) * 5) or (self.maxSteps > 500):
        #    return True
        # else:
        #    return False

    def _get_current_state(self):
        if self.help:
            return np.hstack((self.frontier, self.end_of_lanes,
                              self.number_of_vehicles_loaded[self.mandatory_cargo_mask], self.current_Lane)).astype(
                np.int32)
        else:
            # if np.max(self.end_of_lanes)>(0.2*self.rows):
            possible_actions_one_hot = np.zeros(len(self.vehicle_data[0]))
            if len(self.possible_actions) != 0:
                possible_actions_one_hot[self.possible_actions] = 1

            return np.hstack((self.end_of_lanes, self.lowest_destination,
                              self.number_of_vehicles_loaded[self.mandatory_cargo_mask], possible_actions_one_hot,
                              self.current_Lane))
            # return np.hstack((self.grid_vehicle_type.flatten(), self.lowest_destination,
            #                 self.number_of_vehicles_loaded[self.mandatory_cargo_mask], self.current_Lane))
            # else:
            #    return np.hstack((self.frontier,np.zeros(self.lanes), self.lowest_destination, self.number_of_vehicles_loaded[self.mandatory_cargo_mask], self.current_Lane))
            # return np.hstack((self.loaded_Vehicles.flatten(), self.number_of_vehicles_loaded[self.mandatory_cargo_mask], self.current_Lane))

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
            # print("Action was not Legal. There is an error in the legal action machine")
            return self.current_state, -50, self._is_terminal_state(), None
        else:

            reward = 0  # self.calculateReward()
            number_of_shifts = self._get_number_of_shifts(action)
            reward += number_of_shifts * self.reward_system[1]  # +self.action2vehicleLength[action]*0.6
            # Remove Switching-Option
            if self.action_space[action] == -1:
                self.current_Lane = self._switch_current_lane()
                # reward = -1
                # self.TerminalStateCounter += 1
            else:
                if self.stochastic:
                    if not np.random.choice([True, False], 1, p=[self.p, 1 - self.p]):
                        action = np.random.choice(self.possible_actions)

                slot = self.end_of_lanes[self.current_Lane]
                self.loading_sequence += "{}. Load Vehicle Type \t {} \t in Lane: \t {} \t Row: \t {} \n" \
                    .format(self.sequence_no, action, self.current_Lane, slot)

                self.end_of_lanes[self.current_Lane] += self.vehicle_data[3][action]

                # if self.numberOfVehiclesLoaded[action] > 0 and self.vehicleData[2][action] == 1:
                #    self.numberOfVehiclesLoaded[action] -= 1
                #   reward += 2

                if self.vehicle_data[4][action] == -1:  # infinite vehicles on car park
                    self.number_of_vehicles_loaded[action] += 1
                    reward += self.reward_system[0]
                elif self.number_of_vehicles_loaded[action] < self.vehicle_data[4][action]:
                    self.number_of_vehicles_loaded[action] += 1
                    reward += self.reward_system[0]

                if self.vehicle_data[2][action] == 1:
                    reward += 0.5

                self.loaded_Vehicles[self.current_Lane][self.vehicle_Counter[self.current_Lane]] = action
                self.vehicle_Counter[self.current_Lane] += 1

                # TODO same_groupfactor muss berechnet werden bevor endof lanes geupdated wird
                # same_group_factor = 0
                # if self.inital_end_of_lanes[self.current_Lane] == self.end_of_lanes[self.current_Lane] or \
                #        self.grid_destination.T[self.current_Lane][slot - 1] == self.vehicle_Data[0][action] or \
                #        self.grid_destination.T[self.current_Lane][slot -1 ] == -2:
                #    same_group_factor += 1

                for i in range(self.vehicle_data[3][action]):
                    self.grid.T[self.current_Lane][slot + i] = self.sequence_no
                    self.grid_destination.T[self.current_Lane][slot + i] = self.vehicle_data[1][action]
                    self.grid_vehicle_type.T[self.current_Lane][slot + i] = self.vehicle_data[0][action]

                    # TODO EXPERIMENTING WITH SAME GROUP FACTOR
                #    if self.current_Lane == 0 or \
                #            self.grid_destination.T[self.current_Lane - 1][slot + i] == self.vehicle_Data[0][action] or \
                #            self.grid_destination.T[self.current_Lane - 1][slot + i] == -2:
                #        same_group_factor += 1. / self.vehicle_Data[3][action]*2

                #    if self.current_Lane == self.lanes - 1 or \
                #            self.grid_destination.T[self.current_Lane + 1][slot + i] == self.vehicle_Data[0][action] or \
                #            self.grid_destination.T[self.current_Lane + 1][slot + i] == -2:
                #        same_group_factor += 1. / self.vehicle_Data[3][action]*2

                # reward += same_group_factor*0.005

                if self.vehicle_data[1][action] < self.lowest_destination[self.current_Lane]:
                    self.lowest_destination[self.current_Lane] = self.vehicle_data[1][action]

                self.frontier = self._get_frontier()
                self.sequence_no += 1
                self.current_Lane = self._get_minimal_lanes()[0]  # better name updateCurrentLane

                # if we put a car we reset the TerminalStateCounter
                self.TerminalStateCounter = 0

            self.possible_actions = self.get_possible_actions_of_state()
            self.current_state = self._get_current_state()
            if self._is_terminal_state():
                # Space Utilisation
                # reward += self.rewardSystem[2] * np.sum(-self.endOfLanes + np.ones(self.lanes) * self.rows)
                free_spaces = np.sum(-self.end_of_lanes + np.ones(self.lanes) * self.rows)
                reward += self.reward_system[2] * free_spaces
                # Mandatory Vehicles Loaded?
                # TODO seperate method for this
                # mandatory_vehicles_left_to_load = self.vehicleData[4][self.mandatoryCargoMask]\
                #                          - self.numberOfVehiclesLoaded[self.mandatoryCargoMask]
                mandatory_vehicles_left_to_load = np.sum(self.vehicle_data[4][self.mandatory_cargo_mask] \
                                                         - self.number_of_vehicles_loaded[self.mandatory_cargo_mask])

                # reward += np.sum(mandatory_vehicles_left_to_load) * self.rewardSystem[3]
                reward += mandatory_vehicles_left_to_load * self.reward_system[3]

                return self.current_state, reward, True, None
            else:
                return self.current_state, reward, False, None

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
        destination = self.vehicle_data[1][action]

        no_veh_dest1 = len(np.argwhere(self.grid_destination.T[self.current_Lane] == 1))

        if destination == 2 and no_veh_dest1 != 0:
            return 1
        else:
            return 0

    def save_stowage_plan(self, path):
        with open(path + "_StowagePlan.txt", 'w') as stowage_plan:
            stowage_plan.write('Stowage Plan and Loading Sequence \n')
            stowage_plan.write(self._get_grid_representations())

        # Write Loading Sequence
        with open(path + "_LoadingSequence.txt", 'w') as loading_seq:
            loading_seq.write(self.loading_sequence)

    def get_stowage_plan(self):
        return self.grid, self.loaded_Vehicles

    def _get_grid_representations(self):
        representation = '-----------Loading Sequence----------------------------------------------------------------\n'
        for row in self.grid:
            for col in row:
                if col == -1:
                    representation += 'X\t'
                elif col == 0:
                    representation += '-\t'
                else:
                    representation += str(int(col)) + '\t'
            representation += '\n\n'

        representation += '-----------VehicleType--------------------------------------------------------------------\n'
        for row in self.grid_vehicle_type:
            for col in row:
                if col == -2:
                    representation += 'X\t'
                elif col == -1:
                    representation += '-\t'
                else:
                    representation += str(int(col)) + '\t'
            representation += '\n\n'

        representation += '-----------Destination--------------------------------------------------------------------\n'
        for row in self.grid_destination:
            for col in row:
                if col == -1:
                    representation += 'X\t'
                elif col == 0:
                    representation += '-\t'
                else:
                    representation += str(int(col)) + '\t'
            representation += '\n\n'

        return representation
