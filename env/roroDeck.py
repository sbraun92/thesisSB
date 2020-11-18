import numpy as np
import logging
from env.envChecker import RoRoDeckConsistencyChecker

np.random.seed(0)


class RoRoDeck(object):
    def __init__(self, lanes=8, rows=12, hull_catheti_length=4, vehicle_data=None, reward_system=None,
                 stochastic=False, p=0.98, zeta=0.001, reward_illegal_action=-50):
        """
        Initialise RoRo-Deck environment
        Args:
            lanes(int):                     number of lanes
            rows(int):                      number of rows
            hull_catheti_length(int):       size of ship hull triangle
            vehicle_data(np.array):         loading list
            reward_system(np.array):        weight vector for reward function
            stochastic(bool):               switch for environment stochasticity
            p(float):                       degree of stochasticity (Bernoulli variable)
            zeta(float):                    parameter for reward function
            reward_illegal_action(int):     punishment for taking an illegal action (for reward function)
        """

        logging.getLogger(__name__).info('Initialise RoRo-Deck environment: \tLanes: {}\tRows: {}'.format(lanes, rows))

        # if stochastic equals True the environment will use with probability p the action chosen by the agent and with
        # 1-p a random action from the possible actions
        self.stochastic = stochastic
        self.p = p

        self.loading_sequence = None
        self.lanes = lanes
        self.rows = rows
        self.sequence_no = 1
        self.hull_catheti_length = hull_catheti_length
        self.zeta = zeta
        self.reward_illegal_action = reward_illegal_action
        if reward_system is None:
            self.reward_system = np.array([2, -12, -1, -50])
        else:
            self.reward_system = reward_system

        # Use default loading list "0" if not specified otherwise
        if vehicle_data is None:
            self.vehicle_data = np.array([[0, 1, 2, 3, 4],  # vehicle id
                                          [5, 5, -1, -1, 2],  # number of vehicles on yard
                                          [1, 1, 0, 0, 1],  # mandatory cargo?
                                          [1, 2, 1, 2, 2],  # destination
                                          [2, 3, 2, 3, 2],  # length
                                          [0, 0, 0, 0, 1]])  # reefer cargo?
        else:
            self.vehicle_data = vehicle_data

        # Check if input data is consistent
        RoRoDeckConsistencyChecker(self).check_input_consistency()

        # Log input data
        logging.getLogger(__name__).info('Initialise Reward System with parameters: \n' +
                                         '\t\t\t\t Time step reward: \t\t\t\t\t {}\n'.format(self.reward_system[0]) +
                                         '\t\t\t\t Reward for caused shift:\t\t\t\t{}\n'.format(self.reward_system[1]) +
                                         '\t\t\t\t @Terminal - reward for Space Utilisation: \t\t{}\n'
                                         .format(self.reward_system[2]) +
                                         '\t\t\t\t @Terminal - reward for mandatory cargo not loaded:\t{}'
                                         .format(self.reward_system[3]))
        for vehicle_id in self.vehicle_data[0]:
            logging.getLogger(__name__).info('Vehicle id {}\t'.format(vehicle_id)
                                             + "Destination: {}\t".format(self.vehicle_data[3][vehicle_id])
                                             + "Mandatory?: {}\t".format(bool(self.vehicle_data[2][vehicle_id]))
                                             + "Length: {}\t".format(self.vehicle_data[4][vehicle_id])
                                             + "Number on yard: " + str(
                self.vehicle_data[1][vehicle_id] if self.vehicle_data[1][vehicle_id] != -1 else "inf")
                                             + "\tReefer?: {}\t".format(bool(self.vehicle_data[5][vehicle_id])))

        # Initialise various internal helper structures:
        self.grid = self._create_grid()
        self.grid_reefer = self._create_grid()
        self.grid_reefer.T[0][4:rows] = 1
        self.grid_destination = self.grid.copy()
        self.grid_vehicle_type = self.grid.copy() - 1
        self.end_of_lanes = self._get_end_of_lane()
        self.initial_end_of_lanes = self.end_of_lanes.copy()
        self.current_Lane = self._get_minimal_lanes()[0]
        self.mandatory_cargo_mask = self.vehicle_data[2] == 1
        self.loaded_Vehicles = -np.ones((self.lanes, self.rows), dtype=np.int)
        self.vehicle_Counter = np.zeros(self.lanes, dtype=np.int)
        self.total_capacity = self._get_free_capacity()
        self.number_of_vehicles_loaded = np.zeros(len(self.vehicle_data[0]), dtype=np.int)
        self.minimal_package = np.min(self.vehicle_data[4])
        self.maximal_package = np.max(self.vehicle_data[4])
        self.maximal_shifts = self._maximal_number_of_shifts()
        self.action_space = self.vehicle_data[0]

        self.possible_actions = self.get_possible_actions_of_state()
        self.lowest_destination = np.ones(self.lanes) * np.max(self.vehicle_data[3])
        self.current_state = self._get_current_state()

        logging.getLogger(__name__).info("Environment initialised...")

    def reset(self):
        """
        Reset environment

        Returns:
            initial state

        Raises:
            KeyError: Raises an exception.
        """
        self.loading_sequence = "Loading Sequence of RORO-Deck (Lanes: {} Rows: {})\n\n".format(self.lanes, self.rows)

        # Reset various internal helper structures
        self.sequence_no = 1
        self.grid = self._create_grid()
        self.grid_destination = self.grid.copy()
        self.grid_vehicle_type = self.grid.copy() - 1
        self.grid_reefer = self.grid.copy()
        self.grid_reefer.T[0][4:self.rows] = 1
        self.action_space = self.vehicle_data[0]
        self.end_of_lanes = self._get_end_of_lane()
        self.initial_end_of_lanes = self.end_of_lanes.copy()
        self.number_of_vehicles_loaded = np.zeros(len(self.vehicle_data[0]), dtype=np.int)
        self.total_capacity = self._get_free_capacity()
        self.current_Lane = self._get_minimal_lanes()[0]
        self.possible_actions = self.get_possible_actions_of_state()
        self.loaded_Vehicles = -np.ones((self.lanes, self.rows), dtype=np.int)
        self.vehicle_Counter = np.zeros(self.lanes, dtype=np.int)
        self.lowest_destination = np.ones(self.lanes) * np.max(self.vehicle_data[3])
        self.mandatory_cargo_mask = self.vehicle_data[2] == 1
        self.current_state = self._get_current_state()
        self.minimal_package = np.min(self.vehicle_data[4])
        self.maximal_package = np.max(self.vehicle_data[4])
        self.maximal_shifts = self._maximal_number_of_shifts()

        return self.current_state

    def render(self):
        """Print a representation of an environment (grids of loading sequence, vehicle type and destination)"""
        print(self._get_grid_representations())

    def step(self, action):
        """
        Do one simulation time-step, i.e. one loading maneuver
            1. Check if action was legal
            2. Update data structures (grid, end_of_lanes, sequence number...)
            3. Calculate new state representations

        Args:
            action(int):        Cargo type id (an element of the first row of vehicle_data)

        Returns:
            next state(np.array):   state representation after one simulation step
            reward(float):          reward of taking this action
            is_Terminal(bool):      true if the ship is fully loaded or there is no possible action left to take
            info(dict):             additional info for debugging - not implemented in this version
        """

        if not self._is_action_legal(action):
            return self.current_state, self.reward_illegal_action, self._is_terminal_state(), None
        else:
            # Change action passed if environment should behave random
            if self.stochastic:
                if not np.random.choice([True, False], 1, p=[self.p, 1 - self.p]):
                    action = np.random.choice(self.possible_actions)

            # Needed for reward calculation (must be done before updating data structures)
            number_of_shifts = self._get_number_of_shifts(action)
            is_cargo_mandatory = int(self.vehicle_data[2][action] == 1)

            slot = self.end_of_lanes[self.current_Lane]
            self.loading_sequence += "{}. Load Vehicle Type \t {} \t in Lane: \t {} \t Row: \t {} \n" \
                .format(self.sequence_no, action, self.current_Lane, slot)

            self.end_of_lanes[self.current_Lane] += self.vehicle_data[4][action]

            if self.vehicle_data[1][action] == -1 or \
                    self.number_of_vehicles_loaded[action] < self.vehicle_data[1][action]:
                self.number_of_vehicles_loaded[action] += 1

            self.loaded_Vehicles[self.current_Lane][self.vehicle_Counter[self.current_Lane]] = action
            self.vehicle_Counter[self.current_Lane] += 1

            # Update grids
            for i in range(self.vehicle_data[4][action]):
                self.grid.T[self.current_Lane][slot + i] = self.sequence_no
                self.grid_destination.T[self.current_Lane][slot + i] = self.vehicle_data[3][action]
                self.grid_vehicle_type.T[self.current_Lane][slot + i] = self.vehicle_data[0][action]

            # Update lowest destination data structure
            if self.vehicle_data[3][action] < self.lowest_destination[self.current_Lane]:
                self.lowest_destination[self.current_Lane] = self.vehicle_data[3][action]

            self.sequence_no += 1
            # Update according to lane selection heuristic
            self.current_Lane = self._get_minimal_lanes()[0]

            self.possible_actions = self.get_possible_actions_of_state()
            self.current_state = self._get_current_state()

            if self._is_terminal_state():
                # Calculate reward for terminal state
                free_spaces = np.sum(self._get_free_capacity()) / np.sum(self.total_capacity)
                mandatory_vehicles_left_to_load = np.sum(self.vehicle_data[1][self.mandatory_cargo_mask]
                                                         - self.number_of_vehicles_loaded[self.mandatory_cargo_mask])
                reward_features = np.array(
                    [is_cargo_mandatory, number_of_shifts, free_spaces, mandatory_vehicles_left_to_load])
                reward = np.dot(self.reward_system, reward_features) + self.zeta

                return self.current_state, reward, True, {}
            else:
                # Calculate reward
                reward_features = np.array([is_cargo_mandatory, number_of_shifts, 0, 0])
                reward = np.dot(self.reward_system, reward_features) + self.zeta

                return self.current_state, reward, False, {}

    def action_space_sample(self):
        """ Samples a random action"""
        return np.random.choice(self.possible_actions)

    def get_possible_actions_of_state(self):
        """Determine which cargo types are legal to load (returns e.g. [0,2]: actions 0 and 2 are legal)"""
        possible_actions = []
        for action in self.vehicle_data[0]:
            if self._is_action_legal(action):
                possible_actions += [action]
        return np.array(possible_actions)

    def save_stowage_plan(self, path):
        """Save stowage plan and loading sequence"""
        with open(path + "_StowagePlan.txt", 'w') as stowage_plan:
            stowage_plan.write('Stowage Plan and Loading Sequence \n')
            stowage_plan.write(self._get_grid_representations())

        # Write Loading Sequence
        with open(path + "_LoadingSequence.txt", 'w') as loading_seq:
            loading_seq.write(self.loading_sequence)

    def get_stowage_plan(self):
        """return stowage plan data which is usable for Evaluator (Human users should use e.g. render())"""
        return self.grid, self.loaded_Vehicles

    def _create_grid(self):
        """
        Creates a grid representation of a RORO deck with vessel hull with
                            0: empty space
                           -1: unusable space
        Returns:
        grid(np.array):     Note: if hull dimension does not fit
        """

        # Check if hull dimensions are sensible for deck-dimensions (rows & lanes)
        grid = np.zeros((self.rows, self.lanes), dtype=np.int)
        if self.rows > self.hull_catheti_length and self.lanes >= self.hull_catheti_length * 2:
            for i in range(self.hull_catheti_length):
                t = (self.hull_catheti_length - i)
                grid[i] += np.hstack([-np.ones(t, dtype=np.int), np.zeros(self.lanes - t, dtype=np.int)])
                grid[i] += np.hstack([np.zeros(self.lanes - t, dtype=np.int), -np.ones(t, dtype=np.int)])
        else:
            logging.getLogger(__name__).error("Ship hull does not match grid dimensions -> return without hull")
        return grid

    def _get_end_of_lane(self):
        """
        Returns the indices of the first free row number of each lane.


        Returns:
        end_of_lanes(np.array):     Note: If a lane is full set it to -1.

        """

        end_of_lanes = np.zeros(len(self.grid.T), dtype=np.int)
        for idx, lane in enumerate(self.grid.T):
            empty_space_in_lanes = np.argwhere(lane != 0)
            if empty_space_in_lanes.size != 0:
                end_of_lanes[idx] = empty_space_in_lanes[-1] + 1

            if self.grid.T[idx][-1] != 0:
                end_of_lanes[idx] = -1
        return end_of_lanes

    def _get_free_capacity(self):
        """
        Return array which indicates how much space is free in each lane.


        Returns:
        capacity(np.array):     capacity per lane
        """

        capacity = np.ones(len(self.grid.T)) * len(self.grid)
        capacity -= np.count_nonzero(self.grid, axis=0)
        return capacity

    def _get_minimal_lanes(self):
        """Find lanes with most capacity left (used for lane selection heuristic)"""
        return np.argwhere(self.end_of_lanes == np.min(self.end_of_lanes)).flatten()

    def _is_action_legal(self, action):
        """ Method to check if an action is legal (returns a boolean)"""
        loading_position = self.end_of_lanes[self.current_Lane]
        length_of_vehicle = self.vehicle_data[4][action]

        # Check if the corresponding lane has sufficient capacity for cargo
        if loading_position + length_of_vehicle <= self.rows:
            # Check if still vehicle are due to be loaded or infinite vehicle are in harbour yard to load
            if self.number_of_vehicles_loaded[action] < self.vehicle_data[1][action] or \
                    self.vehicle_data[1][action] == -1:
                # Check if cargo type is a reefer that it can be placed in chosen position
                if self.vehicle_data[5][action] == 1:
                    designated_loading_area = self.grid_reefer.T[self.current_Lane][
                                              loading_position:(loading_position + length_of_vehicle)]
                    return np.all(designated_loading_area == 1)
                else:
                    return True
            else:
                return False
        else:
            return False

    def _is_vessel_full(self):
        """ Method to determine if the vessel is fully loaded"""
        return np.size(np.where(self.end_of_lanes + (np.ones(self.lanes) * self.minimal_package) <= self.rows)) == 0

    def _is_terminal_state(self):
        """
        Check if a terminal state is reached ie. no possible action left,
        all cargo units are loaded or if the vessel is fully loaded

        Returns:
        boolean:     True if terminal, False else
        """

        if np.min(self.end_of_lanes) + self.minimal_package > self.rows \
                or np.all((self.vehicle_data[1] - self.number_of_vehicles_loaded) == 0) \
                or len(self.possible_actions) == 0:
            return True
        else:
            return False

    def _get_current_state(self):
        """
        Calculates the state representation which is passed to the agent.

        Returns:
        state representation(np.array):     as outlined in thesis
        """

        # One hot encoding of illegal actions
        illegal_actions_one_hot = np.ones(len(self.vehicle_data[0]))
        if len(self.possible_actions) != 0:
            illegal_actions_one_hot[self.possible_actions] = 0

        # Calculate mandatory vehicles left to load
        mandatory_vehicles_left = self.vehicle_data[1] - self.number_of_vehicles_loaded

        return np.hstack((self.end_of_lanes,
                          self.lowest_destination,
                          mandatory_vehicles_left[self.mandatory_cargo_mask],
                          illegal_actions_one_hot,
                          self.current_Lane)).astype(np.int16)

    def _get_number_of_shifts(self, action):
        """
        Calculate how many shifts are caused by action - only implemented for two different destinations

        Args:
            action(int):                action executed in step()

        Returns:
            shifts caused (int):        either 1 or 0
        """

        destination = self.vehicle_data[3][action]

        no_veh_destination1 = len(np.argwhere(self.grid_destination.T[self.current_Lane] == 1))

        if destination == 2 and no_veh_destination1 != 0:
            return 1
        else:
            return 0

    def _get_grid_representations(self):
        """
        Method returns a string representation of the deck

        Returns:
            representation(string):     includes three plots: Loading Sequence, Vehicle Type and Destination
        """

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

    def _maximal_number_of_shifts(self):
        possible_shifts = np.array(
            (self.total_capacity - np.ones(self.lanes) * self.minimal_package) / self.minimal_package,
            dtype=np.int)
        return np.sum(possible_shifts)
