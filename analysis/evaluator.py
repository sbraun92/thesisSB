import numpy as np
import logging


class Evaluator(object):
    def __init__(self, vehicle_data, deck_layout):
        """Initialise stowage plan evaluator with input data"""

        self.stowage_plan = None
        self.evaluation_criteria = np.zeros(3)
        self.vehicle_data = vehicle_data
        self.mandatory_vehicle = vehicle_data[2] == 1
        self.number_of_vehicles = np.zeros(len(vehicle_data.T))

        # A Layout of deck (empty or loaded)
        self.deck_layout = deck_layout

    def evaluate(self, stowage_plan):
        """
        Evaluate a stowage plan and calculate the evaluation tuple ESP:
                # 1. mandatory Cargo loaded [%]
                # 2. Number of shifts
                # 3. Space utilisation [%]

        Args:
            stowage_plan(tuple of np.arrays):   return value of env.get_stowage_plan

        Returns:
            ESPTuple-object:                    as outlined in thesis
        """

        if not self.is_stowage_plan_compatible(stowage_plan):
            assert False

        self.stowage_plan = stowage_plan

        mandatory_cargo_loaded = self.calculate_mandatory_cargo_loaded()
        shifts = self.calculate_number_of_shifts()
        space_utilisation = self.calculate_space_utilisation()

        return ESPTuple((mandatory_cargo_loaded, shifts, space_utilisation, self.vehicle_data, self.deck_layout))

    def calculate_number_of_shifts(self):
        """Calculates shifts for a given stowage plan"""
        total_shifts = np.zeros(len(self.stowage_plan[1]))
        destinations = np.unique(self.vehicle_data[3].copy())

        # Loop over all lanes
        for lane_ix, lane in enumerate(self.stowage_plan[1]):
            bad_queue = False
            for ix, vehicle in enumerate(lane):
                if lane[ix + 1] == -1:  # reached end of queue
                    break

                destination_first = self.vehicle_data[3][vehicle]
                destination_second = self.vehicle_data[3][lane[ix + 1]]

                if destination_first == destination_second:
                    if destination_first != destinations[0] and bad_queue:
                        total_shifts[lane_ix] += 1

                if destination_first < destination_second:
                    total_shifts[lane_ix] += 1
                    bad_queue = True

                if destination_first > destination_second and bad_queue:
                    bad_queue = False

        return np.sum(total_shifts)

    def calculate_mandatory_cargo_loaded(self):
        """Calculates the share of mandatory cargo loaded"""
        for i in range(len(self.number_of_vehicles)):
            self.number_of_vehicles[i] = len(np.where(self.stowage_plan[1].flatten() == i)[0])

        loaded_mandatory_veh = np.sum(self.number_of_vehicles[self.mandatory_vehicle])
        all_mandatory_vehicle = np.sum(self.vehicle_data[1][self.mandatory_vehicle])

        return float(loaded_mandatory_veh / all_mandatory_vehicle)

    def calculate_space_utilisation(self):
        """calculates the space utilisation in percentage"""
        grid = self.stowage_plan[0]
        grid = grid.flatten()
        capacity = len(grid) - len(grid[grid == -1])
        free_space = len(grid[grid == 0])

        return 1. - (free_space / capacity)

    def is_stowage_plan_compatible(self, stowage_plan):
        """Shallow check if the stowage plan passed has the same dimension as the dimensions initialised"""
        if len(stowage_plan[0]) == len(self.deck_layout) and len(stowage_plan[0].T) == len(self.deck_layout.T):
            return True
        else:
            return False


class ESPTuple(object):
    def __init__(self, evaluation=None):
        """Initialise ESP-tuple as outlined in thesis"""

        self.mandatory_cargo_loaded = evaluation[0]
        self.shifts = evaluation[1]
        self.space_utilisation = evaluation[2]
        self.vehicle_data = evaluation[3]
        self.deck_layout = evaluation[4]

    def __str__(self):
        """String-representation to use with print() of ESP-tuple"""
        info = "\n" + "*" * 80 + "\nEvaluation of Stowage Plan\n" + \
               "Mandatory Cargo Loaded:\t\t {}\n".format(self.mandatory_cargo_loaded) + \
               "Number of Shifts:\t\t {}\n".format(self.shifts) + \
               "Space Utilisation:\t\t {}".format(self.space_utilisation) + \
               "\n" + "*" * 80
        logging.getLogger(__name__).info(info)

        return info

    # Comparision of two stowagePlan evaluations according to decision tree as outlined in thesis

    # Compare two Evaluations:
    #   1. Evaluation with more mandatory Cargo Loaded is always better
    #   2. if Evaluation equal in 1. than is the Stowage Plan with less shifts better
    #   3. if 1. and 2. equal than the Stowage Plan with a higher SpaceUtilisation is better

    def __eq__(self, other):
        """ESP are equivalent"""
        if self._are_plans_comparable(other):
            if self.shifts == other.shifts \
                    and self.space_utilisation == other.space_utilisation \
                    and self.mandatory_cargo_loaded == other.mandatory_cargo_loaded:
                return True
            else:
                return False
        else:
            return False

    def __gt__(self, other):
        """ESP is better than other ESP"""
        if self._are_plans_comparable(other):
            if self.mandatory_cargo_loaded > other.mandatory_cargo_loaded:
                return True
            elif self.mandatory_cargo_loaded < other.mandatory_cargo_loaded:
                return False
            else:
                if self.shifts < other.shifts:
                    return True
                elif self.shifts > other.shifts:
                    return False
                else:
                    if self.space_utilisation > other.space_utilisation:
                        return True
                    else:
                        return False
        else:
            return False

    def __ge__(self, other):
        """ESP is better or equal to other"""
        if self._are_plans_comparable(other):
            if self.__eq__(other) or self.__gt__(other):
                return True
            else:
                return False
        else:
            return False

    def _are_plans_comparable(self, other):
        """Checks if two ESP had the same input data"""
        if np.array_equal(self.vehicle_data, other.vehicle_data) and \
                np.shape(self.deck_layout) == np.shape(other.deck_layout):
            return True
        else:
            logging.getLogger().error("ESP-tuple are not comparable...")
            return False

    def __hash__(self):
        """make ESP hashable"""
        return hash((self.space_utilisation, self.mandatory_cargo_loaded, self.shifts,
                     str(self.vehicle_data), str(self.deck_layout)))
