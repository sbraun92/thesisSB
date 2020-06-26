import numpy as np
from algorithms.Algorithms import find_row_width
import logging
#from env.roroDeck import RoRoDeck


class EnvSimplifierConsistencyChecker(object):
    def __init__(self, env):
        self.env = env

    def simplify_vehicle_length(self):
        self.env.rows, self.env.vehicle_data[3] = find_row_width(self.env.rows, self.env.vehicle_data[3])

    def check_input_consistency(self):
        #Check vehicle dimensions
        if self.env.rows < self.env.lanes:
            logging.getLogger(__name__).warning("Unusual deck dimensions: More lanes than rows ... might not be fatal thus ignore it")


        # Check the consistency of vehicle_data
        logging.getLogger(__name__).info("Check format of vehicle_data...")
        if not isinstance(self.env.vehicle_data, np.ndarray):
            error_msg = 'vehicle_data argument was of type {} but must be a numpy array'.format(type(self.env.vehicle_data))
            logging.getLogger(__name__).error(error_msg)
            raise TypeError(error_msg)

        if not all(isinstance(x, np.integer) for x in self.env.vehicle_data.flatten()):
            error_msg = 'Not all elements of vehicle_data have been of type np.integer - Please ensure by using e.g. dtype=np.int'
            logging.getLogger(__name__).error(error_msg)
            raise TypeError(error_msg)

        if len(self.env.vehicle_data[0]) != len(np.unique(self.env.vehicle_data[0])):
            error_msg = 'The first row of vehicle_data represents the action id - therefore each element should be unique'
            logging.getLogger(__name__).error(error_msg)
            raise ValueError(error_msg)


        #TODO each destination> 0, quatity bounds