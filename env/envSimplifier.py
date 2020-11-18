import numpy as np
from algorithms.algorithms import find_row_width
import logging
#from env.roroDeck import RoRoDeck


class EnvSimplifierConsistencyChecker(object):
    def __init__(self, env):
        self.env = env

    def simplify_vehicle_length(self):
        self.env.rows, self.env.vehicle_data[4] = find_row_width(self.env.rows, self.env.vehicle_data[4])

    def check_input_consistency(self):
        #Check vehicle dimensions
        logging.getLogger(__name__).info("Check input dimensions...")
        if self.env.rows < self.env.lanes:
            logging.getLogger(__name__).warning("Unusual deck dimensions: More lanes than rows ... might not be fatal thus ignore it")

        if not isinstance(self.env.rows, (np.integer, int)):
            error_msg = 'Row argument was of type {} but must be an integer'.format(type(self.env.rows))
            logging.getLogger(__name__).error(error_msg)
            raise TypeError(error_msg)

        if not self.env.rows > 0:
            error_msg = 'Row argument should be positive but was {} '.format(self.env.rows)
            logging.getLogger(__name__).error(error_msg)
            raise TypeError(error_msg)


        if not isinstance(self.env.lanes, (np.integer, int)):
            error_msg = 'Lane argument was of type {} but must be an integer'.format(type(self.env.lanes))
            logging.getLogger(__name__).error(error_msg)
            raise TypeError(error_msg)

        if not self.env.lanes > 0:
            error_msg = 'Lanes argument should be positive but was {} '.format(self.env.lanes)
            logging.getLogger(__name__).error(error_msg)
            raise TypeError(error_msg)

        if 2*self.env.hull_catheti_length > self.env.lanes:
            error_msg = 'Too much hull_width (was {}): Double hull_catheti_length ({}) is more than Lanes ({})' \
                        '-> Should be less or equal'.format(self.env.hull_catheti_length,self.env.hull_catheti_length*2,self.env.lanes)
            logging.getLogger(__name__).error(error_msg)
            raise TypeError(error_msg)

        if 2*self.env.hull_catheti_length > self.env.rows:
            error_msg = 'Too much hull_catheti_length (was {}): Double hull_catheti_length ({}) is more than Rows ({})' \
                        '-> Should be less or equal'.format(self.env.hull_catheti_length,self.env.hull_catheti_length*2,self.env.lanes)
            logging.getLogger(__name__).error(error_msg)
            raise TypeError(error_msg)


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
        #TODO write out input data

