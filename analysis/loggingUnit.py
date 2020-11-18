import logging
import os
from datetime import datetime


class LoggingBase(object):
    def __init__(self, prepare_training = True):
        """
        Initialise a Logging Unit and set an output path. Thereby create folder system and id each run by date and time

        """
        date = str(datetime.utcnow().date().strftime("%Y%m%d"))
        time = str(datetime.now().strftime("%H%M"))

        self.module_path = str(os.path.dirname(os.path.realpath(__file__))) + '\\out\\'

        if prepare_training:
            self.module_path += date + '\\' + time + '\\'
            os.makedirs(self.module_path, exist_ok=True)

            self.module_path += time

            logging.basicConfig(filename=self.module_path + '_log.log', level=logging.INFO,
                                format="%(asctime)s - %(levelname)s - %(message)s")

            logging.getLogger(__name__).info("Initialised Logger to {}".format(self.module_path))
