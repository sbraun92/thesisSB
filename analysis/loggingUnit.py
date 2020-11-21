import logging
import os
from datetime import datetime
from pathlib import Path


class LoggingBase(object):
    def __init__(self, prepare_training = True):
        """
        Initialise a Logging Unit and set an output path. Thereby create folder system and id each run by date and time

        """
        date = str(datetime.utcnow().date().strftime("%Y%m%d"))
        time = str(datetime.now().strftime("%H%M"))

        self.module_path = Path(str(os.path.dirname(os.path.realpath(__file__))) + '/out/')

        if prepare_training:
            self.module_path = self.module_path.joinpath(Path(date + '/' + time + '/'))
            os.makedirs(Path(self.module_path), exist_ok=True)

            self.module_path = self.module_path.joinpath(time)

            logging.basicConfig(filename=str(self.module_path) + '_log.log', level=logging.INFO,
                                format="%(asctime)s - %(levelname)s - %(message)s")

            logging.getLogger(__name__).info("Initialised Logger to {}".format(str(self.module_path)))