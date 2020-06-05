import logging
import os
from datetime import datetime


class LoggingBase(object):
    def __init__(self):
        date = str(datetime.utcnow().date().strftime("%Y%m%d"))
        time = str(datetime.now().strftime("%H%M"))

        self.module_path = str(os.path.dirname(os.path.realpath(__file__))) + '\\out\\' + date + '\\' +time +'\\'
        os.makedirs(self.module_path, exist_ok=True)

        self.module_path += time


        logging.basicConfig(filename=self.module_path + '_log.log', level=logging.INFO,format="%(asctime)s - %(levelname)s - %(message)s")
        handler = logging.StreamHandler()
        handler.setLevel(logging.INFO)
        #formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")


        #self.logger1 = logging.getLogger('log1')


        #self.logger1.addHandler(logging.FileHandler(self.module_path + '_debugger.log'))

        logging.getLogger('log1').info("Initialise Logger to %s", self.module_path)
