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

        logging.basicConfig(filename=self.module_path + '_log.log', level=logging.INFO)
        handler = logging.StreamHandler()
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter("%(levelname)s - %(message)s")
        handler.setFormatter(formatter)

        self.logger1 = logging.getLogger('log1')
        self.logger1.addHandler(logging.FileHandler(self.module_path + '_debugger.log'))

        #Output logging activities to Console
        # logger1.addHandler(handler)

        self.logger2 = logging.getLogger('log2')
        self.logger2.addHandler(logging.FileHandler(self.module_path + '_FinalLoadingSequence.log'))
        # logging.basicConfig(filename=module_path+'_debugger.log',level=logging.INFO)
        # log2 = logging.basicConfig(filename=module_path+'_LoadingSequence.log',level=logging.INFO)