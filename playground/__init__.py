from env.RoRoDeck import RoRoDeck
from agent.TDQLearning import TDQLearning
from viz.Plotter import Plotter
import os
from datetime import datetime
import logging
date = str(datetime.utcnow().date().strftime("%Y%m%d"))
time = str(datetime.now().strftime("%H%M"))


module_path = str(os.path.dirname(os.path.realpath(__file__)))+'\\out\\'+date+'\\'
os.makedirs(module_path, exist_ok=True)
module_path += time

logging.basicConfig(filename=module_path+'_log.log',level=logging.INFO)
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(levelname)s - %(message)s")
handler.setFormatter(formatter)


logger1 = logging.getLogger('log1')
logger1.addHandler(logging.FileHandler(module_path+'_debugger.log'))
#logger1.addHandler(handler)

logger2 = logging.getLogger('log2')
logger2.addHandler(logging.FileHandler(module_path+'_FinalLoadingSequence.log'))
#logging.basicConfig(filename=module_path+'_debugger.log',level=logging.INFO)
#log2 = logging.basicConfig(filename=module_path+'_LoadingSequence.log',level=logging.INFO)


it = 3000
logging.getLogger('log1').info("Train for "+str(it)+" iterations.")

smoothing_window = int(it/100)


env = RoRoDeck()

#Training
agent = TDQLearning(it)
q_table, totalRewards, stateExpantion, stepsToExit = agent.train(env)


#Plotting
plotter = Plotter(module_path, it)
plotter.plot(totalRewards, stateExpantion, stepsToExit)


logging.getLogger('log1').info("SHUTDOWN")



