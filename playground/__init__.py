from env.roroDeck import RoRoDeck
from agent.TDQLearning import TDQLearning
from viz.Plotter import Plotter
from analysis.loggingUnit import LoggingBase
import os
from datetime import datetime
import pandas as pd
import logging
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
'''
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
'''

#Register Outputpath and Logger
loggingBase = LoggingBase()
module_path = loggingBase.module_path

it = 1000
logging.getLogger('log1').info("Train for " + str(it) + " iterations.")

smoothing_window = int(it / 100)
smoothing_window =200

#Test with a bigger configuration
env = RoRoDeck(True,8,10)

vehicleData = np.array([[0, 1, 2, 3, 4],  # vehicle id
                        [1, 2, 1, 2,2],  # destination
                        [1, 1, 0, 0,1],  # mandatory
                        [2, 3, 2, 3, 5],  # length
                        [7, 7, -1,-1, 2]])  # number of vehicles on yard
                                                      # (-1 denotes there are infinite vehicles of that type)

print(env.vehicleData)
env.render()
# Training
agent = TDQLearning(env,module_path,it)
q_table, totalRewards, stateExpantion, stepsToExit, eps_history = agent.train()

env.render()


'''
ds1 = []
ds2 = []
for i in range(15):
    #old version
    start = datetime.now()

    it = 2000
    logging.getLogger('log1').info("Train for "+str(it)+" iterations.")

    smoothing_window = int(it/100)

    env = RoRoDeck(False)
    #Training
    agent = TDQLearning(env,module_path,it,True)
    q_table, totalRewards, stateExpantion, stepsToExit = agent.train()
    ds1+=[(datetime.now() - start).total_seconds()]

    #Changed version
    start = datetime.now()

    env2 = RoRoDeck(False)
    #Training
    #Try false to check method maxAction
    agent2 = TDQLearning(env,module_path,it,False)
    q_table, totalRewards, stateExpantion, stepsToExit = agent2.train()
    ds2+=[(datetime.now() - start).total_seconds()]

df = pd.DataFrame()
df['orig']= ds1
df['faster'] = ds2
print(df)
sns.lineplot(data=df)
plt.show()
'''
#Plotting
plotter = Plotter(module_path, it)
plotter.plot(totalRewards, stateExpantion, stepsToExit,eps_history)


logging.getLogger('log1').info("SHUTDOWN")
