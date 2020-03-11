from env.RoRoDeck import RoRoDeck
from agent.TDQLearning import TDQLearning
from viz.Plotter import Plotter
import os
from datetime import datetime
import pandas as pd
import logging
import matplotlib.pyplot as plt
import seaborn as sns


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

it = 10000
logging.getLogger('log1').info("Train for " + str(it) + " iterations.")

smoothing_window = int(it / 100)
smoothing_window =200

env = RoRoDeck(True)
env.render()
# Training
agent = TDQLearning(env,module_path,it)
q_table, totalRewards, stateExpantion, stepsToExit = agent.train()

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
plotter.plot(totalRewards, stateExpantion, stepsToExit)


logging.getLogger('log1').info("SHUTDOWN")
