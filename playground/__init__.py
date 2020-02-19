from env.RoRoDeck import RoRoDeck
from agent.TDQLearning import TDQLearning
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
from datetime import datetime
import logging
date = str(datetime.utcnow().date().strftime("%Y%m%d"))
time = str(datetime.now().strftime("%H%M"))


module_path = str(os.path.dirname(os.path.realpath(__file__)))+'\\out\\'+date+'\\'

os.makedirs(module_path, exist_ok=True)

module_path += time

logging.basicConfig(filename=module_path+'_debugger.log',level=logging.INFO)

it = 3000
smoothing_window = int(it/10)


env = RoRoDeck()
env.render()

agent = TDQLearning(it)
q_table, totalRewards, stateExpantion, stepsToExit = agent.train(env)


logging.info("prepare plots")
print("Rewards Max:")
print(max(totalRewards))

sns.set(style="darkgrid")
#smoothing_window = 200
fig2 = plt.figure(figsize=(10, 5))
rewards_smoothed = pd.Series(totalRewards).rolling(smoothing_window, min_periods=smoothing_window).mean()
ax = sns.lineplot(data=rewards_smoothed, linewidth=2.5, dashes=False,color="blue")
plt.plot(rewards_smoothed)
plt.xlabel("Episode")
plt.ylabel("Episode Reward (Smoothed)")
plt.title("Episode Reward over Time (Smoothed over window size {})".format(smoothing_window))
#plt.show()
plt.savefig(module_path+'_Rewards.png')
plt.close(fig2)


#plt.title("State Expansion")
fi3 = plt.figure(figsize=(10, 5))
ax = sns.lineplot(data=pd.Series(stateExpantion), linewidth=2.5, dashes=False,color="black")
plt.xlabel("Episode")
plt.ylabel("States Explored")
plt.title("State Expansion over time")
#plt.show()
fi3.savefig(module_path+'_StateExpansion.png')

#Plot smoothed Steps to Exit
steps_smoothed = pd.Series(stepsToExit).rolling(int(smoothing_window/2), min_periods=int(smoothing_window/2)).mean()
fi4 = plt.figure(figsize=(10, 5))
ax = sns.lineplot(data=steps_smoothed, linewidth=2.5, dashes=False, color="green")
plt.xlabel("Episode")
plt.ylabel("Steps to Finish (Smoothed)")
plt.title("Steps to Finish over Time (Smoothed over window size {})".format(int(smoothing_window/2)))
fi4.savefig(module_path+'_StepsToFinish.png')
