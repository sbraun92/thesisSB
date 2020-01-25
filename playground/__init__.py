from env.RoRoDeck import RoRoDeck
from agent.TDQLearning import TDQLearning
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tqdm
import time

env = RoRoDeck()
agent = TDQLearning(100000)
q_table, totalRewards, stateExpantion, stepsToExit = agent.train(env)

print("Rewards Max:")
print(max(totalRewards))


smoothing_window = 40
fig2 = plt.figure(figsize=(10, 5))
rewards_smoothed = pd.Series(totalRewards).rolling(smoothing_window, min_periods=smoothing_window).mean()
plt.plot(rewards_smoothed)
plt.xlabel("Episode")
plt.ylabel("Episode Reward (Smoothed)")
plt.title("Episode Reward over Time (Smoothed over window size {})".format(smoothing_window))
plt.show()


print("State Expansion MAx:")
print(max(stateExpantion))
#plt.title("State Expansion")
fi3 = plt.figure(figsize=(50, 30))
plt.plot(stateExpantion)
plt.show()

print("Steps to Exit:")
print(min(stepsToExit))
#plt.title("Steps to Exit")
plt.figure(figsize=(50, 30))
plt.plot(stepsToExit)
plt.show()