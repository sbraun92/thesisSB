from env.RoRoDeck import RoRoDeck
from agent.TDQLearning import TDQLearning
import numpy as np
import matplotlib.pyplot as plt
import tqdm
import time

env = RoRoDeck()
agent = TDQLearning(1000)
q_table, totalRewards, stateExpantion, stepsToExit = agent.train(env)

print("Rewards Max:")
print(max(totalRewards))
plt.title("Episode Rewards")
plt.figure(figsize=(50, 30))
plt.plot(totalRewards)

print("State Expansion MAx:")
print(max(stateExpantion))
plt.title("State Expansion")
plt.figure(figsize=(50, 30))
plt.plot(stateExpantion)

print("Steps to Exit:")
print(min(stepsToExit))
plt.title("Steps to Exit")
plt.figure(figsize=(50, 30))
plt.plot(stepsToExit)
plt.show()