from env.roroDeck import RoRoDeck
from agent.TDQLearning import TDQLearning
import pytest
import numpy as np

np.random.seed(0)

def test_agent():
    env = RoRoDeck(False)
    env.rows = 12
    env.lanes = 12
    env.reset()

    agent = TDQLearning(env,None)

    assert len(agent.q_table.keys()) == 0

    agent.numGames=1

    agent.train()

    #env.numGames = 1

    assert len(agent.q_table.keys()) == 48

