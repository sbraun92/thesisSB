from env.roroDeck import RoRoDeck
from agent.TDQLearning import TDQLearning
from agent.SARSA import SARSA
import pytest
import numpy as np

np.random.seed(0)

def test_TDQagent():
    env = RoRoDeck(False)
    env.rows = 12
    env.lanes = 12
    env.reset()
    agent = TDQLearning(env,None)

    assert len(agent.q_table.keys()) == 0
    agent.numGames=1
    agent.train()

    assert len(agent.q_table.keys()) == 48




def test_max_action_method():
    env = RoRoDeck(False)
    env.rows = 12
    env.lanes = 12
    state = env.reset()

    agent = TDQLearning(env, None)

    agent.q_table[state.tobytes()] = np.zeros(4)

    assert np.count_nonzero(agent.q_table[state.tobytes()]) == 0
    agent.q_table[state.tobytes()][2] = 1
    agent.q_table[state.tobytes()][3] = 2

    assert agent.maxAction(agent.q_table,state,None) == 3

    env.possibleActions = np.array([0,1,2])

    assert agent.maxAction(agent.q_table, state, None) == 2


def test_SARSAagent():
    env = RoRoDeck(False)
    env.rows = 12
    env.lanes = 12
    env.reset()
    agent = SARSA(env, None)

    assert len(agent.q_table.keys()) == 0
    agent.numGames = 1
    agent.train()

    assert len(agent.q_table.keys()) == 49


def test_max_action_method():
    env = RoRoDeck(False)
    env.rows = 12
    env.lanes = 12
    state = env.reset()

    agent = SARSA(env, None)

    agent.q_table[state.tobytes()] = np.zeros(4)

    assert np.count_nonzero(agent.q_table[state.tobytes()]) == 0
    agent.q_table[state.tobytes()][2] = 1
    agent.q_table[state.tobytes()][3] = 2

    assert agent.maxAction(agent.q_table,state,None) == 3

    env.possibleActions = np.array([0,1,2])

    assert agent.maxAction(agent.q_table, state, None) == 2