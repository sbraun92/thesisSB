from env.roroDeck import RoRoDeck
from agent.TDQLearning import TDQLearning
from agent.SARSA import SARSA
from agent.dqn import DQNAgent
import pytest
import numpy as np

np.random.seed(0)
#Test for TDQ
def test_TDQagent():
    env = RoRoDeck(False)
    env.rows = 12
    env.lanes = 12
    env.reset()
    agent = TDQLearning(env,None)

    assert len(agent.q_table.keys()) == 0
    agent.number_of_episodes = 1
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

    assert agent.max_action(agent.q_table, state, None) == 3

    env.possible_actions = np.array([0, 1, 2])

    assert agent.max_action(agent.q_table, state, None) == 2
#Test for SARSA

def test_SARSAagent():
    env = RoRoDeck(False)
    env.rows = 12
    env.lanes = 12
    env.reset()
    agent = SARSA(env, None)

    assert len(agent.q_table.keys()) == 0
    agent.number_of_episodes = 1
    agent.train()

    assert len(agent.q_table.keys()) == 50

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

    assert agent.max_action(state) == 3

    env.possible_actions = np.array([0, 1, 2])

    assert agent.max_action(state) == 2

#Test for DQN
def test_dqnAgent():
    env = RoRoDeck(True, lanes=8, rows=12)
    input_dims = np.shape(env.reset())[0]
    n_games = 2
    agent = DQNAgent(gamma=0.999, epsilon=1.0, alpha=0.0005, input_dims=input_dims, n_actions=5, mem_size=10000000,
                     batch_size=64, epsilon_end=0.01, epsilon_dec=0.99999)

    for i in range(n_games):
        done = False
        score = 0
        observation = env.reset()
        while not done:
            # possible_actions = env.possibleActions
            action = agent.choose_action(observation, env.possible_actions)  ## add possible actions here
            state_actions = env.possible_actions
            observation_, reward, done, info = env.step(action)
            new_state_actions = env.possible_actions
            score += reward
            agent.remember(observation, action, reward, observation_, done, state_actions, new_state_actions)
            observation = observation_
            agent.learn()



    assert np.shape(np.nonzero(agent.memory.reward_memory))[1]==59 #TODO was 59 before remving possible action filtre
    assert np.shape(np.nonzero(agent.memory.terminal_memory))[1]==2

def test_load_save():
    pass

def test_Superclass():
    pass

