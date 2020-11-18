import numpy as np

from agent.dqn import DQLearningAgent
from agent.sarsa import SARSA
from agent.tdq import TDQLearning
from env.roroDeck import RoRoDeck

np.random.seed(0)


# Test for Q-Learning
def test_TDQagent():
    env = RoRoDeck()
    env.rows = 12
    env.lanes = 12
    env.reset()
    agent = TDQLearning(env, None)

    assert len(agent.q_table.keys()) == 0
    agent.number_of_episodes = 1
    agent.train()

    assert len(agent.q_table.keys()) == 48


def test_max_action_method_tdq():
    env = RoRoDeck()
    env.rows = 12
    env.lanes = 12
    state = env.reset()

    agent = TDQLearning(env, None)

    agent.q_table[state.tobytes()] = np.zeros(4)

    assert np.count_nonzero(agent.q_table[state.tobytes()]) == 0
    agent.q_table[state.tobytes()][2] = 1
    agent.q_table[state.tobytes()][3] = 2

    # assert agent.max_action(agent.q_table, state, None) == 3
    assert agent.max_action(state, env.possible_actions) == 3

    env.possible_actions = np.array([0, 1, 2])

    assert agent.max_action(state, env.possible_actions) == 2


# Test for SARSA

def test_sarsa_agent():
    env = RoRoDeck()
    env.rows = 12
    env.lanes = 12
    env.reset()
    agent = SARSA(env, None)

    assert len(agent.q_table.keys()) == 0
    agent.number_of_episodes = 1
    agent.train()

    assert len(agent.q_table.keys()) == 48


def test_max_action_method_sarsa():
    env = RoRoDeck()
    env.rows = 12
    env.lanes = 12
    state = env.reset()

    agent = SARSA(env, None)

    agent.q_table[state.tobytes()] = np.zeros(4)

    assert np.count_nonzero(agent.q_table[state.tobytes()]) == 0
    agent.q_table[state.tobytes()][2] = 1
    agent.q_table[state.tobytes()][3] = 2

    assert agent.max_action(state, env.possible_actions) == 3

    env.possible_actions = np.array([0, 1, 2])

    assert agent.max_action(state, env.possible_actions) == 2


# Test for DQN
def test_dqn_agent():
    env = RoRoDeck(lanes=8, rows=12)
    n_games = 2
    agent = DQLearningAgent(env=env, module_path=None, gamma=0.999, epsilon=1.0, alpha=0.0005, mem_size=10000000,
                            batch_size=64, epsilon_min=0.01, epsilon_dec=0.99999)

    actions_taken = 0
    for i in range(n_games):
        done = False
        score = 0
        observation = env.reset()
        while not done:
            action = agent.choose_action(observation, env.possible_actions)
            state_actions = env.possible_actions
            observation_, reward, done, info = env.step(action)
            actions_taken += 1
            new_state_actions = env.possible_actions
            score += reward
            agent.remember(observation, action, reward, observation_, done, state_actions, new_state_actions)
            observation = observation_
            agent.learn()

    assert np.shape(np.nonzero(agent.memory.reward_memory))[1] == actions_taken
    assert np.shape(np.nonzero(agent.memory.terminal_memory))[1] == 2
