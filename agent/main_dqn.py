from agent.dqn import DQNAgent
import numpy as np
from env.roroDeck import RoRoDeck
from viz.Plotter import Plotter
import logging
from analysis.loggingUnit import LoggingBase


if __name__ == '__main__':
    loggingBase = LoggingBase()
    env = RoRoDeck(True,lanes=8,rows=12)
    input_dims = np.shape(env.reset())[0]
    n_games = 3500
    agent = DQNAgent(gamma=0.999, epsilon=1.0, alpha=0.0005, input_dims=input_dims, n_actions=4, mem_size=10000000, batch_size=64, epsilon_end=0.01, epsilon_dec= 0.99999)

    #agent.load_model()

    scores = []
    eps_history = []

    for i in range(n_games):
        done = False
        score = 0
        observation = env.reset()
        while not done:
            #possible_actions = env.possibleActions
            action = agent.choose_action(observation, env.possible_actions) ## add possible actions here
            state_actions = env.possible_actions
            observation_, reward, done, info = env.step(action)
            new_state_actions = env.possible_actions
            score += reward
            agent.remember(observation, action, reward, observation_, done, state_actions,new_state_actions)
            observation = observation_
            agent.learn()

        eps_history.append(agent.epsilon)
        scores.append(score)

        avg_score = np.mean(scores[max(0,i-100):(i+1)])

        if i % 10 == 0 and i > 0:
            print('episode ', i, 'score %.2f' % score, 'avg. score %.2f' % avg_score)
            agent.save_model()

    #loggingBase = LoggingBase()
    module_path = loggingBase.module_path
    plotter = Plotter(module_path,n_games)
    plotter.plotRewardPlot(scores)
    plotter.plotEPSHistory(np.array(eps_history))


