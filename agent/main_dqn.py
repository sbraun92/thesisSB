from agent.dqn import Agent
import numpy as np
from env.roroDeck import RoRoDeck
from viz.Plotter import Plotter
import logging
from analysis.loggingUnit import LoggingBase


if __name__ == '__main__':
    loggingBase = LoggingBase()
    env = RoRoDeck(True,lanes=10,rows=20)
    input_dims = np.shape(env.reset())[0]
    n_games = 8000
    agent = Agent(gamma=0.999, epsilon=1.0, alpha=0.0005, input_dims=input_dims, n_actions=4, mem_size=1000000, batch_size=32, epsilon_end=0.01, epsilon_dec= 0.99997)

    #agent.load_model()

    scores = []
    eps_history = []

    for i in range(n_games):
        done = False
        score = 0
        observation = env.reset()
        while not done:
            #possible_actions = env.possibleActions
            action = agent.choose_action(observation, env.possibleActions) ## add possible actions here
            observation_, reward, done, info = env.step(action)
            score += reward
            agent.remember(observation,action, reward, observation_,done)
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


