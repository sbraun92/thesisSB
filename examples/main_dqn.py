from agent.dqn import DQNAgent
import numpy as np
from env.roroDeck import RoRoDeck
from analysis.Plotter import Plotter
import logging
from analysis.loggingUnit import LoggingBase
from valuation.evaluator import Evaluator

if __name__ == '__main__':
    loggingBase = LoggingBase()
    env = RoRoDeck(False,lanes=8,rows=12,stochastic=False)
#    env.vehicle_Data[4][env.mandatory_cargo_mask]+=4
#    env.vehicle_Data[4][4] = 2 #reefer
    input_dims = np.shape(env.reset())[0]
    n_games = 15000
    agent = DQNAgent(gamma=0.999, epsilon=1.0, alpha=0.0005, input_dims=input_dims, n_actions=5, mem_size=1000000, batch_size=32, epsilon_end=0.01, epsilon_dec= 0.999992)

    #agent.load_model()

    scores = []
    eps_history = []

    for i in range(n_games):
        bad_moves_counter = 0
        done = False
        score = 0
        observation = env.reset()
        while not done:
            #possible_actions = env.possibleActions
            #if i<n_games*(3./4.):
            if i == 10000:
                agent.epsilon = 1
                agent.epsilon_dec = 0.99995
            if i > 10000:
                action = agent.choose_action(observation, env.possible_actions) ## add possible actions here
            else:
                action = agent.choose_action(observation) ## add possible actions here

            state_actions = env.possible_actions
            if action not in state_actions:
                bad_moves_counter +=1
            observation_, reward, done, info = env.step(action)
            new_state_actions = env.possible_actions
            score += reward
            agent.remember(observation, action, reward, observation_, done, state_actions,new_state_actions)
            observation = observation_
            agent.learn()
            if bad_moves_counter >  6:
                break
        eps_history.append(agent.epsilon)
        scores.append(score)

        avg_score = np.mean(scores[max(0,i-100):(i+1)])

        #TODO plot to logger
        logging.getLogger('log1').info('episode ', i, 'score %.2f' % score, 'avg. score %.2f' % avg_score)
        if i % 10 == 0 and i > 0:
            print('episode ', i, 'score %.2f' % score, 'avg. score %.2f' % avg_score)
            agent.save_model(loggingBase.module_path)

    #loggingBase = LoggingBase()
    module_path = loggingBase.module_path
    plotter = Plotter(module_path,n_games)
    plotter.plotRewardPlot(scores)
    plotter.plotEPSHistory(np.array(eps_history))

    #Show the execution mode
    best_score = 0
    state = env.reset()
    done=False
    while not done:
        prediction = agent.q_eval.predict(state[np.newaxis, :])
        pos_action = np.argmax(agent.q_eval.predict(state[np.newaxis, :])[0][env.possible_actions])
        action = env.possible_actions[pos_action]
        state, reward, done, info = env.step(action)
        best_score += reward
    env.render()
    env.saveStowagePlan(module_path)
    agent.save_model(module_path)
    evaluator = Evaluator(env.vehicle_data, env.grid)
    evaluation = evaluator.evaluate(env.getStowagePlan())

    print(evaluation)

