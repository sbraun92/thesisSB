from env.roroDeck import RoRoDeck
from agent.SARSA import SARSA
from agent.dqn import DQNAgent
from agent.TDQLearning import TDQLearning
from analysis.Plotter import Plotter
from valuation.evaluator import *
from analysis.loggingUnit import LoggingBase
from datetime import datetime
import logging
import numpy as np
import pickle
import tensorflow as tf

if __name__ == '__main__':
    # Register Output path and Logger
    loggingBase = LoggingBase()
    module_path = loggingBase.module_path

    number_of_episodes = 12_00
    logging.getLogger('log1').info("Train for {} iterations.".format(number_of_episodes))

    #smoothing_window = int(number_of_episodes / 100) #TODO delete
    smoothing_window = 200

    # Test with a bigger configuration

    vehicleData = np.array([[0, 1, 2, 3, 4],  # vehicle id
                            [1, 2, 1, 2, 2],  # destination
                            [1, 1, 0, 0, 1],  # mandatory
                            [2, 3, 2, 3, 5],  # length
                            [7, 7, -1, -1, 2]])  # number of vehicles on yard
    # (-1 denotes there are infinite vehicles of that type)
    np.random.seed(0)
    tf.random.set_seed(0)
    env = RoRoDeck(False, 12, 14, stochastic=False)


    #######################################################################
    # TDQ Training
    agent = TDQLearning(env, module_path, number_of_episodes)
    model, total_rewards, vehicle_loaded, eps_history, state_expansion = agent.train()
    agent.save_model(module_path)
    evaluator = Evaluator(env.vehicle_data, env.grid)
    # print(env.current_state)
    evaluation = evaluator.evaluate(env.get_stowage_plan())
    # print(evaluation)
    # Plotting
    plotter = Plotter(module_path, number_of_episodes, algorithm="Time Difference Q Learning", smoothing_window=smoothing_window)
    plotter.plot(total_rewards, state_expansion, vehicle_loaded, eps_history)
    ##########################################################################
    # SARSA Training
    agent = SARSA(env, module_path, number_of_episodes)
    model, total_rewards, vehicle_loaded, eps_history, state_expansion = agent.train()
    agent.save_model(module_path)
    evaluator = Evaluator(env.vehicle_data, env.grid)
    # print(env.current_state)
    evaluation = evaluator.evaluate(env.get_stowage_plan())
    # print(evaluation)
    #pickle.dump(total_rewards, open(module_path + '_rewards.p', "wb"))
    # pickle.dump(steps_to_exit, open(module_path + '_steps_to_exit.p', "wb"))
    #pickle.dump(eps_history, open(module_path + '_eps_history.p', "wb"))
    # Plotting
    plotter = Plotter(module_path, number_of_episodes, algorithm="SARSA", smoothing_window=smoothing_window)
    plotter.plot(total_rewards, state_expansion, vehicle_loaded, eps_history)
    #########################################################################
    number_of_episodes = 14_000
    evaluator = Evaluator(env.vehicle_data, env.grid)
    agent = DQNAgent(env=env, module_path=module_path, gamma=0.999, number_of_episodes=number_of_episodes, epsilon=1.0,
                     alpha=0.0005,
                     mem_size=1_000_000, pretraining_duration=10_000,
                     batch_size=32, epsilon_min=0.01, epsilon_dec=0.99999, layers=[250, 250, 250, 250, 250])
    model, total_rewards, steps_to_exit, eps_history, state_expansion = agent.train()
    print(datetime.now())

    pickle.dump(total_rewards, open(module_path + '_rewards.p', "wb"))
    # pickle.dump(steps_to_exit, open(module_path + '_steps_to_exit.p', "wb"))
    pickle.dump(eps_history, open(module_path + '_eps_history.p', "wb"))

    agent.save_model(module_path)
    print(datetime.now())
    evaluation = evaluator.evaluate(env.get_stowage_plan())
    print(evaluation)
    env.render()
    # Plotting
    plotter = Plotter(module_path, number_of_episodes, algorithm="DQN", smoothing_window=100)
    plotter.plot(total_rewards, state_expansion, steps_to_exit, eps_history)
    #########################################################################
    logging.getLogger('log1').info("SHUTDOWN")
