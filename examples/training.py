from env.roroDeck import RoRoDeck
from agent.tdq import TDQLearning
from analysis.plotter import Plotter
from analysis.evaluator import *
from analysis.loggingUnit import LoggingBase
from analysis import algorithms
import logging
import numpy as np

vehicle_data_0 = np.array([[0, 1, 2, 3, 4],  # vehicle id
                         [5, 5, -1, -1, 2],  # number of vehicles on yard
                         [1, 1, 0, 0, 1],  # mandatory
                         [1, 2, 1, 2, 2],  # destination
                         [2, 3, 2, 3, 2],  # length
                         [0, 0, 0, 0, 1]])  # Reefer

vehicle_data_1 =  np.array([[ 0,  1,  2,  3,  4,  5,  6],
                           [ 5,  5, -1, -1,  2,  2,  2],
                           [ 1,  1,  0,  0,  1,  1,  1],
                           [ 1,  2,  1,  2,  2,  1,  2],
                           [ 2,  3,  2,  3,  2,  2,  3],
                           [ 0,  0,  0,  0,  1,  0,  0]])

vehicle_data_2 = np.array([[0, 1, 2, 3, 4],  # vehicle id
                         [5, 5, -1, -1, 2],  # number of vehicles on yard
                         [1, 1, 0, 0, 1],  # mandatory
                         [1, 2, 1, 2, 2],  # destination
                         [3, 4, 2, 3, 2],  # length
                         [0, 0, 0, 0, 1]])  # Reefer


loading_list = [vehicle_data_0]
#loading_list = [vehicle_data_1]
cut_offs = [17]
#loading_list = [vehicle_data_2]

if __name__ == '__main__':
    np.random.seed(0)
    sizes = [(10,14)]
    #sizes = [(10,14),(10,16),(10,18),(12,16),(12,18)]

    #sizes = [(10,12),(10,14),(12,16)]


    for size in sizes:
        for i,ll in enumerate(loading_list):
        # Register Output path and Logger

            loggingBase = LoggingBase()
            module_path = loggingBase.module_path

            #number_of_episodes = 600_000
            number_of_episodes = 7_000
            logging.getLogger(__name__).info('\n'+'*'*80+'\n'+'*'*80+'\n')
            logging.getLogger(__name__).info("Train for {} iterations.".format(number_of_episodes))

            env = RoRoDeck(size[0], size[1], stochastic=False, vehicle_data=ll)
            cut_off = cut_offs[i]

            agent = TDQLearning(env, module_path, number_of_episodes, additional_info='-L_'+str(i+1))
            #agent = DQLearningAgent(env=env, module_path=module_path, gamma=0.999, number_of_episodes=number_of_episodes, epsilon=1.0,
            #                        alpha=0.0005, regularisation=0.001,
             #                       mem_size=1_000_000, pretraining_duration=10_000,
              #                      batch_size=32, epsilon_min=0.01, epsilon_dec=0.9996, performance_threshold=performance_threshold,
               #                     layers=[128,128], additional_info=str(i), illegal_action_threshold=7)

            model, total_rewards, vehicle_loaded, eps_history, state_expansion = agent.train()
            agent.save_model(module_path)
            evaluator = Evaluator(env.vehicle_data, env.grid)
            evaluation = evaluator.evaluate(env.get_stowage_plan())
            print(evaluation)
            metrics, info = algorithms.training_metrics(total_rewards)
            logging.getLogger(__name__).info(info)

        # Plotting
            plotter = Plotter(module_path, number_of_episodes, algorithm="SARSA", show_title=True)
            plotter.plot(total_rewards, state_expansion, vehicle_loaded, eps_history)
            print(info)

    '''
    #smoothing_window = int(number_of_episodes / 100) #TODO delete
    #smoothing_window = 200

    # Test with a bigger configuration

    #vehicleData = np.array([[0, 1, 2, 3, 4, 5, 6, 7],  # vehicle id
    #                        [1, 2, 1, 2, 2, 2, 2, 1],  # destination
    #                        [1, 1, 0, 0, 1, 1, 1, 1],  # mandatory
    #                        [2, 3, 2, 3, 4, 6, 2, 2],  # length
    #                        [7, 7, -1, -1, 2, 1, 1, 1],
    #                        [0, 0, 0, 0, 1, 0, 0, 0]])  # number of vehicles on yard
    # (-1 denotes there are infinite vehicles of that type)
    
    #tf.random.set_seed(0)
   # env = RoRoDeck(False, 14, 28, stochastic=True, vehicle_data = vehicleData)
    #x = RoRoDeck().vehicle_data.astype(np.float)

    #print(x)
    vehicle_data = np.array([[0, 1, 2, 3, 4],  # vehicle id
                                  [1, 2, 1, 2, 2],  # destination
                                  [1, 1, 0, 0, 1],  # mandatory
                                  [4, 8, 4, 8, 4],  # length
                                  [5, 5, -1, -1, 2],  # number of vehicles on yard (-1 denotes there are
                                  # infinite vehicles of that type)
                                  [0, 0, 0, 0, 1]])  # Reefer

    vehicle_data = np.array([[0, 1, 2, 3, 4],           # vehicle id
                                  [5, 5, -1, -1, 2],    # number of vehicles on yard
                                  [1, 1, 0, 0, 1],      # mandatory
                                  [1, 2, 1, 2, 2],      # destination
                                  [2, 3, 2, 3, 2],      # length
                                  [0, 0, 0, 0, 1]])     # Reefer


    #######################################################################
    # TDQ Training
    #agent = TDQLearning(env, module_path, number_of_episodes)
    #model, total_rewards, vehicle_loaded, eps_history, state_expansion = agent.train()
    #agent.save_model(module_path)
    #evaluator = Evaluator(env.vehicle_data, env.grid)
    #evaluation = evaluator.evaluate(env.get_stowage_plan())
    # Plotting
    #plotter = Plotter(module_path, number_of_episodes, algorithm="Time Difference Q Learning", smoothing_window=smoothing_window)
    #plotter.plot(total_rewards, state_expansion, vehicle_loaded, eps_history)
    ##########################################################################
    # SARSA Training
    agent = SARSA(env, module_path, number_of_episodes)
    model, total_rewards, vehicle_loaded, eps_history, state_expansion = agent.train()
    agent.save_model(module_path)
    evaluator = Evaluator(env.vehicle_data, env.grid)
    # print(env.current_state)
    evaluation = evaluator.evaluate(env.get_stowage_plan())
    print(evaluation)
    #pickle.dump(total_rewards, open(module_path + '_rewards.p', "wb"))
    # pickle.dump(steps_to_exit, open(module_path + '_steps_to_exit.p', "wb"))
    #pickle.dump(eps_history, open(module_path + '_eps_history.p', "wb"))
    # Plotting
    plotter = Plotter(module_path, number_of_episodes, algorithm="SARSA", smoothing_window=None)
    plotter.plot(total_rewards, state_expansion, vehicle_loaded, eps_history)
    #########################################################################
    #number_of_episodes = 12_000
    #evaluator = Evaluator(env.vehicle_data, env.grid)
    #agent = DQNAgent(env=env, module_path=module_path, gamma=0.999, number_of_episodes=number_of_episodes, epsilon=1.0,
    #                 alpha=0.0005, regularisation=0.001,
    #                 mem_size=1_000_000, pretraining_duration=10_000,
    #                 batch_size=32, epsilon_min=0.01, epsilon_dec=0.99999, layers=[64,64])#layers=[250, 250, 250, 250, 250, 250])
    #model, total_rewards, steps_to_exit, eps_history, state_expansion = agent.train()
    #print(datetime.now())

    #pickle.dump(total_rewards, open(module_path + '_rewards.p', "wb"))
    # pickle.dump(steps_to_exit, open(module_path + '_steps_to_exit.p', "wb"))
    #pickle.dump(eps_history, open(module_path + '_eps_history.p', "wb"))

    #agent.save_model(module_path)
    #print(datetime.now())
    #evaluation = evaluator.evaluate(env.get_stowage_plan())
    #print(evaluation)
    env.render()
    # Plotting
    #plotter = Plotter(module_path, number_of_episodes, algorithm="DQN", smoothing_window=100)
    #plotter.plot(total_rewards, state_expansion, steps_to_exit, eps_history)
    #########################################################################
    '''
    logging.getLogger(__name__).info("SHUTDOWN")
