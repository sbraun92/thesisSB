from env.roroDeck import RoRoDeck
from agent.SARSA import SARSA
from analysis.Plotter import Plotter
from valuation.evaluator import *
from analysis.loggingUnit import LoggingBase
from datetime import datetime
import logging
import numpy as np
import pickle
if __name__ == '__main__':

    #Register Outputpath and Logger
    loggingBase = LoggingBase()
    module_path = loggingBase.module_path

    it = 8_000_0
    logging.getLogger('log1').info("Train for {it} iterations.")

    smoothing_window = int(it / 100)
    smoothing_window =200

    #Test with a bigger configuration
    env = RoRoDeck(True, 8, 12, stochastic=False)

    vehicleData = np.array([[0, 1, 2, 3, 4],  # vehicle id
                            [1, 2, 1, 2, 2],  # destination
                            [1, 1, 0, 0,1],  # mandatory
                            [2, 3, 2, 3, 5],  # length
                            [7, 7, -1,-1, 2]])  # number of vehicles on yard
                                                          # (-1 denotes there are infinite vehicles of that type)


    '''
    # SARSA FUNCTION APPROX
    agent = SARSALinFunApprox(env,module_path,it)
    q_table, totalRewards, stateExpantion, stepsToExit, eps_history = agent.train()
    #agent.save_model(module_path,type='pickle')
    evaluator = Evaluator(env.vehicle_Data, env.grid)
    print(env.current_state)
    evaluation = evaluator.evaluate(env.getStowagePlan())
    print(evaluation)
    #Plotting
    plotter = Plotter(module_path, it, algorithm="Time Difference Q Learning")
    plotter.plot(totalRewards, stateExpantion, stepsToExit,eps_history)
    '''
    #######################################################################
    # TDQ Training
    #agent = TDQLearning(env,module_path,it)
    #q_table, totalRewards, stepsToExit, eps_history, stateExpantion = agent.train()
    #agent.save_model(module_path,type='pickle')
    evaluator = Evaluator(env.vehicle_data, env.grid)
    #print(env.current_state)
    #evaluation = evaluator.evaluate(env.getStowagePlan())
    #print(evaluation)
    #Plotting
    #plotter = Plotter(module_path, it, algorithm="Time Difference Q Learning")
    #plotter.plot(totalRewards, stateExpantion, stepsToExit,eps_history)
    ##########################################################################
    # SARSA Training
    agent = SARSA(env, module_path, it)
    q_table, total_rewards, steps_to_exit, eps_history, state_expansion = agent.train()
    print(datetime.now())

    pickle.dump(total_rewards, open(module_path + '_rewards.p', "wb"))
    pickle.dump(steps_to_exit, open(module_path + '_steps_to_exit.p', "wb"))
    pickle.dump(eps_history, open(module_path + '_eps_history.p', "wb"))

    agent.save_model(module_path, file_format='pickle')
    print(datetime.now())
    evaluation = evaluator.evaluate(env.get_stowage_plan())
    print(evaluation)
    env.render()
    #Plotting
    plotter = Plotter(module_path, it, algorithm="SARSA")
    plotter.plot(total_rewards, state_expansion, steps_to_exit, eps_history)
    #########################################################################
    agent = DQN()
    _ = agent.train()




    #'''

    '''
    ds1 = []
    ds2 = []
    for i in range(15):
        #old version
        start = datetime.now()
    
        it = 2000
        logging.getLogger('log1').info("Train for "+str(it)+" iterations.")
    
        smoothing_window = int(it/100)
    
        env = RoRoDeck(False)
        #Training
        agent = TDQLearning(env,module_path,it,True)
        q_table, totalRewards, stateExpantion, stepsToExit = agent.train()
        ds1+=[(datetime.now() - start).total_seconds()]
    
        #Changed version
        start = datetime.now()
    
        env2 = RoRoDeck(False)
        #Training
        #Try false to check method maxAction
        agent2 = TDQLearning(env,module_path,it,False)
        q_table, totalRewards, stateExpantion, stepsToExit = agent2.train()
        ds2+=[(datetime.now() - start).total_seconds()]
    
    df = pd.DataFrame()
    df['orig']= ds1
    df['faster'] = ds2
    print(df)
    sns.lineplot(data=df)
    plt.show()
    '''



    logging.getLogger('log1').info("SHUTDOWN")
