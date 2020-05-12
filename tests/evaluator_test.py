from env.roroDeck import RoRoDeck
from valuation.evaluator import *
#from algorithms.inversionNumber import InversionNumberCalculator
import pytest
import numpy as np

#Preperation for Tests
np.random.seed(0)

#Create a random stowagePlan
env1 = RoRoDeck(True)
env2 = RoRoDeck(True)

env1.reset()
env2.reset()

done = False
totalrewards_env1 = 0

while (not done):
    action = env1.action_space_sample()
    observation_, reward, done, info = env1.step(action)
    totalrewards_env1 += reward

done = False
totalrewards_env2 = 0

while (not done):
    action = env2.action_space_sample()
    observation_, reward, done, info = env2.step(action)
    totalrewards_env2 += reward



#Test if the mandatory Cargo Loaded is reasonable
def test_Evaluator_MadatoryCargoLoaded():
    evaluator1 = Evaluator(env1.vehicle_data, env1.grid)
    mandatoryCargoLoaded_env1 = evaluator1.evaluate(env1.get_stowage_plan()).mandatory_cargo_loaded

    evaluator2 = Evaluator(env2.vehicle_data, env2.grid)
    mandatoryCargoLoaded_env2 = evaluator2.evaluate(env2.get_stowage_plan()).mandatory_cargo_loaded

    assert mandatoryCargoLoaded_env1 <= 1
    assert mandatoryCargoLoaded_env1 >= 0

    assert mandatoryCargoLoaded_env2 <= 1
    assert mandatoryCargoLoaded_env2 >= 0


#Test if the space utilisation of stowage plans is reasonable
def test_Evaluator_SpaceUtilisation():
    evaluator1 = Evaluator(env1.vehicle_data, env1.grid)
    spaceUtilisation_env1 = evaluator1.evaluate(env1.get_stowage_plan()).space_utilisation

    evaluator2 = Evaluator(env2.vehicle_data, env2.grid)
    spaceUtilisation_env2 = evaluator2.evaluate(env2.get_stowage_plan()).space_utilisation

    assert spaceUtilisation_env1 <= 1
    assert spaceUtilisation_env1 >= 0

    assert spaceUtilisation_env2 <= 1
    assert spaceUtilisation_env2 >= 0


#Test if the Evaluator and the agents estimate are consensually
def test_AgentEvaluatorConsensus():
    evaluator1 = Evaluator(env1.vehicle_data, env1.grid)
    evaluation1 = evaluator1.evaluate(env1.get_stowage_plan())
    evaluator2 = Evaluator(env2.vehicle_data, env2.grid)
    evaluation2 = evaluator2.evaluate(env2.get_stowage_plan())

    if evaluation1 >= evaluation2:
        assert totalrewards_env1 >= totalrewards_env2
    else:
        assert totalrewards_env1 < totalrewards_env2


#Ignore
def test_time():
    env3 = RoRoDeck(lanes=50,rows=80)
    env3.reset()

    done = False

    while (not done):
        action = env3.action_space_sample()
        observation_, reward, done, info = env3.step(action)


    #env3.render()

    evaluator3 = Evaluator(env3.vehicle_data, env3.grid)
    evaluation3 = evaluator3.evaluate(env3.get_stowage_plan())

    print(evaluation3)