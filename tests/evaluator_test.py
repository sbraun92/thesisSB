from env.roroDeck import RoRoDeck
from evaluator.evaluator import Evaluator
from algorithms.analyser import Analysor
import pytest
import numpy as np


#Preperation for Tests
np.random.seed(0)

#Create a random stowagePlan
env1 = RoRoDeck(True)
env2 = RoRoDeck(True)

done = False
totalrewards_env1 = 0

while (not done):
    action = env1.actionSpaceSample()
    observation_, reward, done, info = env1.step(action)
    totalrewards_env1 += reward

done = False
totalrewards_env2 = 0

while (not done):
    action = env2.actionSpaceSample()
    observation_, reward, done, info = env2.step(action)
    totalrewards_env2 += reward



#Test if the mandatory Cargo Loaded is reasonable
def test_Evaluator_MadatoryCargoLoaded():
    evaluator1 = Evaluator(env1.vehicleData,env1.grid)
    mandatoryCargoLoaded_env1 = evaluator1.evaluate(env1.getStowagePlan()).mandatoryCargoLoaded

    evaluator2 = Evaluator(env2.vehicleData,env2.grid)
    mandatoryCargoLoaded_env2 = evaluator2.evaluate(env2.getStowagePlan()).mandatoryCargoLoaded

    assert mandatoryCargoLoaded_env1 <= 1
    assert mandatoryCargoLoaded_env1 >= 0

    assert mandatoryCargoLoaded_env2 <= 1
    assert mandatoryCargoLoaded_env2 >= 0


#Test if the space utilisation of stowage plans is reasonable
def test_Evaluator_SpaceUtilisation():
    evaluator1 = Evaluator(env1.vehicleData,env1.grid)
    spaceUtilisation_env1 = evaluator1.evaluate(env1.getStowagePlan()).spaceUtilisation

    evaluator2 = Evaluator(env2.vehicleData,env2.grid)
    spaceUtilisation_env2 = evaluator2.evaluate(env2.getStowagePlan()).spaceUtilisation

    assert spaceUtilisation_env1 <= 1
    assert spaceUtilisation_env1 >= 0

    assert spaceUtilisation_env2 <= 1
    assert spaceUtilisation_env2 >= 0


#Test if the Evaluator and the agents estimate are consensually

def test_AgentEvaluatorConsensus():
    evaluator1 = Evaluator(env1.vehicleData,env1.grid)
    evaluation1 = evaluator1.evaluate(env1.getStowagePlan())
    evaluator2 = Evaluator(env2.vehicleData,env2.grid)
    evaluation2 = evaluator2.evaluate(env2.getStowagePlan())

    if evaluation1 >= evaluation2:
        assert totalrewards_env1 >= totalrewards_env2
    else:
        assert totalrewards_env1 < totalrewards_env2