from env.RoRoDeck import RoRoDeck
from evaluator.evaluator import Evaluator
import pytest
import numpy as np

np.random.seed(0)

#Create a random stowagePlan
env = RoRoDeck(True)
env.render()
done = False

print(env.loadedVehicles)

while (not done):
    action = env.actionSpaceSample()
    observation_, reward, done, info = env.step(action)




def test_Evaluator_MadatoryCargoLoaded():
    evaluator = Evaluator(env.vehicleData)
    evaluator.evaluate(env.getStowagePlan())

    assert evaluator.calculateMandatoryCargoLoaded() <= 1

def test_Evaluator_SpaceUtilisation():
    evaluator = Evaluator(env.vehicleData)
    evaluator.evaluate(env.getStowagePlan())

    assert evaluator.calculateSpaceUtilisations()<=1
