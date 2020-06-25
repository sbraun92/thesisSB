import pytest

from env.roroDeck import RoRoDeck
from valuation.evaluator import *

# Preparation for Tests
np.random.seed(0)

# Create a random stowagePlan
env1 = RoRoDeck(True)
env2 = RoRoDeck(True)

env1.reset()
env2.reset()

done = False
total_rewards_env1 = 0

while not done:
    action = env1.action_space_sample()
    observation_, reward, done, info = env1.step(action)
    total_rewards_env1 += reward

done = False
total_rewards_env2 = 0

while not done:
    action = env2.action_space_sample()
    observation_, reward, done, info = env2.step(action)
    total_rewards_env2 += reward


# Test if the mandatory Cargo Loaded is reasonable
def test_evaluator_mandatory_cargo_loaded():
    evaluator1 = Evaluator(env1.vehicle_data, env1.grid)
    mandatory_cargo_loaded_env1 = evaluator1.evaluate(env1.get_stowage_plan()).mandatory_cargo_loaded

    evaluator2 = Evaluator(env2.vehicle_data, env2.grid)
    mandatory_cargo_loaded_env2 = evaluator2.evaluate(env2.get_stowage_plan()).mandatory_cargo_loaded

    assert mandatory_cargo_loaded_env1 <= 1
    assert mandatory_cargo_loaded_env1 >= 0

    assert mandatory_cargo_loaded_env2 <= 1
    assert mandatory_cargo_loaded_env2 >= 0


# Test if the space utilisation of stowage plans is reasonable
def test_evaluator_space_utilisation():
    evaluator1 = Evaluator(env1.vehicle_data, env1.grid)
    space_utilisation_env1 = evaluator1.evaluate(env1.get_stowage_plan()).space_utilisation

    evaluator2 = Evaluator(env2.vehicle_data, env2.grid)
    space_utilisation_env2 = evaluator2.evaluate(env2.get_stowage_plan()).space_utilisation

    assert space_utilisation_env1 <= 1
    assert space_utilisation_env1 >= 0

    assert space_utilisation_env2 <= 1
    assert space_utilisation_env2 >= 0


# Test if the Evaluator and the agents estimate are consensually
def test_agent_evaluator_consensus():
    evaluator1 = Evaluator(env1.vehicle_data, env1.grid)
    evaluation1 = evaluator1.evaluate(env1.get_stowage_plan())
    evaluator2 = Evaluator(env2.vehicle_data, env2.grid)
    evaluation2 = evaluator2.evaluate(env2.get_stowage_plan())

    if evaluation1 >= evaluation2:
        assert total_rewards_env1 >= total_rewards_env2
    else:
        assert total_rewards_env1 < total_rewards_env2

