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

def test_shifts():
    random_actions_1 = [0,3,3,0,3,0,2,1,0,4,0,3,4,4,1,0,2,4,4,2,4,3,2,3,0,0,0,3,
                        4,4,2,2,4,1,3,3,3,1,2,2,3,2,3,4,4,2,1,2,0,3,0,2,4,3,2,1,
                        2,1,4,1,0,0,0,0,0,1,0,3,1,4,1,2,4,0,0,2,0,4,4,4,1,1,4,1,
                        1,3,1,3,3,1,1,1,0,3,0,4,1,4,0,3]

    random_actions_2 = [0,3,3,0,3,0,2,1,0,4,0,3,4,4,1,0,2,4,4,2,4,3,2,3,0,0,0,3,
                        4,4,2,2,4,1,3,3,3,1,2,2,3,2,3,4,4,2,1,2,0,3,0,2,4,3,2,1,
                        2,1,4,1,0,0,0,0,0,1,0,3,1,4,1,2,4,0,0,2,0,4,4,4,1,1,4,1,
                        1,3,1,3,3,1,1,1,0,3,0,4,1,4,0,3]

    random_actions_3 = [0,3,3,0,3,0,2,1,0,4,0,3,4,4,1,0,2,4,4,2,4,3,2,3,0,0,0,3,
                        0,3,3,0,3,0,2,1,0,4,0,3,4,4,1,0,2,4,4,2,4,3,2,3,0,0,0,3,
                        2,1,4,1,0,0,0,0,0,1,0,3,1,4,1,2,4,0,0,2,0,4,4,4,1,1,4,1,
                        4,4,2,2,4,1,3,3,3,1,2,2,4,1,2,4,0,0,2,0,4,4,4,1,1,4,1,4,
                        1,3,1,3,3,1,1,1,0,3,0,4,1,4,0,3,2,4]

    i = 0
    env1.reset()
    done = False
    while not done:
        action = random_actions_1[i]
        observation_, reward, done, info = env1.step(action)
        i += 1

    i = 0
    env2.reset()
    done = False
    while not done:
        action = random_actions_2[i]
        observation_, reward, done, info = env2.step(action)
        i += 1

    env3 = RoRoDeck(lanes=8,rows=20)
    env3.reset()
    i = 0
    done = False
    while not done:
        action = random_actions_3[i%len(random_actions_3)]
        observation_, reward, done, info = env3.step(action)
        i += 1
    evaluator1 = Evaluator(env1.vehicle_data, env1.grid)
    shifts1 = evaluator1.evaluate(env1.get_stowage_plan()).shifts

    evaluator2 = Evaluator(env2.vehicle_data, env2.grid)
    shifts2 = evaluator2.evaluate(env2.get_stowage_plan()).shifts

    evaluator3 = Evaluator(env3.vehicle_data, env3.grid)
    shifts3 = evaluator3.evaluate(env3.get_stowage_plan()).shifts

    assert shifts1 == 3
    assert shifts2 == 3
    assert shifts3 == 19




