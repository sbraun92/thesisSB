from algorithms.algorithms import *
import pytest


def test_gdc():
    assert greatest_common_divisor(4, 2) == 2
    assert greatest_common_divisor(2, 4) == 2
    assert greatest_common_divisor(2, 2) == 2
    assert greatest_common_divisor(0, 4) == 4
    assert greatest_common_divisor(2, 3) == 1

#TODO
def test_row_width():
    pass
#TODO
def test_degree_of_sort():
    pass

def test_avg_reward_training_end():
    assert avg_reward_training_end(np.arange(1000)) == 949.5


def test_avg_reward_slope():
    assert avg_reward_reward_slope(np.ones(1500)) == 0
    assert avg_reward_reward_slope(np.arange(1000)) == 1
    assert avg_reward_reward_slope(np.arange(1000) - 500) == 1
    assert avg_reward_reward_slope(np.arange(1234)) == 1
    assert avg_reward_reward_slope(-1*np.arange(1000)) == -1

def test_training_convergence():
    assert training_convergence(np.arange(1000)) == 1
    assert training_convergence(-0.1*np.arange(1000)) == -0.1
    assert training_convergence(2*np.ones(500)) == 0

