from analysis.algorithms import *


def test_gdc():
    assert greatest_common_divisor(4, 2) == 2
    assert greatest_common_divisor(2, 4) == 2
    assert greatest_common_divisor(2, 2) == 2
    assert greatest_common_divisor(0, 4) == 4
    assert greatest_common_divisor(2, 3) == 1


def test_row_width():
    a = np.array([2, 4, 6])
    b = np.array([8, 10, 12])

    c, d = find_row_width(a, b)
    assert (c == np.array([1, 2, 3])).all() and (d == np.array([4, 5, 6])).all()

    b = np.array([1])
    c, d = find_row_width(a, b)
    assert (c == np.array([2, 4, 6])).all() and (d == np.array([1])).all()


# Corresponds to the example given in thesis
def test_degree_of_sort():

    inversion_no, degree_of_sort = calculate_degree_of_sort([0, 1, 2, 3, 4])
    assert inversion_no == 0 and degree_of_sort == 1.

    inversion_no, degree_of_sort = calculate_degree_of_sort([4, 3, 2, 1, 0])
    assert inversion_no == 10 and degree_of_sort == 0.

    inversion_no, degree_of_sort = calculate_degree_of_sort([1, 0, 4, 2, 3])
    assert inversion_no == 3 and degree_of_sort == 0.7


def test_avg_reward_training_end():
    assert avg_reward_training_end(np.arange(1000)) == 949.5


def test_avg_reward_slope():
    assert avg_reward_reward_slope(np.ones(1500)) == 0
    assert avg_reward_reward_slope(np.arange(1000)) == 1
    assert avg_reward_reward_slope(np.arange(1000) - 500) == 1
    assert avg_reward_reward_slope(np.arange(1234)) == 1
    assert avg_reward_reward_slope(-1 * np.arange(1000)) == -1
