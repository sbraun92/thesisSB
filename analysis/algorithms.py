from sympy.combinatorics import Permutation
import numpy as np
import logging


def calculate_degree_of_sort(seq):
    """
    Calculates the degree of sort of a permutation as outlined in the thesis
    Args:
        seq:                a permutation

    Returns:
        inversion_no:       the inversion number of permutation
        degree_of_sort:     measure as defined in thesis
    """

    size = len(seq)
    inversion_no = Permutation(seq).inversions()

    max_inversion_no = size * (size - 1.) / 2.

    # degree of sort equals 1 if perfectly sorted and 0 if ordered reversely
    degree_of_sort = 1. - inversion_no / max_inversion_no

    return inversion_no, degree_of_sort


def greatest_common_divisor(x, y):
    """Calculates recursively the gcd of integers x and y by Euclidean's algorithm"""
    if y == 0:
        return x
    return greatest_common_divisor(y, x % y)


def find_row_width(end_of_lanes, cargo_length):
    """
    Use Euclidean's algorithm repeatedly on all cargo types and deck length to find gcd
    Args:
        end_of_lanes(np.array):     Capacity of every lane
        cargo_length(np.array):     Lengths of each cargo type

    Returns:
        simplified end_of_lanes
        simplified cargo_length
    """

    logging.getLogger(__name__).info('Use Euclidean\'s algorithm to reduce row dimensions and vehicle lengths...')
    solution = end_of_lanes[0]
    # Find gcd for end_of_lanes
    for usable_row_capacity in end_of_lanes:
        solution = greatest_common_divisor(solution, usable_row_capacity)
    # Find gcd for end_of_lanes and
    for veh in cargo_length:
        solution = greatest_common_divisor(solution, veh)
    # Log solutions
    if solution == 1:
        logging.getLogger(__name__).info('Greatest common divisor equals 1 -> Cannot reduce dimensions')
        return end_of_lanes, cargo_length
    else:
        logging.getLogger(__name__).info('Greatest common divisor equals {} -> Reduce dimensions.\n'.format(solution) +
                                         ' The new unit is now 1/{} of the initial unit'.format(solution))

        return (end_of_lanes / solution).astype(np.int), (cargo_length / solution).astype(np.int)


def avg_reward_training_end(rewards):
    """Calculate average reward of last 100 episodes"""
    return np.mean(rewards[-100:])


def avg_reward_reward_slope(rewards):
    """Estimate beta (average reward increase per episode)"""
    return (1. / (len(rewards) - 100)) * (np.mean(rewards[-100:]) - np.mean(rewards[0:100]))


def training_metrics(rewards):
    """Append training metrics for logging"""
    info = '\n' + '*' * 80 + '\nTraining Evaluation Metrics:\n'
    metrics = list()
    metrics.append(avg_reward_training_end(rewards))
    info += 'Average Reward of last 100 Episodes:\t\t\t\t' + str(metrics[0]) + '\n'
    metrics.append(avg_reward_reward_slope(rewards))
    info += 'Average Increase of Rewards in Training:\t\t\t' + str(metrics[1]) + '\n'

    logging.getLogger(__name__).info(info)

    return metrics, info
