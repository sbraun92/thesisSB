from sympy.combinatorics import Permutation
import numpy as np
import logging
from scipy.stats import linregress
def calculate_degree_of_sort(seq):
    """
    Calculates the degree of sort of a permutation as outlined in the thesis


    Parameters
    ----------
    seq                 a permutation

    Returns
    -------
    inversion_no        the inversion number of permutation
    degree_of_sort      measure as defined in thesis
    """

    size = len(seq)
    inversion_no = Permutation(seq).inversions()

    max_inversion_no = size * (size - 1.) / 2.

    # degree of sort equals 1 if perfectly sorted and 0 if ordered reversely
    degree_of_sort = 1. - inversion_no / max_inversion_no

    return inversion_no, degree_of_sort


def greatest_common_divisor(x, y):
    """Calculates recursively the greatest common divisor of integers x and y by Euclidean's algorithm"""
    if y == 0:
        return x
    return greatest_common_divisor(y, x % y)
#TODO comments
#TODO sonderfaelle
def find_row_width(deck_length, vehicle_width):
    logging.getLogger(__name__).info('Use algorithm of Euclid to reduce row dimensions and vehicle lengths...')
    solution = deck_length
    for veh in vehicle_width:
        solution = greatest_common_divisor(solution, veh)
    if solution == 1:
        logging.getLogger(__name__).info('Greatest common divisor equals 1 -> Cannot reduce dimensions')
        return deck_length, vehicle_width
    else:
        logging.getLogger(__name__).info('Greatest common divisor equals {} -> Reduce dimensions.\n'.format(solution) +
                                         ' The new unit is now 1/{} of the inital unit'.format(solution))
        logging.getLogger(__name__).info('New Row width is {}'.format(int(deck_length / solution)))
        return int(deck_length / solution), (vehicle_width / solution).astype(np.int)


def avg_reward_training_end(rewards):
    return np.mean(rewards[-100:])

def avg_reward_reward_slope(rewards):
    return (1./(len(rewards)-100))*(np.mean(rewards[-100:])-np.mean(rewards[0:100]))

def training_convergence(rewards):
    beta,_,_,_,_ = linregress(np.arange(len(rewards)),rewards)
    return beta

def training_metrics(rewards):
    info = '*'*60 +'\nTraining evaluation metrics:\n'
    metrics = list()
    metrics.append(avg_reward_training_end(rewards))
    info += 'Average Reward of last 100 Episodes:\t\t\t'+str(metrics[0])+'\n'
    metrics.append(avg_reward_reward_slope(rewards))
    info += 'Average Increase of Rewards in Training:\t\t' + str(metrics[1]) + '\n'
    metrics.append(training_convergence(rewards))
    info += 'Average Increase of Rewards of last 100 Episodes:\t' + str(metrics[2])+'\n' + '*'*60+'\n'
    logging.getLogger(__name__).info(info)

    return metrics, info
