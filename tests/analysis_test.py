from algorithms.algorithms import *
import numpy as np


def test_inversion_number():
    seq1 = (np.array([0, 1, 2, 3, 4]), 0)
    seq2 = (np.array([4, 3, 2, 1, 0]), 10)

    sequences = [seq1, seq2]

    for seq in sequences:
        estimated_inv_no, degree_of_sort = calculate_degree_of_sort(seq[0])
        true_inv_no = seq[1]

        assert estimated_inv_no == true_inv_no
        assert 1 >= degree_of_sort >= 0


def test_rewardShaping():
    pass


def test_loggingUnit():
    pass


def test_rewardSystem():
    pass
