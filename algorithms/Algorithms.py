from sympy.combinatorics import Permutation


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
