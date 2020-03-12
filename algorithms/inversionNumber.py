from sympy.combinatorics import Permutation
from sympy.combinatorics.partitions import Partition
import numpy as np
class InversionNumberCalculator(object):
    def __init__(self):
        return


    def calculateInversionNumber(self,seq):

        size = len(seq)
        inversionNo = Permutation(seq).inversions()

        maxInversionNo = size*(size-1.)/2.

        degreeOfSort = 1. - inversionNo / maxInversionNo #degree of sorting 1:perfectly sorted 0: reversed sorted - everything is opposite

        return inversionNo, degreeOfSort
