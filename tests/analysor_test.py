from env.roroDeck import RoRoDeck
from valuation.evaluator import Evaluator
from algorithms.inversionNumber import Analysor
import pytest
import numpy as np


def test_Inversionnumber():

    seq1 = (np.array([0,1,2,3,4]),0)
    seq2 = (np.array([4,3,2,1,0]),10)
    #seq3 = (np.array([1,1,1,1]),0)
    #seq4 = (np.array([-1,1,-1,1]),1)

    sequences = [seq1,seq2]#,seq3,seq4]

    analysor = Analysor()

    for seq in sequences:
        estimated_invNo, degreeOfSort = analysor.calculateInversionNumber(seq[0])
        true_invNo = seq[1]

        assert estimated_invNo == true_invNo
        assert degreeOfSort <=1 and degreeOfSort >= 0