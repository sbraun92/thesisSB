import numpy as np

class Evaluator(object):
    def __init__(self, stowagePlan, mandatoryCargo, weights = None):
        self.stowagePlan = stowagePlan
        # 1. Number of shifts
        # 2. mandatory Cargo loaded [%]
        # 3. Space utilisation [%]
        self.evaluationCriteria = np.zeros(3)

        self.mandatoryCargo = mandatoryCargo

        if weights == None:
            self.weights = np.ones(3)
        else:
            self.weights = weights

    def evaluate(self):
        pass

    def _calculateNumberOfShifts(self):
        pass

    def _calculateMandatoryCargoLoaded(self):
        pass

    def _calculateSpaceUtilisations(self):
        grid = self.stowagePlan[0]  #loading sequence
        grid = grid.flatten()
        capacity = len(grid)- grid[grid==-1]
        freeSpace = len(grid[grid==0])

        return float(1.-(freeSpace/capacity))


