import numpy as np

class Evaluator(object):
    def __init__(self, vehicleData, weights = None):
        self.stowagePlan = None
        # 1. Number of shifts
        # 2. mandatory Cargo loaded [%]
        # 3. Space utilisation [%]
        self.evaluationCriteria = np.zeros(3)

        self.vehicleData = vehicleData

        self.mandatoryVeh = vehicleData[2]==1

        self.numberOfVehicle = np.zeros(len(vehicleData.T))


        if weights == None:
            self.weights = np.ones(3)
        else:
            self.weights = weights

    def evaluate(self,stowagePlan):
        self.stowagePlan = stowagePlan


    def _calculateNumberOfShifts(self):
        pass

    def calculateMandatoryCargoLoaded(self):
        for i in range(len(self.numberOfVehicle)):
            self.numberOfVehicle[i] = len(np.where(self.stowagePlan[1].flatten()==i)[0])
        print(self.numberOfVehicle)
        loadedMandatoryVeh = np.sum(self.numberOfVehicle[self.mandatoryVeh])
        allMandatoryVeh = np.sum(self.vehicleData[4][self.mandatoryVeh])

        return float(loadedMandatoryVeh/allMandatoryVeh)

    def calculateSpaceUtilisations(self):
        grid = self.stowagePlan[0]  #loading sequence
        grid = grid.flatten()
        capacity = len(grid)- len(grid[grid==-1])
        freeSpace = len(grid[grid==0])

        return 1.-(freeSpace/capacity)


