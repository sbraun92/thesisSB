import numpy as np
from evaluator.evaluation import Evaluation


class Evaluator(object):
    def __init__(self, vehicleData, deckLayout, weights = None):
        self.stowagePlan = None
        # 1. Number of shifts
        # 2. mandatory Cargo loaded [%]
        # 3. Space utilisation [%]
        self.evaluationCriteria = np.zeros(3)

        self.vehicleData = vehicleData

        self.mandatoryVeh = vehicleData[2]==1

        self.numberOfVehicle = np.zeros(len(vehicleData.T))

        # A Layout of deck (empty or loaded)
        self.deckLayout = deckLayout


        if weights == None:
            self.weights = np.ones(3)
        else:
            self.weights = weights

    def evaluate(self,stowagePlan):
        if not self.isStowagePlanCompatible(stowagePlan):
            assert False

        self.stowagePlan = stowagePlan

        shifts = self.calculateNumberOfShifts()

        spaceUtilisation = self.calculateSpaceUtilisations()

        mandatoryCargoLoaded = self.calculateMandatoryCargoLoaded()


        return Evaluation((shifts,spaceUtilisation,mandatoryCargoLoaded,self.vehicleData,self.deckLayout))


    def calculateNumberOfShifts(self):
        totalShifts = np.zeros(len(self.stowagePlan[1]))
        destinations = np.unique(self.vehicleData[1].copy())

        #Loop over all lanes
        for lane_ix,lane in enumerate(self.stowagePlan[1]):
            badQueue = False
            for ix,vehicle in enumerate(lane):
                if lane[ix+1] == -1: # reached end of queue
                    break

                destination_first = self.vehicleData[1][vehicle]
                destination_second = self.vehicleData[1][lane[ix+1]]

                if destination_first == destination_second:
                    if destination_first != destinations[0] and badQueue:
                        totalShifts[lane_ix]+=1

                if destination_first < destination_second:
                    totalShifts[lane_ix]+=1
                    badQueue = True

                if destination_first > destination_second and badQueue:
                    badQueue = False

        return np.sum(totalShifts)

    def calculateMandatoryCargoLoaded(self):
        for i in range(len(self.numberOfVehicle)):
            self.numberOfVehicle[i] = len(np.where(self.stowagePlan[1].flatten()==i)[0])

        loadedMandatoryVeh = np.sum(self.numberOfVehicle[self.mandatoryVeh])
        allMandatoryVeh = np.sum(self.vehicleData[4][self.mandatoryVeh])

        return float(loadedMandatoryVeh/allMandatoryVeh)

    def calculateSpaceUtilisations(self):
        grid = self.stowagePlan[0]  #loading sequence
        grid = grid.flatten()
        capacity = len(grid)- len(grid[grid==-1])
        freeSpace = len(grid[grid==0])

        return 1.-(freeSpace/capacity)

    def isStowagePlanCompatible(self, stowagePlan):
        if len(stowagePlan[0]) == len(self.deckLayout) and len(stowagePlan[0].T) == len(self.deckLayout.T):
            return True
        else:
            return False
