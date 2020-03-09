import numpy as np

class Evaluation(object):
    def __init__(self, evaluation):
        self.shifts = evaluation[0]
        self.spaceUtilisation = evaluation[1]
        self.mandatoryCargoLoaded = evaluation[2]
        self.vehicleData = evaluation[3]
        self.deckLayout = evaluation[4]

    def __str__(self):
        return "Number of Shifts: "+str(self.shifts) +" Space Utilisation: "+ str(self.spaceUtilisation) + ", Mandatory Cargo Loaded: " + str(self.mandatoryCargoLoaded) +" "


    #Comparision of two StowagePlan Evaluations

    # Compare two Evaluations:
    #   1. Evaluation with more mandatory Cargo Loaded is always better
    #   2. if Evaluation equal in 1. than is the Stowage Plan with less shifts better
    #   3. if 1. and 2. equal than the Stowage Plan with a higher SpaceUtilisation is better


    def __eq__(self, other):
        if self._arePlansComparable(other):
            if self.shifts == other.shifts \
                    and self.spaceUtilisation == other.spaceUtilisation \
                    and self.mandatoryCargoLoaded == other.mandatoryCargoLoaded:
                return True
            else:
                return False
        else:
            return False

    def __gt__(self,other):
        if self._arePlansComparable(other):
            if self.mandatoryCargoLoaded > other.mandatoryCargoLoaded:
                return True
            elif self.mandatoryCargoLoaded < other.mandatoryCargoLoaded:
                return False
            else:
                if self.shifts < other.shifts:
                    return True
                elif self.shifts > other.shifts:
                    return False
                else:
                    if self.spaceUtilisation > other.spaceUtilisation:
                        return True
                    else:
                        return False
        else:
            return False

    def __ge__(self, other):
        if self._arePlansComparable(other):
            if self.__eq__(other) or self.__gt__(other):
                return True
            else:
                return False
        else:
            return False

    def _arePlansComparable(self,other):
        if np.array_equal(self.vehicleData,other.vehicleData) and np.array_equal(self.deckLayout,other.deckLayout):
            return True
        else:
            return False