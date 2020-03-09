class Analysor(object):
    def __init__(self):
        pass

    def calculateInversionNumber(self,sequence):
        inversionNo = 0
        seq = sequence.flatten()
        size = len(sequence)

        for i in range(size):
            for j in range(i+1,size):
                if seq[i] > seq[j]:
                    inversionNo += 1

        return inversionNo
