import random

class InputParam:
    def __init__(self, colName, path, minVal=0.0, maxVal=99999.0):
        self.colName = colName
        self.path = path
        self.values = []
        self.newValues = []
        self.minVal = minVal
        self.maxVal = maxVal

    def generateValues(self, num):
        self.newValues.clear()
        for _ in range(num):
            self.newValues.append(round(random.random(), 2) * (self.maxVal - self.minVal) + self.minVal)
