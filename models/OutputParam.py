from models.Models import Models

class OutputParam:
    def __init__(self, colName, path, modelname, target, startInterp=0.0, endInterp=0.0):
        self.colName = colName
        self.path = path
        self.values = []
        self.modelname = modelname
        self.model = self.__getMLModel_()
        self.bestmodel = None
        self.target = target
        self.targetValues = [target]
        self.startInterp = startInterp
        self.endInterp = endInterp

    def __getMLModel_(self):
        mdlfound = False
        for mdl in Models.ModelList:
            if mdl.name == self.modelname:
                mdlfound = True
                return mdl
        return ModelManager.MLMList[0]


