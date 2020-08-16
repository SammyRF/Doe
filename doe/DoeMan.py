import numpy as np
import pandas as pd
import math
import random
import sklearn
from utils.Utils import Utils
from utils.JsonHandler import JsonHandler
from models.InputParam import InputParam
from models.OutputParam import OutputParam
from models.Models import Models

class DoeMan:
    @staticmethod 
    def generateDoeMan(cfg, casename):
        # read user defines from json
        jsoncase = JsonHandler(cfg).getCase(casename)
        inputs = [InputParam(ipt['name'], jsoncase['file'], ipt['minVal'], ipt['maxVal']) for ipt in jsoncase['inputs']]
        outputs = [OutputParam(opt['name'], jsoncase['file'], opt['model'], opt['target']) for opt in jsoncase['outputs']]
        # set input output values from csv
        data = pd.read_csv(jsoncase['file'])
        for _, row in data.iterrows():
            for ipt in inputs:
                ipt.values.append(row[ipt.colName]) 
            for opt in outputs:
                opt.values.append(row[opt.colName])
        return DoeMan(inputs, outputs, Models.ModelList, 0.8)

    def __init__(self, inputs, outputs, models, ratio=0.8, randomstate=0):
        self.inputs = inputs
        self.datainputs = self.__transpose_(inputs)
        self.outputs = outputs
        self.dataoutputs = []
        self.models = models
        self.ratio = ratio
        self.randomstate = randomstate

    def __transpose_(self, params):
        return [list(v) for v in zip(*[pa.values for pa in params])]

    def generateInputs(self, num):
        for ipt in self.inputs:
            ipt.generateValues(num)
        return [list(v) for v in zip(*[pa.newValues for pa in self.inputs])]

    def compareModelsErrors(self):
        res = ''
        for opt in self.outputs:
            res += "\nOutput: {}".format(opt.colName)
            for mdl, error in sorted(self.__compareSingleOutputErrors_(opt.values).items(), key=lambda x: x[1]):
                if opt.bestmodel is None:
                    opt.bestmodel = mdl
                res += '\n{:20}- Error: {:.2f}'.format(mdl.name, error)
        return res

    def __compareSingleOutputErrors_(self, dataoutput):
        errors = {}
        for mdl in self.models:
            try:
                errors[mdl] = np.mean(self.__compareSingleModelErrors_(dataoutput, mdl))
            except Exception as e:
                errors[mdl] = float('inf')
            if math.isnan(errors[mdl]):
                errors[mdl] = float('inf')
        return errors

    def __compareSingleModelErrors_(self, dataoutput, model):   # simplified because of runtime, once per model
        trainx, testx, trainy, testy = sklearn.model_selection.train_test_split \
            (self.datainputs, dataoutput, train_size = self.ratio, random_state = self.randomstate)
        return [model.calError(trainx, trainy, testx, testy)]

    def checkError(self):
        res = {}
        for opt in self.outputs:
            mdl = opt.bestmodel if not opt.bestmodel is None else opt.model
            testy, predicty, meanerror = self.__checkSingleOutputError_(opt.values, mdl)
            res[opt] = [testy, predicty, '{} - {} - Error: {:.2f}'.format(Utils.getShortName(opt.colName), mdl.name, meanerror)]
        return res

    def __checkSingleOutputError_(self, dataoutput, model):
        trainx, testx, trainy, testy = sklearn.model_selection.train_test_split \
            (self.datainputs, dataoutput, train_size = self.ratio, random_state = self.randomstate)
        medianerror = model.calError(trainx, trainy, testx, testy)
        predicty = model.predict(testx)
        return testy, predicty, medianerror

    def fitAndPredict(self, num):
        res = {}
        for opt in self.outputs:
            mdl = opt.bestmodel if not opt.bestmodel is None else opt.model
            predicty = self.__fitAndPredictSingleOutput_(opt.values, self.generateInputs(num), mdl)
            targety = [opt.targetValues for _ in range(num)]
            tmplist = []
            for singlepredicty in predicty:
                tmplist.append(np.mean(abs(singlepredicty - np.array(opt.targetValues)) / np.amax(abs(opt.targetValues[0]))*100))
            meanerror = np.mean(tmplist)
            tempname = Utils.getShortName(opt.colName)
            res[opt] = [targety, predicty, '{} - {} - Gap: {:.2f}'.format(tempname, mdl.name, meanerror)]
        return res

    def __fitAndPredictSingleOutput_(self, dataoutput, generatedinputs, model):
        model.fit(self.datainputs, dataoutput)
        return model.predict(generatedinputs)



