from sklearn import preprocessing
from sklearn.metrics import mean_squared_error

class Model:
    def __init__(self, name, model, ploydegree=1):
        self.name = name
        self.model = model
        self.ploydegree = ploydegree
        self.poly = preprocessing.PolynomialFeatures(degree=int(ploydegree))

    def calError(self, trainx, trainy, testx, testy):
        trainx_p = self.poly.fit_transform(trainx)
        self.model.fit(trainx_p, trainy)
        testx_p = self.poly.fit_transform(testx)
        predicty = self.model.predict(testx_p)
        return mean_squared_error(testy, predicty, squared=True)

    def fit(self, trainx, trainy):
        trainx_p = self.poly.fit_transform(trainx)
        self.model.fit(trainx_p, trainy)

    def predict(self, testx):
        testx_p = self.poly.fit_transform(testx)
        return self.model.predict(testx_p)




