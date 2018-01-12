import abc

from sklearn.externals import joblib


class Model(object):

    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def __init__(self, config):
        pass

    @abc.abstractmethod
    def train(self, X, y):
        pass

    @abc.abstractmethod
    def predict(self, X):
        pass

    @abc.abstractmethod
    def evaluate(self, X, y):
        pass

    @abc.abstractmethod
    def save(self):
        pass

    @abc.abstractmethod
    def load(self):
        pass


class SklearnModel(Model):

    __metaclass__ = abc.ABCMeta

    def __init__(self, name, model):
        self.name = name
        self.model = model

    def train(self, X, y):
        self.model.fit(X, y)

    def save(self, filename):
        joblib.dump(self.model, filename, compress=3)

    def load(self, filename):
        loaded_model = joblib.load(filename)
        assert type(loaded_model) == type(self.model)
        self.model = loaded_model

    @abc.abstractmethod
    def predict(self, X):
        pass

    @abc.abstractmethod
    def evaluate(self, X, y):
        pass


class SklearnRegressor(SklearnModel):
    def predict(self, X):
        return self.model.predict(X)

    def evaluate(self, X, y):
        # TODO
        pass


class SklearnClassifier(SklearnModel):
    def __init__(self, name, model, probability=False):
        super(SklearnClassifier, self).__init__(name, model)
        self.use_probability = probability

    def predict(self, X):
        if self.use_probability:
            return self.model.predict_proba(X)
        else:
            return self.model.predict(X)

    def evaluate(self, X, y):
        # TODO
        pass
