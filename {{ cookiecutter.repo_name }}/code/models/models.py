"""In this module, the user implements get_models(config), which returns a list of models that
subclass the code.models.Model class. Each of these models will be evaluated/trained when running
evaluate_models.py or train_models.py."""

import abc

import numpy as np
from sklearn import model_selection
from sklearn.externals import joblib


def get_models(config):
    """Returns a list of models that implement the Model class below. The config passed in is the
    dict defined in the "models" section in config.toml."""
    return []


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
    def evaluate(self, X, y, scoring, cv, groups):
        pass


class SklearnRegressor(SklearnModel):
    def predict(self, X):
        return self.model.predict(X)

    def evaluate(self, X, y, scoring='r2', cv=10, groups=None):
        scores = model_selection.cross_val_score(
            self.model, X, y, scoring=scoring, cv=cv, groups=groups)
        return np.mean(scores)


class SklearnClassifier(SklearnModel):
    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def evaluate(self, X, y, scoring='roc_auc', cv=10, groups=None):
        scores = model_selection.cross_val_score(
            self.model, X, y, scoring=scoring, cv=cv, groups=groups)
        return np.mean(scores)
