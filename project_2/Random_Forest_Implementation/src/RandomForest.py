import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import random
import numpy as np
import operator

class RandomForest():

    def __init__(self, max_features = 2, forest_size = 10, max_depth = 5):
        self.maxFeatures = self.validateArgs(max_features)
        self.forestSize  = self.validateArgs(forest_size)
        self.maxDepth    = self.validateArgs(max_depth)
        self.tree_dict   = {}
        self.X           = None
        self.y           = None
        self.y_unique_val= None

    def setX(self, X):
        if type(X) is np.ndarray and len(X) > 0:
            self.X = X
        else:
            raise ValueError('Argument is not valid: should be ndarray, instead got ',type(X))

    def sety(self, y):
        if type(y) is np.ndarray:
            self.y = y
        else:
            raise ValueError('Argument is not valid: should be ndarray, instead got ',type(y))

    def setTreeDict(self, key_name, tree, y_indexes):
        if isinstance(tree,DecisionTreeClassifier):
            self.tree_dict[key_name] = (tree, y_indexes)
        else:
            raise ValueError('Argument is not valid: should be DecisionTreeClassifier, instead got ', type(tree))

    def setUniqueVals(self):
        self.y_unique_val = self.unique_vals(self.y)
        return

    @classmethod
    def validateArgs(clf, number):
        if number < 0 or number > 10000:
            raise Exception('Argument is not valid: smaller than 0 or greater than 10000')
        return number

    @classmethod
    def validateDataArg(clf, data):
        if len(data) == 0 or (not isinstance(data, pd.DataFrame)):
            raise Exception('Data argument is not valid: either emtpy or not a DataFrame type')
        return data

    @classmethod
    def unique_vals(clf, y):
        if y is not None:
            return np.unique(y)
        else:
            return None

    def bootstrap(self, rand_features):
        """bootstrap the data by demand"""

        rand_index = sorted(random.choices(range(self.X.shape[0]), k=self.X.shape[0]))
        x = (self.X[:, rand_features])[rand_index,:]
        y =  self.y[rand_index,:]
        return x, y

    def fit(self, X, y):
        self.setX(X)
        self.sety(y)
        self.setUniqueVals()

        for tree in range(self.forestSize):
            rand_features = sorted(random.sample(range(self.X.shape[1]), k = self.maxFeatures))
            bootstrapped_x, bootstrapped_y = self.bootstrap(rand_features)
            new_tree = DecisionTreeClassifier(max_depth = self.maxDepth).fit(bootstrapped_x, bootstrapped_y)
            self.setTreeDict(tree, new_tree, rand_features)

    def one_tree_prediction(self, tree_key, x):

        y_indexes = self.tree_dict[tree_key][1]
        x_new = [[x[0][i] for i in y_indexes]] if len(x) == 2 else [[x[i] for i in y_indexes]]
        y_hat = self.tree_dict[tree_key][0].predict(x_new)
        return y_hat[0]

    def predict_row(self, x):
        y_hat_distribution_series = {y_val: 0 for y_val in self.y_unique_val}
        tree_predictions = map(lambda tree: self.one_tree_prediction(tree, x), self.tree_dict.keys())
        for y_hat in tree_predictions:
            y_hat_distribution_series[y_hat] = y_hat_distribution_series[y_hat] + 1
        return max(y_hat_distribution_series.items(), key=operator.itemgetter(1))[0]

    def predict(self, x_pred):
        y_hat = []
        for row in x_pred:
            y_hat.append(self.predict_row(row))
        return np.asarray(y_hat)