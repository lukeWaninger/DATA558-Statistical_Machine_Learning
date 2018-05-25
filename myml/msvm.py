from myml.my_classifier import MyClassifier
from myml.loss_functions import *
from myml.regularization import *
from enum import Enum
import numpy as np


class MySVM(MyClassifier):
    """my linear support vector machine"""

    def __init__(self, x_train, y_train, parameters, *args, **kwargs):
        """initialize the classifier

        Args:
            x_train (ndarray): nXm array of training samples
            y_train (ndarray: nX1 array of labels
            parameters (dict) - required keys:
                loss (str): $\in$ { 'squared_hinge', 'smoothed_hinge' }
                h (float): required when loss is set to smoothed_hinge
            *args:
            **kwargs:
        """
        super().__init__(x_train, y_train, parameters, args, kwargs)
        self.loss_function = SmoothHinge(0.5, RLP(p=2, lamda=))


    def predict(self, x, beta=None):
        """predict binary classification {-1, 1}

        Args
            x (ndarray): nXm array of input samples
            beta (ndarray) - optional: mX1 of weight coefficients

        Returns
            [{-1, 1}]: list containing n predictions
        """
        if beta is None:
            beta = self.coef_

        return [1 if xi @ beta.T > 0 else -1 for xi in x]

    def predict_proba(self, x, beta=None):
        """prediction probabilities

        Args
            x (ndarray): nXm array of input samples
            beta (ndarray) - optional: nXm array of weight coefficients

        Returns
            [float]: list of n probabilities
        """
        if beta is None:
            beta = self.coef_

        return x@beta

    def _compute_grad(self, beta):
