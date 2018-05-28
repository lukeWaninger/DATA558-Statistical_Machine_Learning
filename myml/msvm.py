from myml.my_classifier import MyClassifier
from myml.loss_functions import *


class MySVM(MyClassifier):
    """my linear support vector machine"""

    def __init__(self, x_train, y_train, parameters, *args, **kwargs):
        """initialize the classifier

        Args:
            x_train (ndarray): nXm array of training samples
            y_train (ndarray: nX1 array of labels
            *args:
            **kwargs:
        """
        super().__init__(x_train, y_train, parameters, args, kwargs)
        self.loss_function = SmoothHinge(0.5)

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
        return self.loss_function.gradient(
            self._x, self._y, self._n, self._d, beta, self._param('regularization'))

    def _objective(self, beta):
        return self.loss_function.objective(
            self._x, self._y, self._n, beta, self._param('regularization'))

    def __str__(self):
        p = str(self._param('regularization'))
        return f'SVM-{p}{str(self.loss_function)}'
