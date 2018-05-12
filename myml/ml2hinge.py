from myml.my_classifier import MyClassifier
import numpy as np


class MyL2Hinge(MyClassifier):
    def __init__(self, x_train, y_train, parameters, *args, **kwargs):
        super().__init__(x_train, y_train, parameters, args, kwargs)

    def predict(self, x, beta=None):
        """ predict binary classification {-1, 1} for given sample set

        :param x: nXm ndarray, input samples
        :param beta: mX1 ndarray (optional), weight coefficients. if not provided
                     method will use final training betas
        :return: list, containing predictions {-1, 1}
        """
        if beta is None:
            beta = self.coef_

        return [1 if xi @ beta.T > .5 else -1 for xi in x]

    def predict_proba(self, x, beta=None):
        """ give probabilities that a given sample should receive a positive
        classification

        :param x: nXm ndarray, input samples
        :param beta: mX1 ndarray (optional), weight coefficients. if not provided
                     method will use final training betas
        :return: list containing predictions {-1, 1}
        """
        return x@beta

    def _objective(self, beta):
        """ linear svm objective function

        :param beta: mX1 ndarray, weight coefficients being optimized
        :return: float, objective value.
        """
        x, y, l, n = self._x, self._y, self._param('lambda'), self._n

        yx  = y[:, np.newaxis] * x
        return 1./n * (np.sum(np.maximum(0, 1 - yx@beta)**2)) + l*np.linalg.norm(beta)**2

    def _compute_grad(self, beta):
        """ compute gradient

        :param beta: mX1 ndarray, weight coefficients being optimized
        :return: mX1 ndarray, gradient to applied to betas
        """
        x, y, n = self._x, self._y, self._n,
        reg = 2*self._param('lambda')*beta

        yx   = y[:, np.newaxis]*x
        return -2./n * (np.maximum(0, 1-yx@beta) @ yx) + reg
