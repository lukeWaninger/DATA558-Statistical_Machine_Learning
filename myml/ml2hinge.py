from myml.my_classifier import MyClassifier
import numpy as np


class MyL2Hinge(MyClassifier):
    def __init__(self, x_train, y_train, parameters, *args, **kwargs):
        super().__init__(x_train, y_train, parameters, args, kwargs)

    def predict(self, x, beta=None):
        if beta is None:
            beta = self.coef_

        return [1 if xi @ beta.T > .5 else -1 for xi in x]

    def predict_proba(self, x, beta=None):
        return x@beta

    def _objective(self, beta):
        x, y, l, n = self._x, self._y, self._param('lambda'), self._n

        yx  = y[:, np.newaxis] * x
        yxb = yx @ beta
        return 1./n * (np.sum(np.maximum(0, 1 - yxb)**2)) + l*np.linalg.norm(beta)**2

    def _compute_grad(self, beta):
        x, y, n = self._x, self._y, self._n,
        reg = 2*self._param('lambda')*beta

        yx   = y[:, np.newaxis]*x
        yxb  = yx@beta
        loss = np.maximum(0, 1-yxb)
        return -2./n * (loss @ yx) + reg
