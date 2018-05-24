from myml.my_classifier import MyClassifier
import numpy as np


class MyLinearSVM(MyClassifier):
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

    def _objective(self, beta):
        """linear svm objective function

        Args
           beta (ndarray): mx1 array of weight coefficients being optimized

        Returns
            (float): objective value of defined loss function

        Raises
            ValueError: if loss function is not defined
        """
        loss_function = self._param('loss')
        if loss_function == 'squared_hinge':
            return self.__squared_hinge_loss(beta)

        elif loss_function == 'smoothed_hinge':
            return self.__smoothed_hinge_loss(beta)

        else:
            raise ValueError('loss function not defined')

    def _compute_grad(self, beta):
        """compute gradient

        Args
            beta (ndarray): mX1 array of weight coefficients

        Returns
            (ndarray): mX1 array of gradients

        Raises
            ValueError: if loss function is not defined
        """
        loss_function = self._param('loss')
        if loss_function == 'squared_hinge':
            return self.__squared_hinge_gradient(beta)

        elif loss_function == 'smoothed_hinge':
            return self.__smoothed_hinge_gradient(beta)

        else:
            raise ValueError('loss function not defined')

    def __smoothed_hinge_loss(self, beta):
        """smoothed hinge loss

        Args
            beta (ndarray): mX1 array of weight coefficients

        Returns
            float: objective value
        """
        x, y, l, n = self._x, self._y, self._param('lambda'), self._n
        reg = l*np.linalg.norm(beta)**2
        h = self._param('h')

        loss = np.zeros(n)
        yx = y * (x@beta)
        mask = np.abs(1 - yx)

        loss[mask <= h] = ((1 + h - yx) ** 2 / (4 * h))[mask <= h]
        loss[yx < 1 - h] = (1 - yx)[yx < 1 - h]

        return np.sum(loss) / n + reg

    def __smoothed_hinge_gradient(self, beta):
        """smoothed hinge gradient

       Args
           beta (ndarray): mX1 array of weight coefficients

       Returns
           ndarray: mX1 array of gradient values
       """
        x, y, n, d, h = self._x, self._y, self._n, self._d, self._param('h')
        reg = 2 * self._param('lambda') * beta

        lg = np.zeros([n, d])
        yx = y * x.dot(beta)
        mask = abs(1 - yx)

        lg[mask <= h] = ((2 / (4 * h)) * ((1 + h - yx)[:, np.newaxis]) * (-y[:, np.newaxis] * x))[mask <= h]
        lg[yx < 1 - h] = (-y[:, np.newaxis] * x)[yx < 1 - h]

        return np.array(np.sum(lg, axis=0)/n + reg)

    def __squared_hinge_gradient(self, beta):
        """compute gradient of squared hinge loss

        Args
            beta (ndarray): mX1 array of weight coefficients

        Returns
            (ndarray): mX1 array of gradients
        """
        x, y, n = self._x, self._y, self._n,
        reg = 2 * self._param('lambda') * beta

        yx = y[:, np.newaxis] * x
        return -2. / n * (np.maximum(0, 1 - yx @ beta) @ yx) + reg

    def __squared_hinge_loss(self, beta):
        """squared hinge loss

        Args
            beta (ndarray): mX1 array of weight coefficients

        Returns
            float: objective value
        """
        x, y, l, n = self._x, self._y, self._param('lambda'), self._n
        reg = l * np.linalg.norm(beta) ** 2

        yx = y[:, np.newaxis] * x
        return 1. / n * (np.sum(np.maximum(0, 1 - yx @ beta) ** 2)) + reg