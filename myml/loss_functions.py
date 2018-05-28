from abc import ABC, abstractmethod
import numpy as np


class LossFunction(ABC):
    @abstractmethod
    def objective(self, x, y, n, beta, regularization=None):
        yield

    @abstractmethod
    def gradient(self, x, y, n, d, beta, regularization=None):
        yield


class SmoothHinge(LossFunction):
    def __init__(self, h):
        super().__init__()
        self.__h = h

    def objective(self, x, y, n, beta, regularization=None):
        h = self.__h

        loss = np.zeros(n)
        yx = y * (x @ beta)
        mask = np.abs(1 - yx)

        loss[mask <= h] = ((1 + h - yx) ** 2 / (4 * h))[mask <= h]
        loss[yx < 1 - h] = (1 - yx)[yx < 1 - h]

        return np.sum(loss) / n + regularization.reg(beta)

    def gradient(self, x, y, n, d, beta, regularization=None):
        h, na = self.__h, np.newaxis
        lg = np.zeros([n, d])
        yx = y * x.dot(beta)
        mask = abs(1 - yx)

        lg[mask <= h] = ((1/(2*h)) * ((1 + h - yx)[:, na]) * (-y[:, na] * x))[mask <= h]
        lg[yx < 1 - h] = (-y[:, na] * x)[yx < 1 - h]

        return np.array(np.sum(lg, axis=0) / n + regularization.dreg(beta))

    def __str__(self):
        return f'smooth hinge'


class SquaredHinge(LossFunction):
    def __init__(self):
        super().__init__()

    def objective(self, x, y, n, beta, regularization=None):
        yx = y[:, np.newaxis] * x
        return 1. / n * (np.sum(np.maximum(0, 1 - yx @ beta) ** 2)) + regularization.reg(beta)

    def gradient(self, x, y, n, d, beta, regularization=None):
        yx = y[:, np.newaxis] * x
        return -2. / n * (np.maximum(0, 1 - yx @ beta) @ yx) + regularization.dreg(beta)

    def __str__(self):
        return f'squared hinged'
