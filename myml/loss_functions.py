from abc import ABC, abstractmethod
from enum import Enum
import numpy as np


class LossFunction(ABC):
    def __init__(self, regularizer):
        self._regularizer = regularizer

    @abstractmethod
    def objective(self, x, y, n, beta):
        yield

    @abstractmethod
    def gradient(self, x, y, n, d, beta):
        yield


class SmoothHinge(LossFunction):
    def __init__(self, h, regularizer=None):
        super().__init__(regularizer)
        self.__h = h

    def objective(self, x, y, n, beta):
        h = self.__h

        loss = np.zeros(n)
        yx = y * (x @ beta)
        mask = np.abs(1 - yx)

        loss[mask <= h] = ((1 + h - yx) ** 2 / (4 * h))[mask <= h]
        loss[yx < 1 - h] = (1 - yx)[yx < 1 - h]

        return np.sum(loss) / n + self._regularizer.reg()

    def gradient(self, x, y, n, d, beta):
        h, na = self.__h, np.newaxis
        lg = np.zeros([n, d])
        yx = y * x.dot(beta)
        mask = abs(1 - yx)

        lg[mask <= h] = ((1/(2*h)) * ((1 + h - yx)[:, na]) * (-y[:, na] * x))[mask <= h]
        lg[yx < 1 - h] = (-y[:, na] * x)[yx < 1 - h]

        return np.array(np.sum(lg, axis=0) / n + self._regularizer.dreg())

    def __str__(self):
        return f'smooth hinge-{str(self._regularizer)}'


class SquaredHinge(LossFunction):
    def __init__(self, regularizer=None):
        super().__init__(regularizer)

    def objective(self, x, y, n, beta):
        yx = y[:, np.newaxis] * x
        return 1. / n * (np.sum(np.maximum(0, 1 - yx @ beta) ** 2)) + self._regularizer().reg()

    def gradient(self, x, y, n, d, beta):
        yx = y[:, np.newaxis] * x
        return -2. / n * (np.maximum(0, 1 - yx @ beta) @ yx) + self._regularizer().dreg()

    def __str__(self):
        return f'squared hinged-{str(self._regularizer)}'
