from abc import ABC, abstractmethod
from enum import Enum
import numpy as np


class Regularizer(ABC):
    def __init__(self, lamda):
        self._lambda = lamda

    @abstractmethod
    def dreg(self, betas, k=None):
        yield

    @abstractmethod
    def reg(self, betas, k=None):
        yield

    def set_lambda(self, l):
        self._lambda = l


class RL1(Regularizer):
    def __init__(self, lamda=1.):
        super().__init__(lamda)
        raise NotImplementedError

    def dreg(self, betas, k=None):
        raise NotImplementedError

    def reg(self, betas, k=None):
        return self._lambda * np.sum(np.abs(betas))

    def __str__(self):
        return 'L1'


class RLP(Regularizer):
    def __init__(self, p, lamda=1.):
        super().__init__(lamda)

        self.__p = p

    def dreg(self, betas, k=None):
        p, l, norm = self.__p, self._lambda, np.linalg.norm

        if k is not None:
            return p*l*(k@betas)**(p-1)
        else:
            return p*l*norm(betas)**(p-1)

    def reg(self, betas, k=None):
        p, l, norm = self.__p, self._lambda, np.linalg.norm

        if k is not None:
            return l*(betas @ k @ betas)
        else:
            return p*l*norm(betas)**(p-1)

    def __str__(self):
        return 'LP'


class REGULARIZATION_METHODS(Enum):
    L1 = RL1
    LP = RLP


REGULARIZATION_DISPATCH = {
    'L1': REGULARIZATION_METHODS.L1,
    'LP': REGULARIZATION_METHODS.LP,
    'none': None
}
