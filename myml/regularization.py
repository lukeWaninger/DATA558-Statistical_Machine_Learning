from abc import ABC, abstractmethod
from enum import Enum
import numpy as np


class Regularizer(ABC):
    def __init__(self, lamda, p):
        self._lambda = lamda
        self._p = p

    @abstractmethod
    def dreg(self, betas, k=None):
        yield

    @abstractmethod
    def reg(self, betas, k=None):
        yield

    def set_lambda(self, l):
        self._lambda = l

    def __repr__(self):
        return f'<L{self._p} norm, lamda={self._lambda}>'


class RL1(Regularizer):
    def __init__(self, lamda=1.):
        super().__init__(lamda, p=1)
        self._p = 1
        raise NotImplementedError

    def dreg(self, betas, k=None):
        raise NotImplementedError

    def reg(self, betas, k=None):
        return self._lambda * np.sum(np.abs(betas))

    def __str__(self):
        return 'L1'


class RLP(Regularizer):
    def __init__(self, p, lamda=1.):
        super().__init__(lamda, p=2)

    def dreg(self, betas, k=None):
        p, l, norm = self._p, self._lambda, np.linalg.norm

        if k is not None:
            return p*l*(k@betas)**(p-1)
        else:
            return p*l*norm(betas)**(p-1)

    def reg(self, betas, k=None):
        p, l, norm = self._p, self._lambda, np.linalg.norm

        if k is not None:
            return l*(betas @ k @ betas)
        else:
            return p*l*norm(betas)**(p-1)

    def __str__(self):
        p = str(self._p)
        return f'L{p} l={self._lambda}'


class REGULARIZATION_METHODS(Enum):
    L1 = RL1
    LP = RLP


REGULARIZATION_DISPATCH = {
    'L1': REGULARIZATION_METHODS.L1,
    'LP': REGULARIZATION_METHODS.LP,
    'none': None
}
