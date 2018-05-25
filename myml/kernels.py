from abc import ABC, abstractmethod
from enum import Enum
import numpy as np
from sklearn.metrics.pairwise import pairwise_distances


class Kernel(ABC):
    def __init__(self):
        self._gram = None

    @abstractmethod
    def compute(self, x, xp=None):
        yield

    @property
    def gram(self):
        return self._gram


class KHellinger(Kernel):
    def __init__(self):
        super().__init__()

    def compute(self, x, xp=None):
        xp = x if xp is None else xp

        self._gram = np.sqrt(x @ xp.T)
        return self

    def __str__(self):
        return 'hellinger'


class KLinear(Kernel):
    def __init__(self):
        super().__init__()

    def compute(self, x, xp=None):
        xp = x if xp is None else xp

        self._gram = x @ xp.T
        return self

    def __str__(self):
        return 'linear'


class KPolynomial(Kernel):
    def __init__(self, degree, c=1):
        super().__init__()

        self.__degree = degree
        self.__c = c

    def compute(self, x, xp=None):
        xp = x if xp is None else xp

        self._gram = (x @ xp.T + self.__c) ** self.__degree
        return self

    def __str__(self):
        return 'polynomial'


class KGausianRBF(Kernel):
    def __init__(self, sigma=None, x=None):
        super().__init__()

        if sigma is not None:
            self.__sigma = sigma

        elif sigma is None and x is not None:
            dists = pairwise_distances(x).reshape(-1)
            self.__sigma = np.median(dists)

        else:
            raise ValueError('sigma must be specified')

    def compute(self, x, xp=None):
        exp, na = np.exp, np.newaxis

        def norm(mat):
            return np.linalg.norm(mat, axis=1)

        xp = x if xp is None else xp
        s = self.__sigma

        self._gram = exp(-1/(2*s**2) * ((norm(x)**2)[:, na] + (norm(xp)**2)[na, :] - 2*(x@xp.T)))

    def __str__(self):
        return 'gaussian-rbf'


class KSigmoid(Kernel):
    def __init__(self, alpha, beta):
        super().__init__()

        self.__alpha = alpha
        self.__beta  = beta

    def compute(self, x, xp=None):
        xp = x if xp is None else xp
        a, b = self.__alpha, self.__beta

        return np.tanh(a * (x @ xp.T) + b)

    def __str__(self):
        return 'sigmoid'


class KERNEL(Enum):
    HELLINGER  = KHellinger,
    LINEAR     = KLinear,
    POLYNOMIAL = KPolynomial,
    GAUSSIAN   = KGausianRBF,
    SIGMOID    = KSigmoid


KERNEL_DISPATCH = {
    'hellinger':  KERNEL.HELLINGER,
    'linear':     KERNEL.LINEAR,
    'polynomial': KERNEL.POLYNOMIAL,
    'gaussian':   KERNEL.GAUSSIAN,
    'sigmoid':    KERNEL.SIGMOID,
    'none':       None
}
