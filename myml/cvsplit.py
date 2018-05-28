from myml.kernels import *
from myml.metrics import MetricSet


class TrainingSplit(object):
    def __init__(self, n=0, d=0, train_idx=None, val_idx=None, parameters=None, betas=None):
        self.n = n
        self.d = d
        self.train_idx = train_idx
        self.val_idx = val_idx
        self.parameters = parameters if parameters is not None else {}
        self.kernel = self.__set_kernel() if self.parameters is not None else None
        self.betas = betas if betas is not None else []
        self.train_metrics = None
        self.val_metrics = None

    @property
    def dict(self):
        return {
            'n': self.n,
            'd': self.d,
            'train_idx': self.train_idx if not None else [],
            'val_idx': self.val_idx if not None else [],
            'parameters': self.parameters,
            'betas': self.betas,
            'train_metrics': self.train_metrics.dict if self.train_metrics is not None else 'none',
            'val_metrics': self.val_metrics.dict if self.val_metrics is not None else 'none'
        }

    @property
    def has_kernel(self):
        return self.kernel is not None

    def from_dict(self, data):
        if data == 'none':
            return None

        self.betas = data['betas']
        self.d = data['d']
        self.n = data['n']
        self.train_idx = data['train_idx']
        self.val_idx = data['val_idx']
        self.parameters = data['parameters']
        self.kernel = self.__set_kernel()
        self.train_metrics = MetricSet(data=data['train_metrics'])
        self.val_metrics = MetricSet(data=data['val_metrics'])

        return self

    def get_param(self, param):
        try:
            return getattr(self, param)
        except AttributeError:
            try:
                return self.parameters[param]
            except KeyError:
                raise AttributeError('attribute or key does not exist')

    def set_param(self, param, value):
        try:
            setattr(self, param, value)
        except AttributeError as e:
            if param not in self.parameters.keys():
                print([str(a) for a in e.args])
                raise
            else:
                self.parameters['param'] = value

    def __set_kernel(self):
        try:
            method = self.get_param('kernel')
            return KERNEL_DISPATCH[method] if method is not None else None
        except AttributeError:
            return None
        except KeyError:
            raise KeyError('kernel method does not exist')
