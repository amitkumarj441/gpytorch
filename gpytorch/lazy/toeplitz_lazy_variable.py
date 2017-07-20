from torch.autograd import Variable
from gpytorch import utils
from .lazy_variable import LazyVariable

class ToeplitzLazyVariable(LazyVariable):
    def __init__(self, c, r, W=None):
        if not isinstance(c, Variable) or not isinstance(r, Variable):
            raise RuntimeError('ToeplitzLazyVariable is intended to wrap Variable versions of the first column and row.')

        if W is not None and not isinstance(W, Variable):
            raise RuntimeError('ToeplitzLazyVariable is intended to wrap Variable versions of the first column and row.')

        self.c = c
        self.r = r
        self.W = W

    def evaluate():
        T = utils.toeplitz(self.c, self.r)
        if self.W is not None:
            return torch.dsmm(W, torch.dsmm(W, T).t())
        else:
            return T