from torch.autograd import Variable
import torch
from gpytorch import utils
from .lazy_variable import LazyVariable

import pdb

class ToeplitzLazyVariable(LazyVariable):
    def __init__(self, c, r, J_left=None, C_left=None, J_right=None, C_right=None):
        if not isinstance(c, Variable) or not isinstance(r, Variable):
            raise RuntimeError('ToeplitzLazyVariable is intended to wrap Variable versions of \
                                the first column and row.')

        self.c = c
        self.r = r
        self.J_left = J_left
        self.C_left = C_left
        self.J_right = J_right
        self.C_right = C_right

    def evaluate(self):
        """
        Explicitly evaluate and return the Toeplitz matrix this object wraps as a float Tensor.
        To do this, we explicitly compute W_{left}TW_{right}^{T} and return it.

        Warning: as implicitly stored by this LazyVariable, W is very sparse and T requires O(n)
        storage, where as the full matrix requires O(n^2) storage. Calling evaluate can very easily
        lead to memory issues. As a result, using it should be a last resort.
        """
        T = utils.toeplitz(self.c, self.r)
        if self.J_left is not None:
            W_left = Variable(utils.index_coef_to_sparse(self.J_left, self.C_left))
            W_right = Variable(utils.index_coef_to_sparse(self.J_right, self.C_right))
            return torch.dsmm(W_right, torch.dsmm(W_left, T).t())
        else:
            return T

    def diag(self):
        """
        Gets the diagonal of the Toeplitz matrix wrapped by this object.

        By definition of a Toeplitz matrix, every element along the diagonal is equal
        to c[0] == r[0]. Therefore, we return a vector of length len(self.c) with
        each element equal to c[0].

        If the interpolation matrices exist, then the diagonal of WTW^{T} is simply
        W(T_diag)W^{T}.
        """
        if self.J_left is not None:
            if len(self.J_left) != len(self.J_right):
                raise RuntimeError('diag not supported for non-square interpolated Toeplitz matrices.')
            return self.c[0].expand(len(self.J_left))
        else:
            return self.c[0].expand_as(self.c)

    def __getitem__(self, i):
        if isinstance(i, tuple):
            if self.J_left is not None:
                # J[i[0], :], C[i[0], :]
                J_left_new = self.J_left[i[0]]
                C_left_new = self.C_left[i[0]]

                # J[i[1], :] C[i[1], :]
                J_right_new = self.J_right[i[1]]
                C_right_new = self.C_right[i[1]]
                return ToeplitzLazyVariable(self.c, self.r, J_left_new, C_left_new, J_right_new, C_right_new)
            else:
                r_new = self.r[i[0]]
                c_new = self.c[i[1]]
                if len(r_new) != len(c_new):
                    raise RuntimeError('Slicing an uninterpolated Toeplitz matrix to be non-square is probably \
                                        unintended. If that was the intent, use evaluate() and slice the full matrix.')
                return ToeplitzLazyVariable(c_new, r_new)
        else:
            if self.J_left is not None:
                J_left_new = self.J_left[i]
                C_left_new = self.C_left[i]
                return ToeplitzLazyVariable(self.c, self.r, J_left_new, C_left_new, self.J_right, self.C_right)
            else:
                raise RuntimeError('Slicing an uninterpolated Toeplitz matrix to be non-square is probably unintended. \
                                    If that was the intent, use evaluate() and slice the full matrix.')
