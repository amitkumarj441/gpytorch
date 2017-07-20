import torch
from torch.autograd import Variable
import utils
from lazy import ToeplitzLazyVariable
from .distribution import Distribution
from .observation_model import ObservationModel
from .math.functions import AddDiag, ExactGPMarginalLogLikelihood, Invmm, \
    Invmv, NormalCDF, LogNormalCDF, MVNKLDivergence, ToeplitzMV

import pdb

__all__ = [
    ToeplitzLazyVariable,
    Distribution,
    ObservationModel,
    AddDiag,
    ExactGPMarginalLogLikelihood,
    Invmm,
    Invmv,
    NormalCDF,
    LogNormalCDF,
    MVNKLDivergence,
    ToeplitzMV,
]

def mv(matrix, vector):
    if isinstance(matrix, ToeplitzLazyVariable):
        if matrix.J_left is not None:
            W_left = Variable(index_coef_to_sparse(matrix.J_left, matrix.C_left))
            W_right = Variable(index_coef_to_sparse(matrix.J_right, matrix.C_right))
            # Get W_{r}^{T}v
            Wt_times_v = torch.dsmm(W_right.t(), vector)
            # Get (TW_{r}^{T})v
            TWt_v = ToeplitzMV()(matrix.c,matrix.r,W_times_v)
            # Get (W_{l}TW_{r}^{T})v
            WTWt_v = torch.dsmm(matrix.W_left, TWt_v)
            return WTWt_v
        else:
            # Get Tv
            return ToeplitzMV()(matrix.c, matrix.r, vector)
    else:
        return torch.mv(matrix, vector)


def mm(matrix, vector):
    return torch.mm(matrix, vector)


def add_diag(input, diag):
    if isinstance(input, ToeplitzLazyVariable):
        e1 = Variable(torch.eye(len(input.c))[0])
        c_new = input.c + e1.mul(diag.expand_as(e1))
        r_new = input.r + e1.mul(diag.expand_as(e1))
        return ToeplitzLazyVariable(c_new, r_new, input.J_left, input.C_left, input.J_right, input.C_right)
    else:
        return AddDiag()(input, diag)


def exact_gp_marginal_log_likelihood(covar, target):
    if isinstance(covar, ToeplitzLazyVariable):
        return ExactGPMarginalLogLikelihood(structure='toeplitz')(covar, target)


def invmm(mat1, mat2):
    return Invmm()(mat1, mat2)


def invmv(mat, vec):
    return Invmv()(mat, vec)


def normal_cdf(x):
    return NormalCDF()(x)


def log_normal_cdf(x):
    return LogNormalCDF()(x)


def mvn_kl_divergence(mean_1, chol_covar_1, mean_2, covar_2):
    return MVNKLDivergence()(mean_1, chol_covar_1, mean_2, covar_2)
