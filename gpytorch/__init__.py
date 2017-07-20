import torch
from lazy import ToeplitzLazyVariable
from .distribution import Distribution
from .observation_model import ObservationModel
from .math.functions import AddDiag, ExactGPMarginalLogLikelihood, Invmm, \
    Invmv, NormalCDF, LogNormalCDF, MVNKLDivergence, ToeplitzMV

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
        if matrix.W is not None:
            # Get W^{T}v
            Wt_times_v = torch.dsmm(matrix.W.t(), vector)
            # Get (TW^{T})v
            TWt_v = ToeplitzMV()(matrix.c,matrix.r,W_times_v)
            # Get (WTW^{T})v
            WTWt_v = torch.dsmm(matrix.W, TWt_v)
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
        c_new = input.c.clone()
        r_new = input.r.clone()
        c_new[0] += diag
        r_new[0] += diag
        return ToeplitzLazyVariable(c_new, r_new, input.W)
    else:
        return AddDiag()(input, diag)


def exact_gp_marginal_log_likelihood(covar, target):
    return ExactGPMarginalLogLikelihood()(covar, target)


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
