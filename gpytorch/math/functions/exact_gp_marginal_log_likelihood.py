import math
import torch
from gpytorch.utils import LinearCG, SLQLogDet
from torch.autograd import Function


class ExactGPMarginalLogLikelihood(Function):
    def __init__(self, structure=None):
        self.structure = structure

    def forward(self, *inputs):
        if self.structure == 'toeplitz':
            c, r, y = inputs
            mv_closure

        mat_inv_y = LinearCG().solve(matrix, y)
        # Inverse quad form
        res = mat_inv_y.dot(y)
        # Log determinant
        res += SLQLogDet(num_random_probes=10).logdet(matrix)
        res += math.log(2 * math.pi) * len(y)
        res *= -0.5

        self.save_for_backward(matrix, y)
        self.mat_inv_y = mat_inv_y
        return matrix.new().resize_(1).fill_(res)

    def backward(self, grad_output):
        grad_output_value = grad_output.squeeze()[0]
        matrix, y = self.saved_tensors
        mat_inv_y = self.mat_inv_y

        mat_grad = None
        y_grad = None

        if self.needs_input_grad[0]:
            mat_grad = torch.ger(y.view(-1), mat_inv_y.view(-1))
            mat_grad.add_(-torch.eye(*mat_grad.size()))
            mat_grad = LinearCG().solve(matrix, mat_grad)
            mat_grad.mul_(0.5 * grad_output_value)

        if self.needs_input_grad[1]:
            y_grad = mat_inv_y.mul_(-grad_output_value)

        return mat_grad, y_grad
