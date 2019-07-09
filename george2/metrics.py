# -*- coding: utf-8 -*-

__all__ = ["Metric", "DiagonalMetric", "DenseMetric"]

import torch


class Metric(torch.nn.Module):

    def forward(self, x):
        return torch.norm(x, dim=-1)


class DiagonalMetric(Metric):

    def __init__(self, var=None, scale=None):
        super(DiagonalMetric, self).__init__()
        if var is not None:
            self.var = torch.nn.Parameter(var)
            self.inv_var = 1.0 / self.var
        elif scale is not None:
            self.scale = torch.nn.Parameter(scale)
            self.inv_var = 1.0 / self.scale**2
        else:
            raise ValueError("either 'var' or 'scale' must be defined")

    def forward(self, x):
        alpha = torch.mul(self.inv_var, x)
        return torch.mul(x, alpha).sum(-1)


class DenseMetric(Metric):

    def __init__(self, cov=None, L=None):
        super(DenseMetric, self).__init__()
        if cov is not None:
            self.cov = torch.nn.Parameter(cov)
            self.L = torch.cholesky(cov, upper=False)
        elif L is not None:
            self.L = torch.nn.Parameter(L)
        else:
            raise ValueError("either 'cov' or 'L' must be defined")

    def forward(self, x):
        soln = torch.triangular_solve(x.unsqueeze(-1), self.L, upper=False)
        alpha = soln.solution.squeeze(-1)
        return torch.mul(alpha, alpha).sum(-1)
