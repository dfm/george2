# -*- coding: utf-8 -*-

__all__ = ["GP"]

import torch
import numpy as np


class GP(torch.nn.Module):

    def __init__(self, kernel, mean=None, jitter=None):
        super(GP, self).__init__()
        self.kernel = kernel

        noop = lambda x: torch.zeros(x.size(0), dtype=x.dtype, device=x.device)  # NOQA
        self.mean = noop if mean is None else mean
        self.jitter = noop if jitter is None else jitter

    def compute(self, x, yerr=None):
        self.x = x
        if yerr is None:
            yerr = torch.zeros(
                x.size(0), dtype=self.x.dtype, device=self.x.device)
        self.yvar = yerr**2
        self._compute_factor()

    def _compute_factor(self):
        K = self.kernel(self.x, self.x) + torch.diag(self.yvar)
        self.cholesky_factor = torch.cholesky(K, upper=False)
        self.half_log_det = torch.sum(torch.log(torch.diag(
            self.cholesky_factor)))
        self.normalization = 0.5 * self.x.size(0) * np.log(2*np.pi)

    def resid(self, y, x=None):
        if x is None:
            x = self.x
        return y - self.mean(x)

    def log_likelihood(self, y):
        resid = self.resid(y, x=self.x)
        soln = torch.triangular_solve(resid.unsqueeze(-1),
                                      self.cholesky_factor,
                                      upper=False)
        loglike = -0.5*torch.norm(soln.solution)
        loglike += self.half_log_det
        loglike += self.normalization
        return loglike
