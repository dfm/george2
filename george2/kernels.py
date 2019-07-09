# -*- coding: utf-8 -*-

__all__ = [
    "Kernel",
    "Sum", "Product",
    "ExpSquared",
]

import torch


def _kernel_or_constant(a, other=None):
    if torch.is_tensor(a):
        return Constant(a)
    try:
        a = float(a)
    except TypeError:
        return a

    try:
        dtype = other.dtype
    except AttributeError:
        dtype = None

    try:
        device = other.device
    except AttributeError:
        device = None

    return Constant(torch.tensor(a, dtype=dtype, device=device))


def _apply_binary_op(op, a, b):
    a = _kernel_or_constant(a, other=b)
    b = _kernel_or_constant(b, other=a)
    return op(a, b)


class Kernel(torch.nn.Module):

    def __add__(self, b):
        return _apply_binary_op(Sum, self, b)

    def __radd__(self, b):
        return _apply_binary_op(Sum, b, self)

    def __mul__(self, b):
        return _apply_binary_op(Product, self, b)

    def __rmul__(self, b):
        return _apply_binary_op(Product, b, self)

    def forward(self, x1, x2, diag=False):
        raise NotImplementedError("subclasses should implement 'value'")

    @property
    def device(self):
        devices = list(set(p.device for p in self.parameters()))
        if len(devices) > 1:
            raise RuntimeError("inconsistent devices")
        if not len(devices):
            return None
        return devices[0]

    @property
    def dtype(self):
        dtypes = list(set(p.dtype for p in self.parameters()))
        if len(dtypes) > 1:
            raise RuntimeError("inconsistent dtypes")
        if not len(dtypes):
            return None
        return dtypes[0]


class Operator(Kernel):

    op = None

    def __init__(self, *kernels):
        super(Operator, self).__init__()
        if not len(kernels):
            raise ValueError("at least one kernel is required for an operator")
        self.kernels = torch.nn.ModuleList(kernels)

    def forward(self, x1, x2, diag=False):
        result = self.kernels[0](x1, x2, diag=diag)
        for k in self.kernels[1:]:
            if isinstance(k, Constant):
                result = self.op(result, k.value)
            else:
                result = self.op(result, k(x1, x2, diag=diag))
        return result


class Sum(Operator):
    op = torch.add


class Product(Operator):
    op = torch.mul


class Constant(Kernel):

    def __init__(self, value):
        super(Constant, self).__init__()
        self.value = torch.nn.Parameter(value)

    def forward(self, x1, x2, diag=False):
        if diag:
            shape = min((x1.size(0), x2.size(0)))
        else:
            shape = (x1.size(0), x2.size(0))
        result = torch.empty(shape, dtype=self.value.dtype,
                             device=self.value.device)
        result.fill_(self.value)
        return result


class Stationary(Kernel):

    def __init__(self, metric):
        super(Stationary, self).__init__()
        self.metric = metric

    def value(self, r2):
        raise NotImplementedError("subclasses should implement 'value'")

    def forward(self, x1, x2, diag=False):
        if diag:
            n = min((x1.size(0), x2.size(0)))
            r2 = self.metric(x1[:n, :] - x2[:n, :])
        else:
            r2 = self.metric(x1[:, None, :] - x2[None, :, :])
        return self.value(r2)


class ExpSquared(Stationary):

    def value(self, r2):
        return torch.exp(-0.5 * r2)
