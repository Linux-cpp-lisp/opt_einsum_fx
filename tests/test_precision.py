import pytest

import torch
import torch.fx

from opt_einsum_fx import cast_dtypes, optimize_einsums_full


def test_precision(allclose):
    x = torch.randn(3, 4)
    y = torch.randn(4, 5)
    z = torch.randn(5, 8)

    def einfunc(x, y, z):
        return torch.einsum("ij,jk,kl->il", x, y, z)

    func_res = einfunc(x, y, z)

    func_opt = optimize_einsums_full(einfunc, (x, y, z))
    func_opt = cast_dtypes(func_opt, example_inputs=(x, y.double(), z))

    func_opt_out = func_opt(x, y.double(), z)

    assert allclose(func_res, func_opt_out)
