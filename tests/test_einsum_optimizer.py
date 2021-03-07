import warnings
import pytest

import torch
import torch.fx
from torch.fx.passes.shape_prop import ShapeProp

from opt_einsum_fx import optimize_einsums


def einmatmul(x, y):
    return torch.einsum("ij,jk->ik", x, y)

def test_optimize_einsums():
    x = torch.randn(3, 4)
    y = torch.randn(4, 5)

    func_res = einmatmul(x, y)

    func_fx = torch.fx.symbolic_trace(einmatmul)
    sp = ShapeProp(func_fx)
    sp.run(x, y)

    func_fx_res = func_fx(x, y)
    assert torch.all(func_res == func_fx_res)

    graph_opt = optimize_einsums(func_fx.graph)
    func_fx.graph = graph_opt
    func_fx.recompile()

    func_opt_res = func_fx(x, y)
    assert torch.allclose(func_opt_res, func_fx_res)


def test_fallback():
    # If there is no shape propagation, it should warn
    # and not do anything.
    func_fx = torch.fx.symbolic_trace(einmatmul)
    old_code = func_fx.code

    with pytest.warns(RuntimeWarning):
        graph_opt = optimize_einsums(func_fx.graph)

    func_fx.graph = graph_opt
    func_fx.recompile()
    assert old_code == func_fx.code
