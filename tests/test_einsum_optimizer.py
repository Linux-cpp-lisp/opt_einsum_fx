import warnings
import pytest

import torch
import torch.fx
from torch.fx.passes.shape_prop import ShapeProp

from opt_einsum_fx import optimize_einsums, optimize_einsums_graph, jitable


def einmatmul(x, y):
    return torch.einsum("ij,jk->ik", x, y)


def test_optimize_einsums_graph(allclose):
    x = torch.randn(3, 4)
    y = torch.randn(4, 5)

    func_res = einmatmul(x, y)

    func_fx = torch.fx.symbolic_trace(einmatmul)
    sp = ShapeProp(func_fx)
    sp.run(x, y)

    func_fx_res = func_fx(x, y)
    assert torch.all(func_res == func_fx_res)

    graph_opt = optimize_einsums_graph(func_fx.graph)
    func_fx.graph = graph_opt
    func_fx.recompile()

    func_opt_res = func_fx(x, y)
    assert allclose(func_opt_res, func_fx_res)


def test_fallback():
    # If there is no shape propagation, it should warn
    # and not do anything.
    func_fx = torch.fx.symbolic_trace(einmatmul)
    old_code = func_fx.code

    with pytest.warns(RuntimeWarning):
        graph_opt = optimize_einsums_graph(func_fx.graph)

    func_fx.graph = graph_opt
    func_fx.recompile()
    assert old_code == func_fx.code


def test_torchscript(allclose):
    x = torch.randn(3, 4)
    y = torch.randn(4, 5)
    func_res = einmatmul(x, y)
    mod_opt = optimize_einsums(einmatmul, (x, y))
    mod_opt = jitable(mod_opt)
    mod_opt = torch.jit.script(mod_opt)
    func_opt_res = mod_opt(x, y)
    assert allclose(func_opt_res, func_res)
