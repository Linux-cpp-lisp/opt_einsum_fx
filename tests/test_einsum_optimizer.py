import torch
import torch.fx
from torch.fx.passes.shape_prop import ShapeProp

from opt_einsum_fx import optimize_einsums


def test_optimize_einsums():
    def func(x, y):
        return torch.einsum("ij,jk->ik", x, y)

    x = torch.randn(3, 4)
    y = torch.randn(4, 5)

    func_res = func(x, y)

    func_fx = torch.fx.symbolic_trace(func)
    sp = ShapeProp(func_fx)
    sp.run(x, y)

    func_fx_res = func_fx(x, y)
    assert torch.all(func_res == func_fx_res)

    graph_opt = optimize_einsums(func_fx.graph)
    func_fx.graph = graph_opt
    func_fx.recompile()

    func_opt_res = func_fx(x, y)
    assert torch.allclose(func_opt_res, func_fx_res)
