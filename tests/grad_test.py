import pytest

import copy
import operator

import torch
from torch import fx
from torch.fx.passes.shape_prop import ShapeProp

from opt_einsum_fx.grad import grad


@pytest.mark.parametrize(
    "func,args",
    [
        (lambda x: 2.0 * x, [torch.randn(7, 3)]),
        (lambda x: x / 4.77, [torch.randn(2, 3)]),
        (
            lambda x, y: torch.einsum("ij,jk->ik", x, y),
            [torch.randn(4, 3), torch.randn(3, 6)],
        ),
        (
            lambda x, y: torch.einsum("ij,mn->mn", x, y),
            [torch.randn(4, 5), torch.randn(3, 6)],
        ),
        (lambda x: 4.0 * torch.einsum("ij,kj->ik", x, x), [torch.randn(15, 17)]),
    ],
)
def test_grad_numeric(func, args):
    args = [arg.to(torch.get_default_dtype()) for arg in args]
    graph: fx.Graph = fx.symbolic_trace(func).graph
    # make gradient
    for arg_i, arg in enumerate(args):
        arg.requires_grad_(True)
        true_out = func(*args)
        true_grad = torch.autograd.grad(true_out.sum(), arg)[0]
        arg.requires_grad_(False)
        grad_graph: fx.Graph = copy.deepcopy(graph)
        # remove output
        outnode: fx.Node = None
        for node in grad_graph.nodes:
            if node.op == "output":
                outnode = node.args[0]
                grad_graph.erase_node(node)

        grad_out_node = grad_graph.call_function(
            torch.ones,
            args=(true_out.shape),
            kwargs={"dtype": torch.get_default_dtype()},
        )

        grad_node = grad(
            outnode,
            grad_out_node,
            wrt=[n for n in grad_graph.nodes if n.op == "placeholder"][arg_i],
        )
        grad_graph.output(grad_node)
        grad_module = fx.GraphModule({}, grad_graph)
        grad_fx = grad_module(*args)

        assert torch.allclose(true_grad, grad_fx)


def test_reshape():
    # make a graph
    graph: fx.Graph = fx.Graph()
    x = fx.Proxy(graph.placeholder("x"))
    xshape = x.shape[:-1]
    x = x.reshape(-1, 13)  # 13 is the fixed last dimension
    out = 2.0 * x
    out = out.reshape(xshape + (13,))
    # test grad
    gmod = fx.GraphModule({}, graph)
    sp = ShapeProp(gmod)
    x_real = torch.randn(17, 13)
    sp.run(x_real)
    grad_out_node = graph.call_function(
        torch.ones,
        args=((xshape + (13,)).node,),
        kwargs={"dtype": torch.get_default_dtype()},
    )
    grad_node = grad(out.node, grad_out_node, x.node)
    graph.output(grad_node, torch.Tensor)
    gmod.recompile()
    grad_pred = gmod(x_real)
    assert torch.allclose(grad_pred, torch.as_tensor(2.0))
    assert grad_pred.shape == x_real.shape
