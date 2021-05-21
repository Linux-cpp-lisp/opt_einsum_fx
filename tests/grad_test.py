import pytest

import copy
import operator

import torch
from torch import fx

from opt_einsum_fx.fx_utils import copy_with_attributes, find_in_graph_copy
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

        assert true_grad.shape == grad_fx.shape
        assert torch.allclose(true_grad, grad_fx)


@pytest.mark.parametrize("do_copy", [True, False])
def test_reshape(do_copy):
    # make a graph
    graph: fx.Graph = fx.Graph()
    x = graph.placeholder("x")
    x2 = graph.call_method("reshape", args=(x, (-1, 8, 2, 2)))
    x2.in_shape = (-1, 32)
    out = graph.call_function(operator.mul, args=(2.0, x2))
    out = graph.call_method("reshape", args=(out, (-1, 2, 16)))
    out.in_shape = (-1, 8, 2, 2)
    xshape = graph.call_function(getattr, args=(x, "shape"))
    # test grad
    x_real = torch.randn(17, 32)

    if do_copy:
        graph = copy_with_attributes(graph, attributes=["in_shape"])
        x, out, xshape = find_in_graph_copy(graph, [x, out, xshape])

    xshape = fx.Proxy(xshape)
    grad_out_node = graph.call_function(
        torch.ones,
        args=((xshape[0].node, 2, 16),),
        kwargs={"dtype": torch.get_default_dtype()},
    )
    grad_node = grad(out, grad_out_node, x)
    graph.output(grad_node, torch.Tensor)
    gmod = fx.GraphModule({}, graph)
    grad_pred = gmod(x_real)
    assert torch.allclose(grad_pred, torch.as_tensor(2.0))
    assert grad_pred.shape == x_real.shape
