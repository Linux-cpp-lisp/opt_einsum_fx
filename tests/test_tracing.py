import pytest

import torch

functorch = pytest.importorskip("functorch")

from opt_einsum_fx import optimize_einsums_full
from opt_einsum_fx.tracing import make_fx_dynamic


def explict_reshape(x):
    bdim, fdim = x.shape
    y = x.view(-1).tanh()
    return (y.view(bdim, fdim),)  # use a dim that will be traced as constant


def simple(x):
    # do nothing dynamic
    return (2.7 * x.square() * x.tanh() + 3.0,)


def grad(x):
    x = x.clone().requires_grad_(True)
    y = x.tanh().square().sum()
    grad = torch.autograd.grad([y], [x])[0]
    assert grad is not None  # torchscript
    return (grad,)


def einsum(x):
    # involve an explicit reshape to test
    # involve dynamic dimension in dim != 0 to test
    return (torch.einsum("zi,jz->zij", x, x.view(-1).reshape(x.shape[1], x.shape[0])),)


_einsum_opted = optimize_einsums_full(einsum, example_inputs=(torch.randn(11, 4),))


def einsum_opted(x):
    return _einsum_opted(x)


def grad_einsum_opt(x):
    x = x.clone().requires_grad_(True)
    y = einsum_opted(x)[0].sum()
    grad = torch.autograd.grad([y], [x])[0]
    return (grad,)


# For TorchScript, FX tracing requires that the top-level function be in Python
# So we do a trivial wrapper here
_simple_ts = torch.jit.script(simple)
_grad_ts = torch.jit.script(grad)
_ein_ts = torch.jit.script(einsum)
_ein_opt_ts = torch.jit.script(_einsum_opted)


def simple_ts(x):
    return _simple_ts(x)


def grad_ts(x):
    return _grad_ts(x)


def einsum_ts(x):
    return _ein_ts(x)


def einsum_opt_ts(x):
    return _ein_opt_ts(x)


# TODO test einsum
# TODO test einsum w grad

# TODO: generalize to more than one arg
@pytest.mark.parametrize(
    "func",
    [
        explict_reshape,
        simple,
        grad,
        simple_ts,
        grad_ts,
        einsum,
        einsum_opted,
        grad_einsum_opt,
        einsum_ts,
        einsum_opt_ts,
    ],
)
def test_trace_like_func(func):
    bdims_trace = [1, 3, 4]
    bdims_test = [2, 7, 5]
    trace_inputs = [(torch.randn(bdim, 5),) for bdim in bdims_trace]
    test_inputs = [(torch.randn(bdim, 5),) for bdim in bdims_test]
    # now trace out the function
    traced = make_fx_dynamic(func, example_inputs=trace_inputs)
    # test it on other dimensions
    truth_outs = [func(*e) for e in test_inputs]
    # running this checks that it can run with dynamic shapes
    traced_outs = [traced(*e) for e in test_inputs]
    for truth_out, traced_out in zip(truth_outs, traced_outs):
        for truth_tensor, traced_tensor in zip(truth_out, traced_out):
            assert truth_tensor.shape == traced_tensor.shape
            assert torch.allclose(truth_tensor, traced_tensor)


def test_no_dynamic_simple():
    func = simple
    bdims_trace = [1, 3, 4]
    bdims_test = [2, 7, 5]
    trace_inputs = [(torch.randn(bdim, 5),) for bdim in bdims_trace]
    test_inputs = [(torch.randn(bdim, 5),) for bdim in bdims_test]
    # now trace out the function
    traced = functorch.make_fx(func)(*trace_inputs[0])
    traced_dynamic = make_fx_dynamic(func, example_inputs=trace_inputs)
    # they should be the same, but with one dynamic shape extraction
    # which is one getattr and one getitem
    assert len(traced.graph.nodes) + 2 == len(traced_dynamic.graph.nodes)
    # and should give same results
    truth_outs = [traced(*e) for e in test_inputs]
    traced_outs = [traced_dynamic(*e) for e in test_inputs]
    for truth_out, traced_out in zip(truth_outs, traced_outs):
        for truth_tensor, traced_tensor in zip(truth_out, traced_out):
            assert truth_tensor.shape == traced_tensor.shape
            assert torch.allclose(truth_tensor, traced_tensor)
