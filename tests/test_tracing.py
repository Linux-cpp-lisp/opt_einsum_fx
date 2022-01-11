import pytest

import torch

from opt_einsum_fx.tracing import make_fx_dynamic


def explict_reshape(x):
    bdim, fdim = x.shape
    y = x.view(-1).tanh()
    return y.view(bdim, fdim)  # use a dim that will be traced as constant


def simple(x):
    # do nothing dynamic
    return 2.7 * x.square() * x.tanh() + 3.0


def grad(x):
    x = x.clone().requires_grad_(True)
    y = x.tanh().square().sum()
    grad = torch.autograd.grad(y, x)[0]
    return grad


@pytest.mark.parametrize("func", [explict_reshape, simple, grad])
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
