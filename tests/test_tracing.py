import pytest

import torch

from opt_einsum_fx.tracing import make_fx_dynamic


def f1(x):
    bdim, fdim = x.shape
    y = x.view(-1).tanh()
    return y.view(bdim, fdim)  # use a dim that will be traced as constant


@pytest.mark.parametrize("func", [f1])
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
