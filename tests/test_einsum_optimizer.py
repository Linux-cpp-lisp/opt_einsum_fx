import pytest

import torch
import torch.fx

from opt_einsum_fx import optimize_einsums, optimize_einsums_full, jitable, EfficientShapeProp


def einmatmul(x, y):
    return torch.einsum("ij,jk->ik", x, y)


def eintrace(x, y):
    # these indexings make it square
    b = torch.einsum("ii", x[:, : x.shape[0]])
    return torch.einsum("jj", y[:, : y.shape[0]]) * b


def fusable(x, y):
    z = torch.einsum("ij,jk->ik", x, y)
    return torch.einsum("ik,ij->i", z, x)


def fusable_w_scalars(x, y):
    z = torch.einsum("ij,jk->ik", x, y) / 3.0
    return 4.0 * torch.einsum("ik,ij->i", z, x)


def unfusable(x, y):
    z = torch.einsum("ij,jk->ik", x, y)
    # We use z as something besides an input to the second einsum, so it is unfusable
    return torch.einsum("ik,ij->i", z, x) + z[:, 0]


def unfusable_w_scalars(x, y):
    z = 2.7 * torch.einsum("ij,jk->ik", x, y)
    # We use z as something besides an input to the second einsum, so it is unfusable
    return torch.einsum("ik,ij->i", z, x) + 1.1 * z[:, 0]


def not_einsum(x, y):
    # Try to trip it up with lots of scalar fusion but no einsums
    return 3.0 * 2.7 * x.sum() + (4.6 / y.relu().sum())


def not_einsum2(x, y):
    a = x.tanh().relu().sum() - y.sum()
    b = 3.41 * y.sum().tanh()
    return a - 6.7 * b


@pytest.fixture(
    scope="module",
    params=[
        einmatmul,
        eintrace,
        fusable,
        fusable_w_scalars,
        unfusable,
        unfusable_w_scalars,
        not_einsum,
        not_einsum2,
    ],
)
def einfunc(request):
    return request.param


def test_optimize_einsums(einfunc, allclose):
    x = torch.randn(3, 4)
    y = torch.randn(4, 5)

    func_res = einfunc(x, y)

    func_fx = torch.fx.symbolic_trace(einfunc)
    sp = EfficientShapeProp(func_fx)
    sp.run(x, y)

    func_fx_res = func_fx(x, y)
    assert torch.all(func_res == func_fx_res)

    graph_opt = optimize_einsums(func_fx.graph)
    func_fx.graph = graph_opt
    func_fx.recompile()

    func_opt_res = func_fx(x, y)
    assert allclose(func_opt_res, func_fx_res)


def test_optimize_einsums_full(einfunc, allclose):
    x = torch.randn(3, 4)
    y = torch.randn(4, 5)
    func_res = einfunc(x, y)
    func_opt = optimize_einsums_full(einfunc, (x, y))
    assert allclose(func_res, func_opt(x, y))


def test_fallback():
    # We only bother to test this for one function
    einfunc = fusable
    # If there is no shape propagation, it should warn
    # and not do anything.
    func_fx = torch.fx.symbolic_trace(einfunc)
    old_code = func_fx.code

    with pytest.warns(RuntimeWarning):
        graph_opt = optimize_einsums(func_fx.graph)

    func_fx.graph = graph_opt
    func_fx.recompile()
    assert old_code == func_fx.code


def test_torchscript(einfunc, allclose):
    x = torch.randn(3, 4)
    y = torch.randn(4, 5)
    func_res = einfunc(x, y)
    mod_opt = optimize_einsums_full(einfunc, (x, y))
    mod_opt = jitable(mod_opt)
    mod_opt = torch.jit.script(mod_opt)
    func_opt_res = mod_opt(x, y)
    assert allclose(func_opt_res, func_res)
