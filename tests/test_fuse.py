import pytest

import math
import copy
import operator

import torch
import torch.fx

from opt_einsum_fx import (
    fuse_einsums,
    fuse_scalars,
    fuse_reshapes,
    optimize_einsums_full,
)


def test_einsum_fuse(allclose):
    def fusable(x, y):
        z = torch.einsum("ij,jk->ik", x, y)
        return torch.einsum("ik,ij->i", z, x)

    g = torch.fx.symbolic_trace(fusable)
    new_graph = fuse_einsums(g.graph)
    g.graph = new_graph
    g.recompile()
    x, y = torch.randn(3, 4), torch.randn(4, 5)
    out_truth = fusable(x, y)
    out_fused = g(x, y)
    assert allclose(out_fused, out_truth)


def test_unfusable():
    def unfusable(x, y):
        z = torch.einsum("ij,jk->ik", x, y)
        # We use z as something besides an input to the second einsum, so it is unfusable
        return torch.einsum("ik,ij->i", z, x) + z[:, 0]

    g = torch.fx.symbolic_trace(unfusable)
    old_code = g.code
    new_graph = fuse_einsums(g.graph)
    g.graph = new_graph
    g.recompile()
    # Confirm numerical equivalence
    x, y = torch.randn(3, 4), torch.randn(4, 5)
    out_truth = unfusable(x, y)
    out_fused = g(x, y)
    # Here we use normal allclose --- since unfusable is unfusable,
    # nothing should have changed.
    assert torch.allclose(out_fused, out_truth)
    # Confirm no fusion:
    assert old_code == g.code


def test_doublefuse(allclose):
    def doublefuse(a, b, c, d):
        # quadruple matmul with a final transpose
        e1 = torch.einsum("ij,jk->ik", a, b)
        e2 = torch.einsum("ab,bc->ac", e1, c)
        return torch.einsum("tr,ry->yt", e2, d)

    g = torch.fx.symbolic_trace(doublefuse)
    new_graph = fuse_einsums(g.graph)
    g.graph = new_graph
    g.recompile()
    a, b, c, d = (
        torch.randn(3, 4),
        torch.randn(4, 5),
        torch.randn(5, 2),
        torch.randn(2, 3),
    )
    out_truth = doublefuse(a, b, c, d)
    out_fused = g(a, b, c, d)
    assert allclose(out_fused, out_truth)


def test_inconsistent():
    def inconsistent(x, y):
        z = torch.einsum("ij,jk->ik", x, y)
        # Note that the dimension labels for z have the wrong length
        return torch.einsum("i,ij->i", z, x)

    g = torch.fx.symbolic_trace(inconsistent)
    with pytest.raises(RuntimeError):
        _ = fuse_einsums(g.graph)


def scalar_fusable1(x, y):
    return 7.0 * torch.einsum("ij,jk->ik", x, y / 3) / 2


def scalar_fusable2(x, y):
    return 4.0 * torch.einsum("ij,jk->ik", x, 2.0 * y / 3) / 2


def scalar_fusable3(x, y):
    return 4.0 * torch.einsum("ij,jk->ik", x / 1.2, 1.7 * 2.0 * y / 3) / 2


def scalar_unfusable(x, y):
    z = 3 * torch.einsum("ij,jk->ik", x, y) / 4.0
    # We use z as something besides an input to the second einsum, so it is unfusable
    return (2.0 * torch.einsum("ik,ij->i", z, x)) + z[:, 0]


def just_scalars(x, y):
    return 3.0 * x


def just_many_scalars(x, y):
    return 3.0 / 3.4 * x / 4.0


def in_place(x, y):
    # This *shouldn't* be fused.
    a = x.clone()
    b = a.mul_(4.0)
    return 3.0 * b


def unused(x, y):
    b = 2.3 * x / 4.5  # noqa
    return 4.6 * torch.einsum("ij,jk->ik", x, y)


def constants(x, y):
    return math.pi * torch.einsum("ij,jk->ik", x, math.e * y / 3) / 2


# In all cases but unfusable, after fusion, the graph should have 5 nodes:
# two placeholders, one einsum, one mul, and one output
@pytest.mark.parametrize(
    "func",
    [
        (scalar_fusable1, 5),
        (scalar_fusable2, 5),
        (scalar_fusable3, 5),
        (
            scalar_unfusable,
            9,  # two placeholders, one einsum one mul, one einsum one mul, one getitem, one sum, and one output = 9
        ),
        (just_scalars, 4),
        (just_many_scalars, 4),
        (in_place, 6),
        (constants, 5),
        (unused, 6),
    ],
)
@pytest.mark.parametrize("in_place_muls", [False, True])
def test_scalar_fuse(allclose, func, in_place_muls):
    func, truth_num_nodes = func
    g = torch.fx.symbolic_trace(func)
    print("old graph\n", g.graph)
    new_graph = fuse_scalars(g.graph, in_place_muls=in_place_muls)
    print("new graph\n", new_graph)
    g.graph = new_graph
    assert len(g.graph.nodes) == truth_num_nodes
    g.recompile()
    x, y = torch.randn(3, 4), torch.randn(4, 5)
    out_truth = func(x, y)
    out_fused = g(x, y)
    assert allclose(out_fused, out_truth)


@pytest.mark.parametrize("in_place_muls", [False, True])
def test_scalar_positioning(allclose, in_place_muls):
    def f(x, y, z):
        return 0.784 * torch.einsum("ij,jk,kl->il", x, y, z)

    x, y, z = torch.randn(2, 100), torch.randn(100, 2), torch.randn(2, 100)

    # note that the smallest here is y
    g = torch.fx.symbolic_trace(f)
    print("old graph\n", g.graph)
    g = optimize_einsums_full(g, (x, y, z), in_place_muls=in_place_muls)
    print("new graph\n", g.graph)
    # optimal placement is on the 2x2 intermediate
    assert list(g.graph.nodes)[4].target == ("mul_" if in_place_muls else "mul")
    out_truth = f(x, y, z)
    out_fused = g(x, y, z)
    assert allclose(out_fused, out_truth)


def test_reshape_fuse(allclose):
    def func(x):
        y = x.reshape(-1, 2, 3)
        z = y.reshape(-1, 6)
        return 3.14 * z

    graphmod = torch.fx.symbolic_trace(func)
    orig_num_nodes = len(graphmod.graph.nodes)
    gmod_fuse = copy.deepcopy(graphmod)
    fuse_reshapes(gmod_fuse.graph, in_place=True)
    gmod_fuse.recompile()
    new_num_nodes = len(gmod_fuse.graph.nodes)
    assert new_num_nodes == orig_num_nodes - 1
    inp = torch.randn(1, 4, 2, 6)
    assert allclose(graphmod(inp), gmod_fuse(inp))
