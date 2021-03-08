import pytest

import torch
import torch.fx

from opt_einsum_fx import fuse_einsums


def fusable(x, y):
    z = torch.einsum("ij,jk->ik", x, y)
    return torch.einsum("ik,ij->i", z, x)


def unfusable(x, y):
    z = torch.einsum("ij,jk->ik", x, y)
    # We use z as something besides an input to the second einsum, so it is unfusable
    return torch.einsum("ik,ij->i", z, x) + z[:, 0]


def doublefuse(a, b, c, d):
    # quadruple matmul with a final transpose
    e1 = torch.einsum("ij,jk->ik", a, b)
    e2 = torch.einsum("ab,bc->ac", e1, c)
    return torch.einsum("tr,ry->yt", e2, d)


def test_einsum_fuse():
    g = torch.fx.symbolic_trace(fusable)
    new_graph = fuse_einsums(g.graph)
    g.graph = new_graph
    g.recompile()
    x, y = torch.randn(3, 4), torch.randn(4, 5)
    out_truth = fusable(x, y)
    out_fused = g(x, y)
    assert torch.allclose(out_fused, out_truth)


def test_unfusable():
    g = torch.fx.symbolic_trace(unfusable)
    old_code = g.code
    new_graph = fuse_einsums(g.graph)
    g.graph = new_graph
    g.recompile()
    # Confirm numerical equivalence
    x, y = torch.randn(3, 4), torch.randn(4, 5)
    out_truth = unfusable(x, y)
    out_fused = g(x, y)
    assert torch.allclose(out_fused, out_truth)
    # Confirm no fusion:
    assert old_code == g.code


def test_doublefuse():
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
    assert torch.allclose(out_fused, out_truth)


def inconsistent(x, y):
    z = torch.einsum("ij,jk->ik", x, y)
    # Note that the dimension labels for z have the wrong length
    return torch.einsum("i,ij->i", z, x)


def test_inconsistent():
    g = torch.fx.symbolic_trace(inconsistent)
    with pytest.raises(RuntimeError):
        _ = fuse_einsums(g.graph)
