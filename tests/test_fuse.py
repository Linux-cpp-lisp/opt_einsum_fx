import torch
import torch.fx

from opt_einsum_fx import fuse_einsums


def fusable(x, y):
    z = torch.einsum("ij,jk->ik", x, y)
    return torch.einsum("ik,ij->i", z, x)


def test_einsum_fuse():
    g = torch.fx.symbolic_trace(fusable)
    new_graph = fuse_einsums(g.graph)
    g.graph = new_graph
    g.recompile()
    x, y = torch.randn(3, 4), torch.randn(4, 5)
    out_truth = fusable(x, y)
    out_fused = g(x, y)
    assert torch.allclose(out_fused, out_truth)
