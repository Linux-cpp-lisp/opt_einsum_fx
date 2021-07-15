import pytest

import functools
import random

import torch
from opt_einsum.parser import find_output_str

import opt_einsum_fx  # noqa: F401


if not torch.cuda.is_available():
    pytest.skip("No CUDA; skipping cuTENSOR tests.", allow_module_level=True)


@pytest.mark.parametrize(
    "einstr",
    [
        "ij,jk->ik",
        "abc,cba",
        "abc,c->ba",
        "a,a->a",  # TODO: this breaks at shape of 1
        "ijk,zuvij->zuvk",
        "zui,zvj->zuvij",
        "zui,zuj->zuij",
    ],
)
@pytest.mark.parametrize(
    "_", [None] * 3
)  # Since the test is randomized, run it multiple times
def test_like_torch(einstr, _):
    device = "cuda"
    atol = {torch.float32: 1e-5, torch.float64: 1e-8}[torch.get_default_dtype()]
    # Shapes
    modes = set(einstr) - {",", "-", ">"}
    extents = {m: random.choice([1, 2, 3, 5, 11, 27, 50]) for m in modes}
    if "->" in einstr:
        modesAB, modesC = einstr.split("->")
    else:
        modesAB, modesC = einstr, find_output_str(einstr)
    modesA, modesB = modesAB.split(",")
    # Make tensors
    A = torch.randn(
        size=[extents[m] for m in modesA],
        device=device,
    )
    B = torch.randn(
        size=[extents[m] for m in modesB],
        device=device,
    )
    # Get the truth output
    A_true, B_true = A.detach().clone(), B.detach().clone()
    A_true.requires_grad_(True)
    B_true.requires_grad_(True)
    true_out = torch.einsum(einstr, A_true, B_true)
    true_grad_A = torch.autograd.grad(true_out.sum(), A_true, create_graph=True)[0]
    true_gradgrad_AB = torch.autograd.grad(true_grad_A.sum(), B_true)[0]
    true_grad_B = torch.autograd.grad(true_out.sum(), B_true)[0]
    # Get the cuTENSOR outputs
    A.requires_grad_(True)
    B.requires_grad_(True)
    out = torch.ops._opt_einsum_fx.einsum(einstr, A, B)
    grad_A = torch.autograd.grad(out.sum(), A, create_graph=True)[0]
    gradgrad_AB = torch.autograd.grad(grad_A.sum(), B)[0]
    grad_B = torch.autograd.grad(out.sum(), B)[0]
    # Check
    assert torch.allclose(true_out, out, atol=atol)
    assert torch.allclose(true_grad_A, grad_A, atol=atol)
    assert torch.allclose(true_gradgrad_AB, gradgrad_AB, atol=atol)
    assert torch.allclose(true_grad_B, grad_B, atol=atol)


@pytest.mark.parametrize(
    # TODO: figure out bfloat16
    "dtype,atol",
    [
        (torch.float16, 1e-4),
        (torch.float32, 1e-5),
        (torch.float64, 1e-8),
    ],  # , torch.bfloat16]
)
def test_dtypes(dtype, atol):
    if torch.get_default_dtype() != torch.float32:
        pytest.skip("This test manages its own dtypes")
    device = "cuda"
    einstr = "ij,jab->iba"
    # Make tensors
    A = torch.randn(
        size=(5, 5),
        dtype=dtype,
        device=device,
    )
    B = torch.randn(
        size=(5, 6, 3),
        dtype=dtype,
        device=device,
    )
    # Get the truth output
    A_true, B_true = A.clone(), B.clone()
    A_true.requires_grad_(True)
    B_true.requires_grad_(True)
    true_out = torch.einsum(einstr, A_true, B_true)
    true_grad_A = torch.autograd.grad(true_out.sum(), A_true, retain_graph=True)[0]
    true_grad_B = torch.autograd.grad(true_out.sum(), B_true)[0]
    # Get the cuTENSOR outputs
    A.requires_grad_(True)
    B.requires_grad_(True)
    out = torch.ops._opt_einsum_fx.einsum(einstr, A, B)
    grad_A = torch.autograd.grad(out.sum(), A, retain_graph=True)[0]
    grad_B = torch.autograd.grad(out.sum(), B)[0]
    # Check
    assert torch.allclose(true_out, out)
    assert torch.allclose(true_grad_A, grad_A)
    assert torch.allclose(true_grad_B, grad_B)
