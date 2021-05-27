Tutorial
========

``opt_einsum_fx`` is a library for optimizng einsums and functions involving them using ` ``opt_einsum`` <https://optimized-einsum.readthedocs.io/en/stable/>`_ and `PyTorch FX <https://pytorch.org/docs/stable/fx.html>`_ compute graphs.

The library currently supports:
 - Fusing multiple einsums into one
 - Optimizing einsums using the ` ``opt_einsum`` <https://optimized-einsum.readthedocs.io/en/stable/>`_ library
 - Fusing multiplication and division with scalar constants, including fusing _through_ operations, like einsum, that commute with scalar multiplication.
 - Placing multiplication by fused scalar constants onto the smallest intermediate in a chain of operations that commute with scalar multiplication. 

``opt_einsum_fx`` is based on ` ``torch.fx`` <https://pytorch.org/docs/stable/fx.html>`_, a framework for converting between PyTorch Python code and a programatically manipulable compute graph. To use this package, it must be possible to get your function or model as a `torch.fx.Graph`: the limitations of FX's symbolic tracing are discussed `here <https://pytorch.org/docs/stable/fx.html#limitations-of-symbolic-tracing>`_.

Minimal example
---------------

 .. code-block:: python

    import torch
    import torch.fx
    import opt_einsum_fx

    def einmatvecmul(a, b, vec):
        """Batched matrix-matrix-vector product using einsum"""
        return torch.einsum("zij,zjk,zk->zi", a, b, vec)

    graph_mod = torch.fx.symbolic_trace(einmatvecmul)
    print("# Original code:\n", graph_mod.code)
    graph_opt = opt_einsum_fx.optimize_einsums_full(
        model=graph_mod,
        example_inputs=(
            torch.randn(7, 4, 5),
            torch.randn(7, 5, 3),
            torch.randn(7, 3)
        )
    )
    print("# Optimized code:\n", graph_opt.code)

outputs:

 .. code-block:: python

    # Original code:
    import torch
    def forward(self, a, b, vec):
        einsum_1 = torch.functional.einsum('zij,zjk,zk->zi', a, b, vec);  a = b = vec = None
        return einsum_1
        
    # Optimized code:
    import torch
    def forward(self, a, b, vec):
        einsum_1 = torch.functional.einsum('cb,cab->ca', vec, b);  vec = b = None
        einsum_2 = torch.functional.einsum('cb,cab->ca', einsum_1, a);  einsum_1 = a = None
        return einsum_2

The ``optimize_einsums_full`` function has four passes:

 1. Scalar accumulation --- use the multilinearity of einsum to fuse all constant coefficients and divisors of operands and outputs
 2. Fusing einsums --- gives greater flexibility to (3)
 3. Optimized contraction with ``opt_einsum``
 4. Moving constant scalar coefficients through operations they commute with in order to place them on the smallest possible intermediate results

We can measure the performance improvement (this is on a CPU):

 .. code-block:: python

    from torch.utils.benchmark import Timer

    batch = 1000
    a, b, vec = torch.randn(batch, 4, 5), torch.randn(batch, 5, 8), torch.randn(batch, 8)

    g = {"f": graph_mod, "a": a, "b": b, "vec": vec}
    t_orig = Timer("f(a, b, vec)", globals=g)
    print(t_orig.timeit(10_000))

    g["f"] = graph_opt
    t_opt = Timer("f(a, b, vec)", globals=g)
    print(t_opt.timeit(10_000))

gives ~2x improvement:

 .. code-block:: none

    f(a, b, vec)
    276.58 us
    1 measurement, 10000 runs , 1 thread

    f(a, b, vec)
    118.84 us
    1 measurement, 10000 runs , 1 thread

Depending on your function and dimensions you may see even larger improvements.

JIT
---

Currently, pure Python and TorchScript have different call signatures for `torch.tensordot` and `torch.permute`, both of which can appear in optimized einsums:

 .. code-block:: python

    graph_script = torch.jit.script(graph_opt)  # => RuntimeError: Arguments for call are not valid...

A function is provided to convert ``torch.fx.GraphModule`` s containing these operations from their Python signatures — the default — to a TorchScript compatible form:

 .. code-block:: python

    graph_script = torch.jit.script(opt_einsum_fx.jitable(graph_opt))