# opt_einsum_fx

[![Documentation Status](https://readthedocs.org/projects/opt-einsum-fx/badge/?version=latest)](https://opt-einsum-fx.readthedocs.io/en/latest/?badge=latest)

Optimizing einsums and functions involving them using [`opt_einsum`](https://optimized-einsum.readthedocs.io/en/stable/) and PyTorch [FX](https://pytorch.org/docs/stable/fx.html) compute graphs.

Issues, questions, PRs, and any thoughts about further optimizing these kinds of operations are welcome!

For more information please see [the docs](https://opt-einsum-fx.readthedocs.io/en/stable/).

## Installation

### PyPI

The latest release can be installed from PyPI:
```bash
$ pip install opt_einsum_fx
```

### Source

To get the latest code, run:

```bash
$ git clone https://github.com/Linux-cpp-lisp/opt_einsum_fx.git
```
and install it by running
```bash
$ cd opt_einsum_fx/
$ pip install .
```

You can run the tests with
```bash
$ pytest tests/
```

## Minimal example

```python
import torch
import torch.fx
import opt_einsum_fx

def einmatvecmul(a, b, vec):
    """Batched matrix-matrix-vector product using einsum"""
    return torch.einsum("zij,zjk,zk->zi", a, b, vec)

graph_mod = torch.fx.symbolic_trace(einmatvecmul)
print("Original code:\n", graph_mod.code)
graph_opt = opt_einsum_fx.optimize_einsums_full(
    model=graph_mod,
    example_inputs=(
        torch.randn(7, 4, 5),
        torch.randn(7, 5, 3),
        torch.randn(7, 3)
    )
)
print("Optimized code:\n", graph_opt.code)
```
outputs
```
Original code:
import torch
def forward(self, a, b, vec):
    einsum_1 = torch.functional.einsum('zij,zjk,zk->zi', a, b, vec);  a = b = vec = None
    return einsum_1

Optimized code:
import torch
def forward(self, a, b, vec):
    einsum_1 = torch.functional.einsum('cb,cab->ca', vec, b);  vec = b = None
    einsum_2 = torch.functional.einsum('cb,cab->ca', einsum_1, a);  einsum_1 = a = None
    return einsum_2
```

We can measure the performance improvement (this is on a CPU):
```python
from torch.utils.benchmark import Timer

batch = 1000
a, b, vec = torch.randn(batch, 4, 5), torch.randn(batch, 5, 8), torch.randn(batch, 8)

g = {"f": graph_mod, "a": a, "b": b, "vec": vec}
t_orig = Timer("f(a, b, vec)", globals=g)
print(t_orig.timeit(10_000))

g["f"] = graph_opt
t_opt = Timer("f(a, b, vec)", globals=g)
print(t_opt.timeit(10_000))
```
gives ~2x improvement:
```
f(a, b, vec)
  276.58 us
  1 measurement, 10000 runs , 1 thread

f(a, b, vec)
  118.84 us
  1 measurement, 10000 runs , 1 thread
```
Depending on your function and dimensions you may see even larger improvements.

## License

`opt_einsum_fx` is distributed under an [MIT license](LICENSE).