# opt_einsum_fx

_under development_

Einsum optimization using opt_einsum and PyTorch FX graph rewriting.

This library currently supports:
 - Fusing multiple einsums into one
 - Optimizing einsums using the [`opt_einsum`](https://optimized-einsum.readthedocs.io/en/stable/) library
 - Fusing multiplication and division with scalar constants, including fusing _through_ operations, like einsum, that commute with scalar multiplication.
 - Placing multiplication by fused scalar constants onto the smallest intermediate in a chain of operations that commute with scalar multiplication. 

## Installation

Source code for releases can be found at https://github.com/Linux-cpp-lisp/opt_einsum_fx/releases. Otherwise, to get the latest code, run:

```bash
$ git clone https://github.com/Linux-cpp-lisp/opt_einsum_fx.git
```

After downloading source code, run:
```bash
$ cd opt_einsum_fx/
$ pip install .
```
You can run the tests with
```bash
$ pytest tests/
```

## Usage

`opt_einsum_fx` is based on [`torch.fx`](https://pytorch.org/docs/stable/fx.html), a framework for converting between PyTorch Python code and a programatically manipulable compute graph. To use this package, it must be possible to get your function or model as a `torch.fx.Graph`: the limitations of FX's symbolic tracing are discussed [here](https://pytorch.org/docs/stable/fx.html#limitations-of-symbolic-tracing).

### Minimal example

```python
import torch
import torch.fx
import opt_einsum_fx

def einmatvecmul(a, b, vec):
    """Matrix-matrix-vector product using einsum"""
    return torch.einsum("ij,jk,k->i", a, b, vec)

graph_mod = torch.fx.symbolic_trace(einmatvecmul)
print("Original code:\n", graph_mod.code)
graph_opt = opt_einsum_fx.optimize_einsums_full(
    model=graph_mod,
    example_inputs=(
        torch.randn(4, 5),
        torch.randn(5, 3),
        torch.randn(3)
    )
)
print("Optimized code:\n", graph_opt.code)
```
outputs
```
Original code:
import torch
def forward(self, a, b, vec):
    einsum_1 = torch.functional.einsum('ij,jk,k->i', a, b, vec);  a = b = vec = None
    return einsum_1
    
Optimized code:
import torch
def forward(self, a, b, vec):
    tensordot_1 = torch.functional.tensordot(vec, b, dims = ((0,), (1,)));  vec = b = None
    tensordot_2 = torch.functional.tensordot(tensordot_1, a, dims = ((0,), (1,)));  tensordot_1 = a = None
    return tensordot_2
```
The `optimize_einsums_full` function has four passes:

 1. Scalar accumulation --- use the multilinearity of einsum to fuse all constant coefficients and divisors of operands and outputs
 2. Fusing einsums --- gives greater flexibility to (3)
 3. Optimized contraction with ``opt_einsum``
 4. Moving constant scalar coefficients through operations they commute with in order to place them on the smallest possible intermediate results
 
### JIT

Currently, pure Python and TorchScript have different call signatures for `torch.tensordot` and `torch.permute`, both of which can appear in optimized einsums:
```python
graph_script = torch.jit.script(graph_opt)  # => RuntimeError: Arguments for call are not valid...
```
A function is provided to convert `torch.fx.GraphModule`s containing these operations from their Python signatures — the default — to a TorchScript compatible form:
```python
graph_script = torch.jit.script(opt_einsum_fx.jitable(graph_opt))
```

### More information

More information can be found in docstrings in the source; the tests in `tests/` also serve as usage examples.

## License

`opt_einsum_fx` is distributed under an [MIT license](LICENSE).