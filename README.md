# opt_einsum_fx
Einsum optimization using opt_einsum and PyTorch FX graph rewriting.

This library currently supports:
 - Fusing multiple einsums into one, when possible
 - Optimizing einsums using the [`opt_einsum`](https://optimized-einsum.readthedocs.io/en/stable/) library

## Installation

```bash
$ pip install 
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
graph_opt = opt_einsum_fx.optimize_einsums(
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
The `optimize_einsums` function has three passes:

  1. Einsum fusion: if the only use of the result of an einsum is as an operand to another einsum, it can be fused into the later einsum
  2. Shape propagation: use [`torch.fx.passes.shape_prop.ShapeProp`](https://github.com/pytorch/pytorch/blob/master/torch/fx/passes/shape_prop.py) and the provided example inputs to determine the shapes of the operands of all einsums
  3. Einsum optimization: generate optimized contractions for those shapes using `opt_einsum`

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