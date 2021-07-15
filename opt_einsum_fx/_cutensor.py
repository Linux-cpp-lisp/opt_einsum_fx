# Load extension
import importlib
import torch

# From pytorch_scatter:
# https://github.com/rusty1s/pytorch_scatter/blob/master/torch_scatter/__init__.py
spec = importlib.machinery.PathFinder().find_spec("_opt_einsum_fx")
_HAS_EXTENSION: bool = spec is not None
if _HAS_EXTENSION:
    torch.ops.load_library(spec.origin)


def is_cuTENSOR_available() -> bool:
    return _HAS_EXTENSION
