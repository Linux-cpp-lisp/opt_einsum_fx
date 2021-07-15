from ._script import jitable
from ._opt_ein import optimize_einsums, optimize_einsums_full
from ._fuse import fuse_einsums, fuse_scalars

__all__ = [
    "jitable",
    "optimize_einsums",
    "optimize_einsums_full",
    "fuse_einsums",
    "fuse_scalars",
]


# Load extension
import importlib
import torch

# From pytorch_scatter:
# https://github.com/rusty1s/pytorch_scatter/blob/master/torch_scatter/__init__.py
spec = importlib.machinery.PathFinder().find_spec("_opt_einsum_fx")
_HAS_EXTENSION: bool = spec is not None
if _HAS_EXTENSION:
    torch.ops.load_library(spec.origin)
