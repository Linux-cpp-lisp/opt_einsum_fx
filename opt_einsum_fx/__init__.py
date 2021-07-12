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

# suffix = 'cuda' if torch.cuda.is_available() else 'cpu'

torch.ops.load_library(
    importlib.machinery.PathFinder()
    .find_spec(
        "_acrotensor",
    )
    .origin
)
