# Load extension
import importlib

import torch
from torch import fx

# From pytorch_scatter:
# https://github.com/rusty1s/pytorch_scatter/blob/master/torch_scatter/__init__.py
spec = importlib.machinery.PathFinder().find_spec("_opt_einsum_fx")
_HAS_EXTENSION: bool = spec is not None
if _HAS_EXTENSION:
    torch.ops.load_library(spec.origin)


def is_cuTENSOR_available() -> bool:
    return _HAS_EXTENSION


def make_einsums_cuTENSOR(graph: fx.Graph) -> None:
    for node in graph.nodes:
        if node.op == "call_function" and node.target in (
            torch.einsum,
            torch.functional.einsum,
        ):
            if node.args[0].count(",") == 1:  # only pairwise
                node.target = torch.ops._opt_einsum_fx.einsum
