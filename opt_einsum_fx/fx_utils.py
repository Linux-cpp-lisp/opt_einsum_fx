from typing import Optional
from packaging import version

import torch
from torch import fx

_TORCH_IS_GE_19: bool = version.parse(torch.__version__) >= version.parse("1.9.0")

# The torch FX APIs are not stable, so we need helper wrappers

if _TORCH_IS_GE_19:

    def get_shape(n: fx.Node) -> Optional[torch.Size]:
        """Get the shape of a node after ``ShapeProp``"""
        try:
            return n.meta["tensor_meta"].shape
        except KeyError:
            return None


else:

    def get_shape(n: fx.Node) -> Optional[torch.Size]:
        """Get the shape of a node after ``ShapeProp``"""
        try:
            return n.shape
        except AttributeError:
            return None
