from typing import Optional

import torch
from torch import fx


def get_shape(n: fx.Node) -> Optional[torch.Size]:
    """Get the shape of a node after ``ShapeProp``"""
    try:
        return n.meta["tensor_meta"].shape
    except KeyError:
        return None
    except AttributeError:
        return None
