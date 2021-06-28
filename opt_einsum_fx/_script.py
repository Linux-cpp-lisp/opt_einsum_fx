from typing import Union
from packaging import version

import torch
from torch import fx


# see https://github.com/pytorch/pytorch/issues/53487
def jitable(obj: Union[fx.GraphModule, fx.Graph]) -> Union[fx.GraphModule, fx.Graph]:
    """Convert some torch calls into their TorchScript signatures.

    In place. Currently deals with ``tensordot`` and ``permute``.

    Args:
        obj: the ``fx.Graph`` or ``fx.GraphModule`` to process.

    Returns:
        ``obj``, modified in-place.
    """
    if isinstance(obj, fx.GraphModule):
        graph = obj.graph
    else:
        graph = obj

    torch_is_ge_19: bool = version.parse(torch.__version__) >= version.parse("1.9.0")

    for node in graph.nodes:
        if node.op == "call_function":
            if (
                node.target == torch.tensordot
                or node.target == torch.functional.tensordot
            ):
                if "dims" in node.kwargs:
                    args = list(node.args)
                    kwargs = dict(node.kwargs)
                    dim_self, dim_other = kwargs.pop("dims")
                    assert len(args) == 2  # tensors 1 and 2
                    if torch_is_ge_19:
                        # In torch >= 1.9.0, they've corrected the torchscript interface
                        # to align with the python one:
                        args.append((list(dim_self), list(dim_other)))
                    else:
                        args.append(list(dim_self))
                        args.append(list(dim_other))
                    node.args = tuple(args)
                    node.kwargs = kwargs
        elif node.op == "call_method":
            if node.target == "permute":
                self_arg, args = node.args[0], node.args[1:]
                if not isinstance(args[0], list):
                    node.args = [self_arg, list(args)]
    graph.lint()
    if isinstance(obj, fx.GraphModule):
        obj.recompile()
    return obj
