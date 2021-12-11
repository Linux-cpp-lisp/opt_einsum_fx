from typing import Any, NamedTuple

import opt_einsum
import torch
from torch.fx.node import Node

from ._fuse import _EINSUM_FUNCS


class SimpleMeta(NamedTuple):
    """
    The full ShapeProp defines and uses a NamedTuple to
    store a whole bunch of metadata about the tensors
    going into and out of the Node op. But we don't
    have most of that info, and anyway, I don't think
    most of it's used in opt_einsum or opt_einsum_fx.
    (These are only concerned with computing a summation
    order.)

    Rather than give dummy or default values, which I
    only *assume* would be fine, I'm defining a NamedTuple
    with only the values we actually know. So if I'm wrong
    we will get a very clear error message, rather than
    some invisible error.
    """

    shape: torch.Size
    dtype: torch.dtype


class EfficientShapeProp(torch.fx.Interpreter):
    """
    Like ShapeProp, traverses a graph Node-by-Node
    and records the shape and type of the result
    into each Node.

    Except we treat 'einsum' as a special case.
    We don't actually execute 'einsum' on tensors,
    since the einsums will typically not be optimized
    yet (ShapeProp is called before optimization),
    and inefficient summation order can create
    enormous intermediate tensors, which often creates
    needless out-of-memory errors.

    So we override 'run_node' only for 'einsums'.
    It's straightforward to determine the shape of the
    result just from the output indices.

    (The call to opt_einsum that will typically follow
    this, also doesn't actually build the tensors
    during its exploration.)
    """

    def __init__(
        self, module, garbage_collect_values: bool = True, autocast_default_type=None,
    ):
        super().__init__(module, garbage_collect_values=garbage_collect_values)
        self.autocast_default_type = autocast_default_type

    def _check_dtypes(self, dtypes):
        dtype_set = set(dtypes)
        if self.autocast_default_type is not None:
            assert 1 <= len(dtype_set) <= 2
            return self.autocast_default_type
        else:
            assert len(dtype_set) == 1
            return next(iter(dtype_set))

    def run_node(self, n: Node) -> Any:
        if n.op == "call_function" and n.target in _EINSUM_FUNCS:
            args, kwargs = self.fetch_args_kwargs_from_env(n)
            equation, *operands = args
            shapes = [op.shape for op in operands]

            dtype_out = self._check_dtypes(op.dtype for op in operands)
            meta = SimpleMeta(einsum_shape(equation, *shapes), dtype_out)
            result = torch.zeros(
                (1,) * len(meta.shape), dtype=meta.dtype, device=operands[0].device
            ).expand(meta.shape)
        elif n.op == "call_function" and n.target == torch.tensordot:
            args, kwargs = self.fetch_args_kwargs_from_env(n)
            shape_a = [
                dim for i, dim in enumerate(args[0].shape) if i not in kwargs["dims"][0]
            ]
            shape_b = [
                dim for i, dim in enumerate(args[1].shape) if i not in kwargs["dims"][1]
            ]

            dtype_out = self._check_dtypes(op.dtype for op in args)

            meta = SimpleMeta(shape_a + shape_b, dtype_out)
            result = torch.zeros(
                (1,) * len(meta.shape), dtype=meta.dtype, device=args[0].device
            ).expand(meta.shape)
        else:
            result = super().run_node(n)

            if isinstance(result, torch.Tensor):
                meta = SimpleMeta(result.shape, result.dtype)
            else:
                meta = None

        n.meta = dict()
        n.meta["tensor_meta"] = meta
        n.meta["type"] = type(result)

        return result

    def propagate(self, *args):
        return super().run(*args)


def einsum_shape(subscripts, *shapes):
    """
    Given an einsum equation and input shapes, returns the output
    shape of the einsum.

    Args:
       subscripts: the einsum formula
       shapes: the input shapes
    """
    Shaped = NamedTuple("Shaped", [("shape", tuple)])
    input_subscripts, output_subscript, _ = opt_einsum.parser.parse_einsum_input(
        (subscripts,) + tuple(Shaped(shape) for shape in shapes)
    )
    dims = {
        i: dim
        for ii, shape in zip(input_subscripts.split(","), shapes)
        for i, dim in zip(ii, shape)
    }
    return tuple(dims[i] for i in output_subscript)
