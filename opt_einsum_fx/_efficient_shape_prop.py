from typing import Any, NamedTuple

import opt_einsum
import torch
from torch.fx.node import Node  # map_aggregate
from torch.fx.passes.shape_prop import ShapeProp  # TensorMetadata

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


class EfficientShapeProp(ShapeProp):
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

    def run_node(self, n: Node) -> Any:
        if n.op != "call_function" or n.target not in _EINSUM_FUNCS:
            return super().run_node(n)

        equation, *operands = n.args
        shapes = [op.meta['tensor_meta'].shape for op in operands]
        if len(operands) > 0:
            dtypes = [op.meta['tensor_meta'].dtype for op in operands]
            dtype = dtypes[0]
            assert all(d == dtype for d in dtypes), "Tensors in einsum have different dtypes!"
        else:
            dtype = float

        shape = einsum_shape(equation, *shapes)

        n.meta['tensor_meta'] = SimpleMeta(shape, dtype)
        n.meta['type'] = torch.Tensor

        return torch.zeros(shape, dtype=dtype, device='cpu')


def einsum_shape(subscripts, *shapes):
    """
    Given an einsum equation and input shapes, returns the output
    shape of the einsum.

    Args:
       subscripts: the einsum formula
       shapes: the input shapes
    """
    Shaped = NamedTuple('Shaped', [('shape', tuple)])
    input_subscripts, output_subscript, _ = opt_einsum.parser.parse_einsum_input(
        (subscripts,) + tuple(Shaped(shape) for shape in shapes)
    )
    dims = {
        i: dim
        for ii, shape in zip(input_subscripts.split(','), shapes)
        for i, dim in zip(ii, shape)
    }
    return tuple(dims[i] for i in output_subscript)
