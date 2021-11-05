from collections import namedtuple
from typing import Any, NamedTuple, Optional, Tuple

import opt_einsum
import torch
import torch.fx
from torch.fx.node import Node, map_aggregate


def einsum_shape(subscripts, *shapes):
    Shaped = namedtuple('Shaped', ['shape'])

    input_subscripts, output_subscript, _ = opt_einsum.parser.parse_einsum_input(
        (subscripts,) + tuple(Shaped(shape) for shape in shapes)
    )
    dims = {
        i: dim
        for ii, shape in zip(input_subscripts.split(','), shapes)
        for i, dim in zip(ii, shape)
    }
    return tuple(dims[i] for i in output_subscript)


class TensorMetadata(NamedTuple):
    # TensorMetadata is a structure containing pertinent information
    # about a tensor within a PyTorch program.

    # General Tensor metadata
    shape: torch.Size
    dtype: torch.dtype
    requires_grad: bool
    stride: Tuple[int]
    memory_format: Optional[torch.memory_format]

    # Quantization metadata
    is_quantized: bool
    qscheme: Optional[torch.qscheme]
    q_scale: Optional[float]
    q_zero_point: Optional[int]


def _extract_tensor_metadata(result: torch.Tensor) -> TensorMetadata:
    """
    Extract a TensorMetadata NamedTuple describing `result`.
    """
    shape = result.shape
    dtype = result.dtype
    requires_grad = result.requires_grad
    stride = result.stride()

    memory_formats = {
        torch.contiguous_format,
        torch.channels_last,
        torch.channels_last_3d,
    }

    memory_format = None

    for query_format in memory_formats:
        if result.is_contiguous(memory_format=query_format):
            memory_format = query_format
            break

    is_quantized = result.is_quantized
    qscheme = None
    q_scale = None
    q_zero_point = None

    if is_quantized:
        qscheme = result.qscheme()

        if qscheme in {torch.per_tensor_affine, torch.per_tensor_symmetric}:
            q_scale = result.q_scale()
            q_zero_point = result.q_zero_point()

    return TensorMetadata(
        shape, dtype, requires_grad, stride, memory_format, is_quantized, qscheme, q_scale, q_zero_point)


class ShapeProp(torch.fx.Interpreter):
    def run_node(self, n: Node) -> Any:
        if n.op == 'call_function' and n.target == torch.einsum:
            args, kwargs = self.fetch_args_kwargs_from_env(n)
            assert not kwargs
            subscripts = args[0]
            shapes = [x.shape for x in args[1:]]
            shape = einsum_shape(subscripts, *shapes)
            result = torch.empty(shape, dtype=args[1].dtype)
        elif n.op == 'call_function' and n.target == torch.tensordot:
            args, kwargs = self.fetch_args_kwargs_from_env(n)
            shape_a, shape_b = [x.shape for x in args]
            inds_a, inds_b = kwargs['dims']
            shape_a = [n for i, n in enumerate(shape_a) if i not in inds_a]
            shape_b = [n for i, n in enumerate(shape_b) if i not in inds_b]
            result = torch.empty(shape_a + shape_b, dtype=args[0].dtype)
        else:
            result = super().run_node(n)

        found_tensor = False

        def extract_tensor_meta(obj):
            if isinstance(obj, torch.Tensor):
                nonlocal found_tensor
                found_tensor = True
                return _extract_tensor_metadata(obj)
            else:
                return obj

        meta = map_aggregate(result, extract_tensor_meta)
        if found_tensor:
            n.meta['tensor_meta'] = meta

        n.meta['type'] = type(result)
        return result

    def propagate(self, *args):
        """
        Run `module` via interpretation and return the result and
        record the shape and type of each node.

        Args:
            *args (Tensor): the sample input.

        Returns:
            Any: The value returned from executing the Module
        """
        return super().run(*args)
