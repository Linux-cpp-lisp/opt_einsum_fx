from typing import Union

import torch
from torch import fx

from ._efficient_shape_prop import SimpleMeta, EfficientShapeProp

CAST_DTYPE_OPS = [
    torch.einsum,
    torch.functional.einsum,
    torch.tensordot,
    torch.functional.tensordot,
]


def cast_dtypes(
    model: Union[fx.Graph, fx.GraphModule],
    example_inputs: tuple,
    default_dtype=None,
    tracer_class: type = fx.Tracer,
) -> fx.Graph:
    """Check operators in `CAST_DTYPE_OPS` and up-cast"""
    if default_dtype is None:
        default_dtype = torch.get_default_dtype()
    # iterate over nodes
    # find ones in cast ops
    # check args
    # if any dift from default, check that its only one
    # cast other args, and cast result back to default
    output_graph = False
    if isinstance(model, fx.GraphModule):
        graph: fx.Graph = model.graph
    elif isinstance(model, fx.Graph):
        graph: fx.Graph = model
        model = torch.nn.Module()
        output_graph = True
    else:
        tracer: fx.Tracer = tracer_class()
        graph: fx.Graph = tracer.trace(model)
        model = tracer.root

    graphmod: fx.GraphModule = fx.GraphModule(model, graph)
    sp = EfficientShapeProp(graphmod, autocast_default_type=default_dtype)
    sp.run(*example_inputs)

    tracer = fx.proxy.GraphAppendingTracer(graph)

    for node in graph.nodes:
        if node.op == "call_function" and node.target in CAST_DTYPE_OPS:
            dtypes = [
                n.meta["tensor_meta"].dtype for n in node.args if isinstance(n, fx.Node)
            ]
            if all(dt == default_dtype for dt in dtypes):
                continue  # nothing to do here
            assert len(set(dtypes)) == 2
            target_dtype = next(dt for dt in dtypes if dt != default_dtype)
            # now replace the args with casts
            new_args = []
            for arg in node.args:
                if (
                    isinstance(arg, fx.Node)
                    and arg.meta["tensor_meta"].dtype != target_dtype
                ):
                    with graph.inserting_before(node):
                        # cast it
                        new_args.append(
                            fx.Proxy(arg, tracer=tracer).to(dtype=target_dtype).node
                        )
                else:
                    new_args.append(arg)
            node.args = tuple(new_args)
            # convert back
            with graph.inserting_after(node):
                new_output = fx.Proxy(node, tracer=tracer).to(dtype=default_dtype)
            # have to give it the right metadata
            new_node = new_output.node
            node.replace_all_uses_with(new_node)
            new_node.args = (node,) + new_node.args[1:]
            new_node.meta = node.meta.copy()
            new_node.meta["tensor_meta"] = SimpleMeta(
                shape=node.meta["tensor_meta"].shape, dtype=default_dtype
            )

    out_mod = fx.GraphModule(model, graph)
    if output_graph:
        return out_mod.graph
    else:
        out_mod.recompile()
        return out_mod
