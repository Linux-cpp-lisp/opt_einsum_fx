from typing import Callable, Union
import warnings

import torch
from torch import fx
from torch.fx.passes.shape_prop import ShapeProp

import opt_einsum
from opt_einsum.contract import _core_contract

from ._fuse import fuse_einsums, fuse_scalars, _EINSUM_FUNCS


def optimize_einsums_full(
    model: Union[torch.nn.Module, Callable, fx.Graph],
    example_inputs: tuple,
    contract_kwargs: dict = {},
    tracer_class: type = fx.Tracer,
) -> Union[fx.GraphModule, fx.Graph]:
    """Optimize einsums in ``model`` for ``example_inputs``.

    All of the restrictions of ``torch.fx`` symbolic tracing apply.

    Applies, in order, four optimizations:
        1. Scalar accumulation --- use the multilinearity of einsum to collect all constant coefficients and divisors of operands and outputs
        2. Fusing einsums --- gives greater flexibility to (3)
        3. Optimized contraction with ``opt_einsum``.
        4. Moving constant scalar coefficients through operations they commute with in order to place them on the smallest possible intermediate results

    Args:
        model (torch.nn.Module or callable or fx.Graph): the model, function, or ``fx.Graph`` to optimize.
        example_inputs (tuple): arguments to ``model`` whose shapes will determine the einsum optimizations.
        tracer_class (type, optional): the tracer class to use to turn ``model`` into an ``fx.Graph`` if it isn't already an ``fx.GraphModule`` or ``fx.Graph``.

    Returns:
        An optimized ``fx.GraphModule``, or if ``model`` is an ``fx.Graph``, an optimized ``fx.Graph``.
    """
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

    # 1. Scalar accumulation
    # without shape information, this just accumulates scalars and moves them to the end of chains of linear operations
    graph = fuse_scalars(graph)

    # 2. Fuse any einsums we can
    # This gives opt_einsum the most freedom possible to rearange things
    # Since we already moved scalars to the end of chains of linear operations, any scalars between linear operations should already have been moved
    graph = fuse_einsums(graph, in_place=True)
    out_mod = fx.GraphModule(model, graph)

    # 3. Shape propagation
    sp = ShapeProp(out_mod)
    sp.run(*example_inputs)

    # 4. Optimize einsums
    out_mod.graph = optimize_einsums(out_mod.graph, contract_kwargs)
    out_mod.recompile()

    # 5. Shape prop (again)
    # We need shapes to put the scalars in the best place
    sp = ShapeProp(out_mod)
    sp.run(*example_inputs)

    # 6. Final scalar fusion to move scalars
    out_mod.graph = fuse_scalars(out_mod.graph, in_place=True)

    if output_graph:
        return out_mod.graph
    else:
        out_mod.recompile()
        return out_mod


# Based on "Proxy Retracing" example in https://pytorch.org/docs/stable/fx.html
def optimize_einsums(graph: fx.Graph, contract_kwargs: dict = {}) -> fx.Graph:
    """Optimize einsums in a ``torch.fx.Graph`` using ``opt_einsum``.

    ``graph`` must have shape information such as that populated by ``torch.fx.passes.shape_prop.ShapeProp``.
    """
    defaults = {
        "optimize": "optimal",
    }
    defaults.update(contract_kwargs)
    contract_kwargs = defaults

    new_graph = fx.Graph()
    # env keeps track of new injected nodes in addition to existing ones,
    # making sure they get into new_graph
    env = {}
    node_processed: bool = False
    for node in graph.nodes:
        node_processed = False
        if node.op == "call_function" and node.target in _EINSUM_FUNCS:
            # Get shapes:
            try:
                shapes = [a.shape for a in node.args[1:]]
            except AttributeError:
                warnings.warn(
                    f"einsum {repr(node)} lacked shape information; "
                    "not optimizing. "
                    "Did you forget to run ShapeProp on this graph?",
                    RuntimeWarning,
                )
            else:
                # We have shapes, so:
                # Determine the optimal contraction
                path, path_info = opt_einsum.contract_path(
                    node.args[0],  # the einstr
                    *shapes,
                    shapes=True,
                    **contract_kwargs,
                )
                # By wrapping the arguments with proxies,
                # we can dispatch to opt_einsum and implicitly
                # add it to the Graph by symbolically tracing it.
                proxy_args = [
                    fx.Proxy(env[x.name]) if isinstance(x, fx.Node) else x
                    for x in node.args
                ]
                # Use _core_contract to avoid `len()` calls that
                # fx can't deal with
                output_proxy = _core_contract(
                    proxy_args[1:],
                    path_info.contraction_list,
                    backend="torch",
                    evaluate_constants=False,
                )

                # Operations on `Proxy` always yield new `Proxy`s, and the
                # return value of our decomposition rule is no exception.
                # We need to extract the underlying `Node` from the `Proxy`
                # to use it in subsequent iterations of this transform.
                new_node = output_proxy.node
                env[node.name] = new_node
                node_processed = True

        if not node_processed:
            # Default case: just copy the node over into the new graph.
            new_node = new_graph.node_copy(node, lambda x: env[x.name])
            env[node.name] = new_node

    new_graph.lint()
    return new_graph
