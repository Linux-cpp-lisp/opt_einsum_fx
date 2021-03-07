from typing import Tuple, Dict, Any
import warnings

import torch
from torch import fx

import opt_einsum
from opt_einsum.contract import _core_contract


# Based on "Proxy Retracing" example in https://pytorch.org/docs/stable/fx.html
def optimize_einsums(graph: fx.Graph) -> fx.Graph:
    """Optimize einsums in a ``torch.fx.Graph``."""
    new_graph = fx.Graph()
    # env keeps track of new injected nodes in addition to existing ones,
    # making sure they get into new_graph
    env = {}
    for node in graph.nodes:
        if node.op == 'call_function' and node.target == torch.einsum:
            # Determine the optimal contraction
            path, path_info = opt_einsum.contract_path(
                node.args[0],  # the einstr
                *[a.shape for a in node.args[1:]],
                shapes=True
            )
            # By wrapping the arguments with proxies,
            # we can dispatch to opt_einsum and implicitly
            # add it to the Graph by symbolically tracing it.
            proxy_args = [
                fx.Proxy(env[x.name]) if isinstance(x, fx.Node) else x 
                for x in node.args
            ]
            print("path", path)
            # Use _core_contract to avoid `len()` calls that 
            # fx can't deal with
            output_proxy = _core_contract(
                proxy_args[1:],
                path_info.contraction_list,
                backend='torch',
                evaluate_constants=False
            )

            # Operations on `Proxy` always yield new `Proxy`s, and the
            # return value of our decomposition rule is no exception.
            # We need to extract the underlying `Node` from the `Proxy`
            # to use it in subsequent iterations of this transform.
            new_node = output_proxy.node
            env[node.name] = new_node
        else:
            # Default case: we don't have a decomposition rule for this
            # node, so just copy the node over into the new graph.
            new_node = new_graph.node_copy(node, lambda x: env[x.name])
            env[node.name] = new_node

    new_graph.lint()
    return new_graph
