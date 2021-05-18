from typing import Sequence, Callable
import operator

import torch
from torch import fx


def _equivalent(n1: fx.Node, n2: fx.Node) -> bool:
    if n1.op != n2.op:
        return False
    if n1.target != n2.target:
        return False
    if any(a1 != a2 for a1, a2 in zip(n1.args, n2.args)):
        return False
    return True


def deduplicate(
    graph: fx.Graph,
    exclude_functions: Sequence[Callable] = [],
    exclude_methods: Sequence[str] = [],
) -> int:
    """Deduplicate a graph in-place.

    Args:
        graph: the graph
        exclude_functions: list of functions to ignore for deduplication
        exclude_methods: list of method names to ignore for deduplication

    Returns:
        How many nodes were removed.
    """
    graph.lint()
    exclude_functions = set(exclude_functions)
    exclude_methods = set(exclude_methods)
    seen = []
    n_removed = 0
    for node in graph.nodes:
        if node.op == "call_function":
            if node.target in exclude_functions:
                continue
        elif node.op == "call_method":
            if node.target[-1] == "_":
                # mul_, add_, etc. --- in-place methods
                continue
            if node.target in exclude_methods:
                continue
        else:
            continue
        replaced: bool = False
        for seen_node in seen:
            if _equivalent(node, seen_node):
                node.replace_all_uses_with(seen_node)
                graph.erase_node(node)
                replaced = True
                n_removed += 1
                break
        if not replaced:
            seen.append(node)
    graph.lint()
    return n_removed


# based on unreleased code in PyTorch:
# https://github.com/pytorch/pytorch/blob/master/torch/fx/graph.py#L1073
def eliminate_dead_code(graph: fx.Graph, impure_targets=[]) -> None:
    """Eliminate dead code in-place."""
    graph.lint()  # ensure topological sort
    for node in reversed(graph.nodes):
        if (
            node.op in ("call_function", "call_method", "get_attr")
            and len(node.users) == 0
        ):
            if node.target in impure_targets:
                continue
            if isinstance(node.target, str) and node.target.endswith("_"):
                # mul_, etc.
                continue
            graph.erase_node(node)
