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
) -> fx.Graph:
    """Deduplicate a graph in-place"""
    exclude_functions = set(exclude_functions)
    exclude_methods = set(exclude_methods)
    seen = []
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
                break
        if not replaced:
            seen.append(node)
    graph.lint()
    return graph
