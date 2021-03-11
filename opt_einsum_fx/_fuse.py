from typing import Optional, Tuple
import string
import copy
import operator
import numbers

import torch
from torch import fx

from opt_einsum.parser import find_output_str

_EINSUM_FUNCS = {torch.functional.einsum, torch.einsum}


# == Einsum fusion ==


def _get_einstrs(einstr):
    if "..." in einstr:
        raise NotImplementedError("Ellipsis `...` in einsum string not supported yet")
    tmp = einstr.split("->")
    if len(tmp) == 1:
        ops = tmp[0]
        out = find_output_str(ops)
    elif len(tmp) == 2:
        ops, out = tmp
    else:
        raise ValueError(f"Invalid einstr {einstr}")
    return ops.split(","), out


def fuse_einsums(graph: fx.Graph, in_place: bool = False) -> fx.Graph:
    """Fuse einsums when possible.

    When the output of one einsum is only used as an operand in another einsum, the two einsums can be fused into one.
    """
    if not in_place:
        graph = copy.deepcopy(graph)

    for node in graph.nodes:
        if node.op == "call_function" and node.target in _EINSUM_FUNCS:
            our_inp_einstrs, our_out_einstr = _get_einstrs(node.args[0])
            assert len(our_inp_einstrs) == len(node.args) - 1
            avail_letters = iter(
                set(string.ascii_lowercase)
                - set.union(*(set(e) for e in our_inp_einstrs))
            )
            new_our_einstrs = []
            new_our_args = []
            we_fused_nodes = []
            # Iterate over operands
            for inp_idex, inp in enumerate(node.args[1:]):
                if (
                    inp.op == "call_function"
                    and inp.target in _EINSUM_FUNCS
                    and len(inp.users) == 1
                ):
                    # This operand is the output of another einsum, and is not used by any other operation
                    # As a result, we can fuse it
                    its_inp_einstrs, its_out_einstr = _get_einstrs(inp.args[0])
                    if len(its_out_einstr) != len(our_inp_einstrs[inp_idex]):
                        raise RuntimeError(
                            f"Inconsistent rank: einsum `{node}`'s input {inp_idex} is the result of einsum {inp}; the output of `{inp}` is labeled `{its_out_einstr}` (rank {len(its_out_einstr)}), but the corresponding input of `{node}` is labeled `{our_inp_einstrs[inp_idex]}` (rank {len(our_inp_einstrs[inp_idex])})"
                        )
                    # First, we need to figure out which of its output dimensions correspond to our dimensions:
                    its_dim_to_ours = dict(
                        zip(its_out_einstr, our_inp_einstrs[inp_idex])
                    )
                    # assign any labels that don't show up in the output of the previous einsum --- and thus dont have labels in the current einsum --- to new letters
                    its_remaining_labels = set.union(
                        *(set(e) for e in its_inp_einstrs)
                    ) - set(its_dim_to_ours.keys())
                    try:
                        its_dim_to_ours.update(
                            dict((i, next(avail_letters)) for i in its_remaining_labels)
                        )
                    except StopIteration:
                        # We ran out of letters
                        raise NotImplementedError(
                            f"At einsum {node}, ran out of letters when trying to fuse parameter einsum {inp}. A fallback for this case is not yet implimented."
                        )
                    else:
                        # We had enough letters, finish adding the fuse
                        del its_remaining_labels
                        new_our_args.extend(inp.args[1:])
                        new_our_einstrs.extend(
                            "".join(its_dim_to_ours[d] for d in es)
                            for es in its_inp_einstrs
                        )
                        we_fused_nodes.append(inp)
                else:
                    # This argument is not from an einsum, or is from an einsum that is used elsewhere as well
                    # Thus we just pass it through
                    new_our_einstrs.append(our_inp_einstrs[inp_idex])
                    new_our_args.append(inp)
            # -- end iter over prev einsum inputs --
            # Set the new values for the einstrs
            node.args = (f"{','.join(new_our_einstrs)}->{our_out_einstr}",) + tuple(
                new_our_args
            )
            # Remove fused inputs
            for to_remove in we_fused_nodes:
                graph.erase_node(to_remove)
        # -- end case for einsum nodes --
    # -- end iter over nodes --
    return graph


# == Scalar fusion ==


# TODO: should the accumulation of constants happen in more than double precision?
def _get_node_and_scalar(node: fx.Node) -> Tuple[fx.Node, Optional[numbers.Number]]:
    """Get a multiplicative scalar for an operation, if applicable."""
    # This supports in-place *= and /= because fx traces them as normal operator.mul/div.
    if node.op == "call_function":
        if node.target == operator.mul or node.target == torch.mul:
            if isinstance(node.args[0], numbers.Number):
                return node.args[1], node.args[0]
            elif isinstance(node.args[1], numbers.Number):
                return node.args[0], node.args[1]
        elif node.target == operator.truediv or node.target == torch.div:
            if isinstance(node.args[1], numbers.Number):
                return node.args[0], 1.0 / node.args[1]
    elif node.op == "call_method":
        # TODO: this could _technically_ be wrong if the nodes `self` argument is not a (proxy to) a Tensor
        if node.target == "mul" or node.target == "mul_":
            if isinstance(node.args[1], numbers.Number):
                return node.args[0], node.args[1]
        elif node.target == "div" or node.target == "div_":
            if isinstance(node.args[1], numbers.Number):
                return node.args[0], 1.0 / node.args[1]
    return node, None


def _accumulate_scalars(graph: fx.Graph):
    """Use the multilinearity of einsum to unify and remove constant scalars around einsums.

    This is NOT a transformation --- it is a convinience function that collects information into einsum nodes and REMOVES other nodes, making the graph incorrect. It should only be used with a transformation that puts them back.
    """
    for node in graph.nodes:
        if node.op == "call_function" and node.target in _EINSUM_FUNCS:
            total_scalar = 1.0
            # First, find if this einsum is multipled/divided as its only use --
            # if it isn't, we can't fuse it with any following operations
            while len(node.users) == 1:
                user = list(node.users.keys())[0]
                new_node, new_scalar = _get_node_and_scalar(user)
                if new_scalar is not None:
                    total_scalar *= new_scalar
                    # Eliminate the accumulated scalar multiplication
                    user.replace_all_uses_with(node)
                    graph.erase_node(user)
                else:
                    # The next user isn't a constant mul, so break
                    break

            # Now assimilate inputs
            for orig_inp in node.args[1:]:
                inp = orig_inp
                while len(inp.users) == 1:
                    # No one else uses this input
                    new_node, new_scalar = _get_node_and_scalar(inp)
                    if new_scalar is not None:
                        total_scalar *= new_scalar
                        inp.replace_all_uses_with(new_node)
                        graph.erase_node(inp)
                        inp = new_node
                    else:
                        break

            # Now we need to add back in the accumulated scalar
            node.scalar_coefficient = total_scalar


def fuse_scalars(graph: fx.Graph, in_place: bool = False) -> fx.Graph:
    """Use the multilinearity of einsum to unify and remove constant scalars around einsums."""
    if not in_place:
        graph = copy.deepcopy(graph)

    _accumulate_scalars(graph)

    for node in graph.nodes:
        if node.op == "call_function" and node.target in _EINSUM_FUNCS:
            if hasattr(node, "scalar_coefficient"):
                with graph.inserting_after(node):
                    new_node = graph.call_method("mul", tuple())  # placeholder
                    node.replace_all_uses_with(new_node)
                    new_node.args = (node, node.scalar_coefficient)
    graph.lint()
    return graph
