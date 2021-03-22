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
#
# Note that in general we do not support scalar fusion through in-place operations; it complicates following things through the compute graph too much
# TODO: ^ ???


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
        if node.target == "mul":
            if isinstance(node.args[1], numbers.Number):
                return node.args[0], node.args[1]
        elif node.target == "div":
            if isinstance(node.args[1], numbers.Number):
                return node.args[0], 1.0 / node.args[1]
    return node, None


# Operations that are (almost) "multilinear", in the sense that they commute with scalar multiplication of their operands
SCALAR_COMMUTE_OPS = [
    torch.einsum,
    torch.functional.einsum,
    torch.tensordot,
    torch.functional.tensordot,
    "permute",
    # "reshape",
    "mul",
    "div",
    operator.mul,
    operator.truediv,
]


def prod(x):
    """Compute the product of a sequence."""
    out = 1
    for a in x:
        out *= a
    return out


def fuse_scalars(graph: fx.Graph, in_place: bool = False) -> fx.Graph:
    """Use the multilinearity of einsum to unify and remove constant scalars around einsums."""
    if not in_place:
        graph = copy.deepcopy(graph)

    # Clear any previous state this graph has
    for node in graph.nodes:
        if hasattr(node, "in_lin_chain"):
            delattr(node, "in_lin_chain")

    # Find chains of multilinear ops
    seen_nodes = set()
    linear_chains = []
    for node in graph.nodes:
        if id(node) in seen_nodes:
            continue

        # Determine a linear chain
        cur_linear_chain = []
        while (
            id(node) not in seen_nodes
            and getattr(node, "target", None) in SCALAR_COMMUTE_OPS
        ):
            seen_nodes.add(id(node))
            node.in_lin_chain = len(linear_chains)
            cur_linear_chain.append(node)
            # Continue building the chain regardless, since the merger uses this
            users = list(node.users.keys())
            if len(users) > 0:
                # Get the next node in the chain
                node = users[0]
            else:
                # This isn't used in the graph at all, break the chain
                node = None
            if len(users) != 1:
                # End this chain
                break

        # If the next user, which is now in node, was seen but is itself in a linear chain, this means we merge them
        # TODO: thoroughly test this
        if hasattr(node, "in_lin_chain") and len(cur_linear_chain) > 0:
            # Merge
            merge_into = node.in_lin_chain
            for n in cur_linear_chain:
                n.in_lin_chain = merge_into
            linear_chains[merge_into].extend(cur_linear_chain)
        else:
            # This is a new chain
            linear_chains.append(cur_linear_chain)

    # Accumulate scalars in them
    scalars = []
    for lin_chain_i, lin_chain in enumerate(linear_chains):
        if len(lin_chain) < 2:
            # There's nothing to do here: either the chain is empty,
            # or there's only one operation — even if its a scalar multiplication,
            # theres nothing for us to do with it
            scalars.append(None)
            continue

        # Accumulate scalars
        scalar_node_idexes = []
        total_scalar = 1.0
        for node_i, node in enumerate(lin_chain):
            new_node, scalar = _get_node_and_scalar(node)
            if scalar is not None:
                total_scalar *= scalar
                scalar_node_idexes.append(node_i)

        is_all_scalars = len(scalar_node_idexes) == len(lin_chain)

        # Remove scalar nodes
        for node_i in scalar_node_idexes:
            node = lin_chain[node_i]
            new_node, scalar = _get_node_and_scalar(node)
            assert scalar is not None

            if is_all_scalars and node_i == len(lin_chain) - 1:
                # If it's all scalars, we just put the total_scalar into the last operation
                # and don't save a scalar for later
                with graph.inserting_after(node):
                    new_node = graph.call_function(
                        operator.mul,
                        (total_scalar, new_node),
                    )
                total_scalar = None

            node.replace_all_uses_with(new_node)
            graph.erase_node(node)

        # Save the scalar for this chain
        scalars.append(total_scalar)
        # Remove all of the removed scalar operations from the lin chain
        # See https://stackoverflow.com/a/11303234/1008938
        for index in sorted(
            (scalar_node_idexes[:-1] if is_all_scalars else scalar_node_idexes),
            reverse=True,
        ):
            del lin_chain[index]

    del seen_nodes

    # Make sure everything is still OK
    graph.lint()

    # Now we have chains without scalar operations; we can go through and add back in the scalars in the optimal place
    for lin_chain_i, lin_chain in enumerate(linear_chains):
        if (
            len(lin_chain) == 0
            or scalars[lin_chain_i] == 1.0
            or scalars[lin_chain_i] is None
        ):
            # Nothing to do with an empty chain
            # No reason to add back a scalar that does nothing
            # None signals don't process from above
            continue

        # Find the smallest argument or the output
        smallest_node_i = None
        smallest_arg_i = None
        smallest_size = float("inf")
        for node_i, node in enumerate(lin_chain):
            for arg_i, arg in enumerate(node.args):
                if hasattr(arg, "shape"):
                    if prod(arg.shape) < smallest_size:
                        smallest_node_i = node_i
                        smallest_arg_i = arg_i
                        smallest_size = prod(arg.shape)

        # Put the accumulated scalar on a node
        if (smallest_node_i is None) or (
            hasattr(lin_chain[-1], "shape")
            and prod(lin_chain[-1].shape) < smallest_size
        ):
            # The output is the smallest, put it there
            # OR there was no smallest argument, put it on the end of the chain
            with graph.inserting_after(lin_chain[-1]):
                new_node = graph.call_function(operator.mul, tuple())  # placeholder
                lin_chain[-1].replace_all_uses_with(new_node)
                new_node.args = (lin_chain[-1], scalars[lin_chain_i])
        else:
            # The smallest was someone's arg, so we replace that with a scalar multiplication:
            with graph.inserting_before(lin_chain[smallest_node_i]):
                new_arg = graph.call_function(
                    operator.mul,
                    (
                        lin_chain[smallest_node_i].args[smallest_arg_i],
                        scalars[lin_chain_i],
                    ),
                )
                new_args = list(lin_chain[smallest_node_i].args)
                new_args[smallest_arg_i] = new_arg
                lin_chain[smallest_node_i].args = tuple(new_args)

    graph.lint()
    return graph
