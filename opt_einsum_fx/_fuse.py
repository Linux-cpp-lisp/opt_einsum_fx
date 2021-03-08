from typing import Callable, Union
import string
import copy

import torch
from torch import fx

from ._opt_ein import _EINSUM_FUNCS


def fuse_einsums(graph: fx.Graph, in_place: bool = False):
    """Fuse einsums when possible.

    When the output of one einsum is only used as an operand in another einsum, the two einsums can be fused into one.
    """
    if not in_place:
        graph = copy.deepcopy(graph)

    for node in graph.nodes:
        if node.op == "call_function" and node.target in _EINSUM_FUNCS:
            # TODO: deal with no output indexes
            our_inp_einstrs, our_out_einstr = node.args[0].split("->")
            our_inp_einstrs = our_inp_einstrs.split(",")
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
                    # TODO: deal with no output indexes
                    its_inp_einstrs, its_out_einstr = inp.args[0].split("->")
                    its_inp_einstrs = its_inp_einstrs.split(",")
                    assert len(its_out_einstr) == len(our_inp_einstrs[inp_idex])
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
