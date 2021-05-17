from typing import Optional, List
import operator
import numbers

import torch
from torch import fx


def _eingrad(einnode: fx.Node, wrt: int, grad_out: fx.Node):
    wrt -= 1
    # ein w/o wrt and only with output labels that wrt has in original
    # add term with grad_out labeled by original out labels
    einstr = einnode.args[0]
    operands = list(einnode.args[1:])
    op_labels, grad_out_labels = einstr.split("->")
    op_labels = op_labels.split(",")
    new_labels = (
        [grad_out_labels] + op_labels[:wrt] + op_labels[wrt + 1 :]
    )  # without wrt
    new_einstr = ",".join(new_labels) + "->" + op_labels[wrt]
    new_operands = [grad_out] + operands[:wrt] + operands[wrt + 1 :]
    return einnode.graph.call_function(
        torch.einsum, args=(new_einstr,) + tuple(new_operands), type_expr=torch.Tensor
    )


def _mulgrad(node: fx.Node, wrt: int, grad_out: fx.Node):
    other = node.args[(wrt + 1) % 2]
    assert isinstance(
        other, numbers.Number
    ), "does not support tensor-tensor multiplication"
    if node.target == operator.truediv or node.target == "div":
        other = 1.0 / other
    elif node.target == operator.mul or node.target == "mul":
        pass
    else:
        raise ValueError(f"Invalid node target `{node.target}` for _mulgrad")
    return node.graph.call_function(operator.mul, args=(other, grad_out))


# Mapping from target function to function of signature
# f(node: fx.Node, wrt: int, grad_out: fx.Node) -> fx.Node
_GRAD_FUNCS = {torch.einsum: _eingrad, operator.mul: _mulgrad, "mul": _mulgrad}


def _accumulate(n1: Optional[fx.Node], n2: Optional[fx.Node]) -> fx.Node:
    if n1 is None:
        return n2
    elif n2 is None:
        return n1
    else:
        return n1.graph.call_function(operator.add, args=(n1, n2), type_expr=n1.type)


def grad(out: fx.Node, grad_out: fx.Node, wrt: fx.Node) -> Optional[fx.Node]:
    # backpropagate
    out.grad = grad_out
    has_grad = {out}

    node = out
    while True:
        for arg_i, arg in enumerate(node.args):
            if not isinstance(arg, fx.Node):
                continue
            grad_func = _GRAD_FUNCS.get(node.target, None)
            if grad_func is None:
                raise NotImplementedError(f"Unsupported func `{node.target}`")
            arg.grad = _accumulate(
                getattr(arg, "grad", None), grad_func(node, arg_i, node.grad)
            )
            has_grad.add(arg)
        while True:
            node = node.prev
            if getattr(node, "grad", None) is not None:
                break
        if node == wrt:
            break

    final_grad = wrt.grad

    # clear state
    for node in has_grad:
        del node.grad

    out.graph.lint()

    return final_grad
