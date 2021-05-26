from typing import Optional
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

    # from https://github.com/HIPS/autograd/blob/master/autograd/numpy/numpy_vjps.py#L601
    # > subscripts that only appear in subs_wrt (and not in other subscript lists
    # > or in the output) are implicitly being summed out, as if contracted
    # > against a tensor of ones. we make that tensor of ones explicit to handle
    # > the necessary vjp broadcasting inside einsum.
    # So we do similarly:
    new_labels = (
        [grad_out_labels] + op_labels[:wrt] + op_labels[wrt + 1 :]
    )  # without wrt
    new_operands = [grad_out] + operands[:wrt] + operands[wrt + 1 :]
    naked_labels = list(set(op_labels[wrt]) - set("".join(new_labels)))
    if len(naked_labels) > 0:
        wrt_proxy = fx.Proxy(operands[wrt])
        wrt_shape = wrt_proxy.shape
        naked_shape = tuple(
            wrt_shape[op_labels[wrt].index(label)].node for label in naked_labels
        )
        naked_ones = einnode.graph.call_function(
            torch.ones,
            args=(naked_shape,),
            kwargs={"device": wrt_proxy.device.node, "dtype": wrt_proxy.dtype.node},
        )
        new_labels.append("".join(naked_labels))
        new_operands.append(naked_ones)

    new_einstr = ",".join(new_labels) + "->" + op_labels[wrt]
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


def _prod(x):
    """Compute the product of a sequence."""
    out = 1
    for a in x:
        out *= a
    return out


def _reshape_grad(node: fx.Node, wrt: int, grad_out: fx.Node):
    """
    The node must have `in_shape` set.
    """
    if wrt != 0:
        return None
    assert node.target in ("reshape", "view")
    orig = node.args[0]
    assert isinstance(orig, fx.Node)
    shape = node.args[1:]
    if len(shape) == 1 and isinstance(shape[0], tuple):
        shape = shape[0]
    in_shape = getattr(node, "in_shape", None)
    if in_shape is None:
        raise RuntimeError(
            "Differentiating reshape requires static input shape information in `in_shape`."
        )
    assert isinstance(in_shape, tuple)
    return node.graph.call_method(node.target, args=(grad_out, in_shape))


def _mmgrad(node: fx.Node, wrt: int, grad_out: fx.Node):
    assert wrt == 1
    m = node.args[0]
    mt = node.graph.call_method("t", args=(m,))
    return node.graph.call_function(torch.mm, args=(mt, grad_out))


# Mapping from target function to function of signature
# f(node: fx.Node, wrt: int, grad_out: fx.Node) -> fx.Node
_GRAD_FUNCS = {
    torch.einsum: _eingrad,
    operator.mul: _mulgrad,
    "mul": _mulgrad,
    operator.truediv: _mulgrad,
    "div": _mulgrad,
    "reshape": _reshape_grad,
    "view": _reshape_grad,
    torch.mm: _mmgrad,
}


def _accumulate(n1: Optional[fx.Node], n2: Optional[fx.Node]) -> fx.Node:
    if n1 is None:
        return n2
    elif n2 is None:
        return n1
    else:
        return n1.graph.call_function(operator.add, args=(n1, n2), type_expr=n1.type)


def grad(out: fx.Node, grad_out: fx.Node, wrt: fx.Node) -> Optional[fx.Node]:
    assert isinstance(out, fx.Node)
    assert isinstance(wrt, fx.Node)
    out.graph.lint()
    # requires_grad
    requires_grad = {wrt}
    for node in wrt.graph.nodes:
        if node in requires_grad:
            requires_grad.update(node.users)

    # backpropagate
    out.grad = grad_out

    node = out
    while True:
        for arg_i, arg in enumerate(node.args):
            if not isinstance(arg, fx.Node):
                continue
            if arg not in requires_grad:
                continue
            if arg.type is not None and arg.type != torch.Tensor:
                # Assume like torchscript no type => tensor
                continue
            grad_func = _GRAD_FUNCS.get(node.target, None)
            if grad_func is None:
                raise NotImplementedError(f"Unsupported func `{node.target}`")
            arg.grad = _accumulate(
                getattr(arg, "grad", None), grad_func(node, arg_i, node.grad)
            )
        while True:
            node = node.prev
            if node == wrt:
                break
            if getattr(node, "grad", None) is not None:
                break
        if node == wrt:
            break

    final_grad = wrt.grad

    # clear state
    for node in requires_grad:
        if hasattr(node, "grad"):
            del node.grad

    out.graph.lint()

    return final_grad
