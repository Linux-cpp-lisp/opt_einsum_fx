from typing import List, Tuple, Union, Callable
import copy

from torch import fx
from functorch import make_fx


def _count_unique_unhashable(iterable):
    """Count 'unique' objects in `iterable` by `==`, even if unhashable

    Assumes that `==` is transitive.
    """
    count: int = 0
    seen = []
    for process in iterable:
        if not any(e == process for e in seen):
            # not previously seen
            count += 1
            seen.append(process)
    return count


def make_fx_dynamic(f: Callable, example_inputs: List[tuple]) -> fx.GraphModule:
    """Like ``functorch.make_fx``, but tries to make a graph that can handle (trivial) dynamic shapes.

    Any shape that should be dynamic must have multiple values in the `example_inputs`.
    Because this code uses heuristics based on the shapes in the trace to make the graph
    dynamic, "unusual" values for the sizes of the variable dimensions are probably safest.

    **Importantly**, if two different dynamic dimensions have the same sizes across all of
    the ``example_inputs``, they *are assumed to always be the same size*.

    Args:
        f: the function to ``make_fx``, which takes and returns a tuple of tensors.
        example_inputs: a list of example inputs.

    Returns:
        An ``fx.GraphModule``.
    """
    if len(example_inputs) == 1:
        # nothing can be dynamic, short circut:
        return make_fx(f)(*example_inputs[0])

    # Determine which shapes are dynamic for each argument
    dynamic_shapes: List[Tuple[Union[List[int], int], ...]] = []
    dynamic_shape_sizes_to_arg = {}
    for arg_i, values in enumerate(zip(*example_inputs)):
        assert (
            _count_unique_unhashable(t.ndim for t in values) == 1
        ), "all should have same ndim"
        dynamic_shape = [list() for _ in values[0].shape]
        for shape in (t.shape for t in values):
            for i, e in enumerate(shape):
                dynamic_shape[i].append(e)
        # check if is dynamic:
        dynamic_shape = [
            s[0] if _count_unique_unhashable(s) == 1 else s for s in dynamic_shape
        ]
        dynamic_shapes.append(dynamic_shape)
        for dim_i, s in enumerate(dynamic_shape):
            if isinstance(s, list):
                dynamic_shape_sizes_to_arg[tuple(s)] = (arg_i, dim_i)

    # Trace out the function with each of the example inputs
    f_tracer = make_fx(f)
    graphmods: List[fx.GraphModule] = [f_tracer(*e) for e in example_inputs]
    assert (
        _count_unique_unhashable(len(gm.graph.nodes) for gm in graphmods) == 1
    ), "All traces were not the same length!"

    # make a new version of the graph for the output
    # the following loop will check consistancy of all the traces
    graphmod_out = copy.deepcopy(graphmods[0])
    graph_out = graphmod_out.graph
    graph_out_tracer = fx.proxy.GraphAppendingTracer(graph_out)
    graph_out_iter = iter(graph_out.nodes)
    graph_out_arg_proxies = []
    first_real_node_graph_out = None
    for n in graph_out_iter:
        if n.op == "placeholder":
            graph_out_arg_proxies.append(fx.Proxy(n, tracer=graph_out_tracer))
        else:
            # first real node
            first_real_node_graph_out = n
            break

    # make nodes that get the sizes of the dynamic dimensions from the inputs:
    dynamic_size_nodes_out = {}  # (arg_i, dim_i) -> Node
    # insert before first node
    with graph_out.inserting_before(first_real_node_graph_out):
        for arg_i, dim_i in dynamic_shape_sizes_to_arg.values():
            dynamic_size_nodes_out[(arg_i, dim_i)] = (
                graph_out_arg_proxies[arg_i].shape[dim_i].node
            )

    # walk the graph(s) and rewrite them:
    node_out = first_real_node_graph_out
    for node_i, nodes in enumerate(zip(*[gm.graph.nodes for gm in graphmods])):
        if node_i < len(graph_out_arg_proxies):
            # it's an arg node
            assert nodes[0].op == "placeholder"
            continue
        # iterate over the nodes of all the graphs in parallel
        # do a hacky check that the graphs are all the same
        assert (
            _count_unique_unhashable(str(n) for n in nodes) == 1
        ), "Got different nodes at same place in graph across traces!"
        assert str(nodes[0]) == str(node_out), "output graph inconsistant with traces"

        # only arguments that are *not* Nodes are sizes
        # (we only care about "static" sizes that bake in the example input)

        # build a new argument list for the final graph's node
        args_out = []
        for arg_i, args in enumerate(zip(*[n.args for n in nodes])):
            # args is the value of the arg across all the difft nodes
            is_naked_int: bool = False
            if isinstance(args[0], fx.Node):
                # pass through existing arg
                args_out.append(node_out.args[arg_i])
                continue  # don't care not a shape

            if isinstance(args[0], int):
                # make a tuple for DRY
                is_naked_int = True
                args = [(v,) for v in args]

            if (isinstance(args[0], tuple) or isinstance(args[0], list)) and isinstance(
                args[0][0], int
            ):
                # it's a shape
                # make a new shape
                new_shape = []
                for dim_i, dim_sizes in enumerate(zip(*args)):
                    # check each dim for dynamicness
                    if _count_unique_unhashable(dim_sizes) > 1:
                        # it's dynamic
                        # now check that it matches one of our dynamic input dims
                        key = dynamic_shape_sizes_to_arg[tuple(dim_sizes)]
                        # ^ if this lookup fails, there's a dynamic shape
                        #   that is not exactly one from the input
                        #   this will correctly error out, though with uninformative
                        #   IndexError.
                        # now, replace the wrong static constant with the getattr
                        size_node = dynamic_size_nodes_out[key]
                        new_shape.append(size_node)
                    else:
                        # it's a static constant, pass through
                        new_shape.append(dim_sizes[0])
                # now we have a new shape for the argument
                assert len(new_shape) == len(args[0])
                # unwrap naked int
                if is_naked_int:
                    new_arg = new_shape[0]
                else:
                    # convert back to orig list/tuple type
                    new_arg = type(args[0])(new_shape)
            else:
                # it's some other constant or tuple/list of nodes
                # since it could have nodes, must take from graph_out
                new_arg = node_out.args[arg_i]
                if not (
                    (isinstance(args[0], tuple) or isinstance(args[0], list))
                    and isinstance(args[0][0], fx.Node)
                ):
                    # sanity check that its consistent
                    assert _count_unique_unhashable(args) == 1

            # add argument
            args_out.append(new_arg)

        # now, we have an updated args list to put into the output graph
        node_out.args = tuple(args_out)

        # also iterate through the output graph:
        node_out = next(
            graph_out_iter, None
        )  # last iter have a default to avoid StopIteration

    # check the final graph
    graphmod_out.graph.lint()
    graphmod_out.recompile()

    return graphmod_out
