import pytest

import operator

from torch import fx

from opt_einsum_fx.fx_utils import deduplicate


@pytest.mark.parametrize("n_duplicates", [0, 1, 2, 10])
def test_dedup(n_duplicates: int):
    # make a graph with redundancy
    g = fx.Graph()
    x = g.placeholder("x")
    n1 = g.call_function(operator.mul, args=(2.0, x))
    # add duplicates
    duplicates = [
        g.call_function(operator.mul, args=(2.0, x)) for _ in range(n_duplicates)
    ]
    n3 = g.call_function(operator.mul, args=(2.7, x))  # not same
    out = g.call_function(operator.add, args=(n1, n3))
    for d in duplicates:
        out = g.call_function(operator.add, args=(out, d))  # return n1 + n2 + n3 + ...
    g.output(out)
    # dedup
    n_nodes_orig = len(g.nodes)
    n_removed = deduplicate(g)
    assert len(g.nodes) == n_nodes_orig - n_duplicates
    assert n_removed == n_duplicates
