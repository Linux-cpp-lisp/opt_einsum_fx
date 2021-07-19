import argparse
import logging

import torch
from torch import fx
from torch.utils.benchmark import Timer

from opt_einsum_fx import optimize_einsums_full, jitable
from opt_einsum_fx._fuse import _get_einstrs


# https://stackoverflow.com/a/15008806/1008938
def t_or_f(arg):
    ua = str(arg).upper()
    if "TRUE".startswith(ua):
        return True
    elif "FALSE".startswith(ua):
        return False
    else:
        raise ValueError(str(arg))


def main():
    parser = argparse.ArgumentParser(prog="benchmark")
    parser.add_argument("einstr", type=str)
    parser.add_argument("modes", type=str, help="mode sizes in format i=3,j=7,k=10")
    parser.add_argument("--jit", type=t_or_f, default=True)
    parser.add_argument("--cuda", type=t_or_f, default=True)
    parser.add_argument("--backward", type=t_or_f, default=True)
    parser.add_argument("--opt-ein", type=t_or_f, default=True)
    parser.add_argument("--cuTENSOR", type=t_or_f, default=True)
    parser.add_argument("-n", type=int, default=1000)

    args = parser.parse_args()

    device = "cuda" if (torch.cuda.is_available() and args.cuda) else "cpu"
    args.cuda = device == "cuda"

    print("======= Benchmark with settings: ======")
    for key, val in vars(args).items():
        print(f"{key:>18} : {val}")
    print("=" * 40)

    # Make inputs
    modes = {}
    for tmp in args.modes.split(","):
        k, v = tmp.split("=")
        assert len(k) == 1
        modes[k] = int(v)

    subs, out_subs = _get_einstrs(args.einstr)
    shapes = []
    for sub in subs:
        shapes.append(tuple(modes[k] for k in sub))

    # from https://pytorch.org/docs/master/_modules/torch/utils/benchmark/utils/timer.html#Timer.timeit
    warmup = max(int(args.n // 100), 1)

    inputs = iter(
        [
            tuple(
                torch.randn(shape, device=device, requires_grad=args.backward)
                for shape in shapes
            )
            for _ in range(args.n + warmup + 1)
        ]
    )

    # Make kernel
    graph = fx.Graph()
    inps = [graph.placeholder(f"x{i}") for i in range(len(shapes))]
    out = graph.call_function(torch.einsum, args=tuple([args.einstr] + inps))
    graph.output(out)
    gmod = fx.GraphModule({}, graph)
    if args.opt_ein:
        gmod = optimize_einsums_full(gmod, next(inputs), use_cuTENSOR=args.cuTENSOR)
    # compile
    if args.jit:
        gmod = jitable(gmod)
        gmod.recompile()
        gmod = torch.jit.script(gmod)

    print("starting...")

    # tanh() forces it to realize the grad as a full size matrix rather than expanded (stride 0) ones
    t = Timer(
        stmt=(
            "gmod.zero_grad()\n"
            "out = gmod(*next(inputs))\n"
            + ("out.tanh().sum().backward()\n" if args.backward else "")
        ),
        globals={"gmod": gmod, "inputs": inputs},
    )

    perloop = t.timeit(args.n)

    print()
    print(perloop)


if __name__ == "__main__":
    main()
