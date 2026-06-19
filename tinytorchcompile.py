"""
tinytorchcompile: a tiny version (500 lines) of torch.compile showing its main idea: operator fusion.

How it works
------------
You do regular pytorch operations on tensors, like `(w * x + b).relu().sum()`. These are
"lazy tensors": instead of computing right away, each operation just records what you asked
for. tinytorchcompile then looks at the recorded chain, writes a small C program that does all
of it in a single loop, compiles that C with your system compiler, and runs it. The result
matches NumPy, but it's faster.

Why a single loop is the whole point
------------------------------------
Run `(w * x + b).relu().sum()` the normal way and each step walks the entire array and
writes a brand-new array back to memory: one pass for the multiply, one for the add, one
for relu, one for the sum. These memory transfers are the bottleneck, not the computation.

"Operator fusion" means doing all four steps for one element before moving to the next, so
the in-between arrays are never created. Same answer, a fraction of the memory traffic.
Fusing the lazy tensor's recorded chain into one loop is the single trick behind why
`torch.compile` is fast; the PyTorch 2 paper shows that *without* fusion a compiler is no
faster than plain eager code. Everything else (using all CPU cores, SIMD, GPUs) is built on
top of this.

How it works, in four steps
---------------------------
1. Trace   - the lazy tensor records each operation into a small graph instead of running it.
2. Lower   - turn each node into a "recipe" for computing one output element.
3. Fuse    - operator fusion: inline those recipes into each other so a chain becomes one loop.
4. Codegen - print the loop as C, compile it, and call it.

When one loop isn't possible (fusion boundaries)
------------------------------------------------
Steps fuse only when each output element depends just on the *matching* input elements. If
a later step needs the *whole* result of an earlier one, you can't merge them and must keep
that intermediate in memory. Row-wise softmax, `x.exp() / x.exp().sum(axis=1)`, is the
classic case: the divide needs the complete row sum first, so it compiles to two loops, not
one. Reductions (`sum`) feeding later ops, and `matmul`, are these boundaries - `schedule`
only inlines pointwise producers and stops at them, so softmax comes out as two kernels.
`matmul` itself can't fuse with a pointwise producer, but the pointwise ops that *follow* it
do fuse: an MLP layer `(x @ w + b).relu()` becomes a matmul kernel plus one fused kernel for
the bias-add and relu epilogue - the same shape of win that TorchInductor gets on real layers.

Where to look in this file
--------------------------
- `Tensor`       - the lazy tensor; operators just record graph nodes.
- `lower`        - turns each node into a Buffer whose `inner(index)` is the per-element recipe.
- `schedule`     - the greedy fuser; encodes the fusion boundaries above.
- `emit_pointwise` / `emit_reduction` / `emit_matmul` - print the C for each kind of loop.
- `compile` / `compile_graph` - the user-facing entry points; run `python tinytorchcompile.py` for a demo.

References: the fusion-is-the-win result is from the PyTorch 2 paper: "PyTorch 2: Faster Machine
Learning Through Dynamic Python Bytecode Transformation and Graph Compilation" (Ansel et al.,
ASPLOS 2024). This file models only the codegen/fusion backend (TorchInductor), not Dynamo
or autograd.
"""
from __future__ import annotations

import ctypes
import hashlib
import os
import shutil
import subprocess
import tempfile
import threading
from contextlib import contextmanager

import numpy as np

POINTWISE = {"add", "sub", "mul", "div", "neg", "relu", "exp"}  # these fuse; reductions/matmul are boundaries


def broadcast_shape(a, b):
    nd = max(len(a), len(b))
    a, b = (1,) * (nd - len(a)) + tuple(a), (1,) * (nd - len(b)) + tuple(b)
    out = []
    for x, y in zip(a, b):
        if x == y or x == 1 or y == 1:
            out.append(max(x, y))
        else:
            raise ValueError("cannot broadcast %s vs %s" % (a, b))
    return tuple(out)


class Tensor:
    """ a lazy tensor: every op records a node in the graph instead of computing """
    _n = 0

    def __init__(self, op, inputs, shape, const=None, axis=None):
        self.op, self.inputs, self.shape = op, list(inputs), tuple(shape)
        self.const, self.axis = const, axis
        self.name = "b%d" % Tensor._n
        Tensor._n += 1

    def _wrap(self, x):
        return x if isinstance(x, Tensor) else Tensor("const", [], (), const=float(x))

    def _ew(self, op, o, swap=False):  # elementwise op with numpy broadcasting
        a, b = self, self._wrap(o)
        if swap:
            a, b = b, a
        return Tensor(op, [a, b], broadcast_shape(a.shape, b.shape))

    def __add__(s, o): return s._ew("add", o)
    def __radd__(s, o): return s._ew("add", o, True)
    def __sub__(s, o): return s._ew("sub", o)
    def __rsub__(s, o): return s._ew("sub", o, True)
    def __mul__(s, o): return s._ew("mul", o)
    def __rmul__(s, o): return s._ew("mul", o, True)
    def __truediv__(s, o): return s._ew("div", o)
    def __rtruediv__(s, o): return s._ew("div", o, True)
    def __neg__(s): return Tensor("neg", [s], s.shape)
    def relu(s): return Tensor("relu", [s], s.shape)
    def exp(s): return Tensor("exp", [s], s.shape)

    def matmul(s, o):
        assert isinstance(o, Tensor) and len(s.shape) == 2 and len(o.shape) == 2 and s.shape[1] == o.shape[0], \
            "bad matmul shapes %s @ %s" % (s.shape, getattr(o, "shape", None))
        return Tensor("matmul", [s, o], (s.shape[0], o.shape[1]))

    def __matmul__(s, o): return s.matmul(o)

    def sum(s, axis=None):  # full reduction, or over the last axis
        if axis is None:
            return Tensor("sum", [s], (1,), axis=None)
        if axis < 0:
            axis += len(s.shape)
        return Tensor("sum", [s], tuple(1 if i == axis else d for i, d in enumerate(s.shape)), axis=axis)

    def __repr__(s): return "Tensor(%s, shape=%s)" % (s.op, s.shape)


def input(shape):
    return Tensor("input", [], tuple(shape))


def topo(root):  # graph nodes, producers before consumers
    order, seen = [], set()

    def go(v):
        if id(v) in seen:
            return
        seen.add(id(v))
        for c in v.inputs:
            go(c)
        order.append(v)

    go(root)
    return order


# V.ops holds the active interpreter. The same IR body (a closure) means different things
# depending on which handler is installed - this is TorchInductor's central trick.
class _V(threading.local):
    ops = None


V = _V()


@contextmanager
def use_ops(handler):
    prev = V.ops
    V.ops = handler
    try:
        yield
    finally:
        V.ops = prev


class Index:
    def __init__(self, shape, coords):
        self.shape, self.coords = tuple(shape), tuple(coords)


def cstrides(shape):  # row-major (C-contiguous) strides
    st = [1] * len(shape)
    for i in range(len(shape) - 2, -1, -1):
        st[i] = st[i + 1] * shape[i + 1]
    return st


def bstrides(in_shape, out_shape):  # broadcasting: stride 0 on the size-1 dims
    nd = len(out_shape)
    in_shape = (1,) * (nd - len(in_shape)) + tuple(in_shape)
    st = cstrides(in_shape)
    return [0 if in_shape[i] == 1 and out_shape[i] != 1 else st[i] for i in range(nd)]


def offset(coords, strides):
    terms = [str(c) if s == 1 else "(%s*%d)" % (c, s) for c, s in zip(coords, strides) if s != 0]
    return " + ".join(terms) if terms else "0"


class Analysis:
    """ interpreter: record which buffers the body reads (for deps + fusion legality) """
    def __init__(self): self.reads = []
    def constant(self, v): return 0.0
    def load(self, buf, idx): self.reads.append(buf); return 0.0
    def add(self, a, b): return 0.0
    def sub(self, a, b): return 0.0
    def mul(self, a, b): return 0.0
    def div(self, a, b): return 0.0
    def neg(self, a): return 0.0
    def relu(self, a): return 0.0
    def exp(self, a): return 0.0


class Codegen:
    """ interpreter: emit C text for the body, e.g. mul(a, b) -> "(a * b)" """
    def constant(self, v): return repr(float(v))
    def load(self, buf, idx): return "%s[%s]" % (buf.cname, offset(idx.coords, bstrides(buf.shape, idx.shape)))
    def add(self, a, b): return "(%s + %s)" % (a, b)
    def sub(self, a, b): return "(%s - %s)" % (a, b)
    def mul(self, a, b): return "(%s * %s)" % (a, b)
    def div(self, a, b): return "(%s / %s)" % (a, b)
    def neg(self, a): return "(-%s)" % a
    def relu(self, a): return "fmax(%s, 0.0)" % a
    def exp(self, a): return "exp(%s)" % a


class Buffer:
    """ a lowered buffer; its loop body is the closure inner(index) """
    def __init__(self, node, shape, inner, is_input=False, is_reduction=False, in_shape=None, axis=None, is_matmul=False):
        self.node, self.name, self.shape = node, node.name, tuple(shape)
        self.inner, self.is_input, self.is_reduction = inner, is_input, is_reduction
        self.in_shape = tuple(in_shape) if in_shape else None
        self.axis, self.is_matmul, self.inlined, self.cname = axis, is_matmul, False, node.name
        self.a_buf = self.b_buf = self.a_shape = self.b_shape = None


def read(buf, idx):  # fused producer -> recompute its body inline; else load from memory
    if buf.inlined and not buf.is_input:
        return buf.inner(idx)
    return V.ops.load(buf, idx)


def lower(root):  # each graph node becomes a buffer with a define-by-run loop body
    order, bufs = topo(root), {}
    for n in order:
        if n.op == "input":
            bufs[id(n)] = Buffer(n, n.shape, None, is_input=True)
        elif n.op == "const":
            b = Buffer(n, (), (lambda idx, c=n.const: V.ops.constant(c)))
            b.inlined = True
            bufs[id(n)] = b
        elif n.op in POINTWISE:
            ins = [bufs[id(c)] for c in n.inputs]
            bufs[id(n)] = Buffer(n, n.shape, (lambda idx, op=n.op, ins=ins: getattr(V.ops, op)(*[read(x, idx) for x in ins])))
        elif n.op == "sum":
            src = bufs[id(n.inputs[0])]
            bufs[id(n)] = Buffer(n, n.shape, (lambda idx, src=src: read(src, idx)),
                                 is_reduction=True, in_shape=n.inputs[0].shape, axis=n.axis)
        elif n.op == "matmul":
            b = Buffer(n, n.shape, None, is_matmul=True)
            b.a_buf, b.b_buf = bufs[id(n.inputs[0])], bufs[id(n.inputs[1])]
            b.a_shape, b.b_shape = n.inputs[0].shape, n.inputs[1].shape
            bufs[id(n)] = b
    return order, bufs


def materialized(b):
    return (not b.inlined) and (not b.is_input)


def schedule(order, bufs, fuse=True):  # fusion = inline pointwise producers; matmul is a boundary
    if not fuse:
        return [bufs[id(n)] for n in order if materialized(bufs[id(n)])]
    users = {}
    for n in order:
        for c in n.inputs:
            users.setdefault(id(c), []).append(n)
    while True:  # greedily inline the largest fusable producer until a fixpoint
        best = None
        for n in order:
            b = bufs[id(n)]
            if b.node.op not in POINTWISE or b.inlined:
                continue
            consumers = [bufs[id(c)] for c in users.get(id(n), [])]
            if not consumers or any(c.is_matmul for c in consumers):
                continue
            size = prod(b.shape)
            if best is None or size > best[0]:
                best = (size, b)
        if best is None:
            break
        best[1].inlined = True
    return [bufs[id(n)] for n in order if materialized(bufs[id(n)])]


def prod(shape):
    p = 1
    for s in shape:
        p *= s
    return p


def deps_of(b):  # replay the body under Analysis to find the buffers it reads
    if b.is_matmul:
        uniq = {d.cname: d for d in (b.a_buf, b.b_buf)}
        return [uniq[k] for k in sorted(uniq)]
    a = Analysis()
    shape = b.in_shape if b.is_reduction else b.shape
    with use_ops(a):
        b.inner(Index(shape, [0] * len(shape)))
    uniq = {d.cname: d for d in a.reads}
    return [uniq[k] for k in sorted(uniq)]


def params(deps, out):
    return ", ".join(["double* " + d.cname for d in deps] + ["double* " + out.cname])


def emit_pointwise(b, deps):  # one loop nest; the fused chain is one C expression
    s = b.shape
    v = ["i%d" % k for k in range(len(s))]
    with use_ops(Codegen()):
        expr = b.inner(Index(s, v))
    loops = "".join("for(long %s=0;%s<%d;%s++){" % (v[k], v[k], s[k], v[k]) for k in range(len(s)))
    body = "%s%s[%s] = %s;%s" % (loops, b.cname, offset(v, cstrides(s)), expr, "}" * len(s))
    return "static void kernel_%s(%s) {\n  %s\n}" % (b.name, params(deps, b), body)


def emit_reduction(b, deps):  # accumulate over the last axis
    ish, outer, red = b.in_shape, b.in_shape[:-1], b.in_shape[-1]
    ov = ["o%d" % k for k in range(len(outer))]
    with use_ops(Codegen()):
        elem = b.inner(Index(ish, ov + ["k"]))
    outer_loops = "".join("for(long %s=0;%s<%d;%s++){" % (ov[k], ov[k], outer[k], ov[k]) for k in range(len(outer)))
    inner = "double acc=0.0; for(long k=0;k<%d;k++){ acc = acc + (%s); } %s[%s]=acc;" % (
        red, elem, b.cname, offset(ov + ["0"], cstrides(b.shape)))
    return "static void kernel_%s(%s) {\n  %s\n}" % (b.name, params(deps, b), outer_loops + inner + "}" * len(outer))


def emit_matmul(b, deps):  # naive triple loop (BLAS-grade tiling is left out on purpose)
    m, k = b.a_shape
    _, n = b.b_shape
    body = ("for(long i=0;i<%d;i++) for(long j=0;j<%d;j++){ double acc=0.0; "
            "for(long p=0;p<%d;p++){ acc += %s[i*%d+p] * %s[p*%d+j]; } %s[i*%d+j]=acc; }") % (
        m, n, k, b.a_buf.cname, k, b.b_buf.cname, n, b.cname, n)
    return "static void kernel_%s(%s) {\n  %s\n}" % (b.name, params(deps, b), body)


def emit(b, deps):
    if b.is_matmul:
        return emit_matmul(b, deps)
    return emit_reduction(b, deps) if b.is_reduction else emit_pointwise(b, deps)


def wrapper(mats, in_bufs, root, depsmap):  # alloc intermediates, call kernels in order, free
    sig = ", ".join(["double* " + b.cname for b in in_bufs] + ["double* out"])
    lines = ["double* %s=(double*)malloc(sizeof(double)*%d);" % (b.cname, prod(b.shape)) for b in mats if b is not root]
    lines += ["kernel_%s(%s);" % (b.name, ", ".join([d.cname for d in depsmap[b.name]] + [b.cname])) for b in mats]
    lines += ["free(%s);" % b.cname for b in mats if b is not root]
    return "void run(%s) {\n  %s\n}" % (sig, "\n  ".join(lines))


def codegen(order, bufs, ins, root):  # one kernel per materialized buffer + a wrapper
    in_bufs = [bufs[id(n)] for n in ins]
    for k, b in enumerate(in_bufs):
        b.cname = "in%d" % k
    rb = bufs[id(root)]
    mats = [bufs[id(n)] for n in order if materialized(bufs[id(n)])]
    for b in mats:
        b.cname = "out" if b is rb else "t_" + b.name
    depsmap = {b.name: deps_of(b) for b in mats}
    kernels = [emit(b, depsmap[b.name]) for b in mats]
    return "#include <math.h>\n#include <stdlib.h>\n\n" + "\n\n".join(kernels) + "\n\n" + wrapper(mats, in_bufs, rb, depsmap)


def eval_numpy(root, inputs, arrays):
    """ reference oracle: run the graph eagerly in numpy (what the tests check against) """
    feed = {id(n): np.ascontiguousarray(a, np.float64) for n, a in zip(inputs, arrays)}
    binop = {"add": lambda x, y: x + y, "sub": lambda x, y: x - y, "mul": lambda x, y: x * y, "div": lambda x, y: x / y}
    unop = {"neg": lambda x: -x, "relu": lambda x: np.maximum(x, 0.0), "exp": np.exp}
    v = {}
    for n in topo(root):
        if n.op == "input":
            v[id(n)] = feed[id(n)]
        elif n.op == "const":
            v[id(n)] = np.float64(n.const)
        elif n.op in binop:
            v[id(n)] = binop[n.op](v[id(n.inputs[0])], v[id(n.inputs[1])])
        elif n.op in unop:
            v[id(n)] = unop[n.op](v[id(n.inputs[0])])
        elif n.op == "matmul":
            v[id(n)] = v[id(n.inputs[0])] @ v[id(n.inputs[1])]
        elif n.op == "sum":
            x = v[id(n.inputs[0])]
            v[id(n)] = np.asarray(x.sum()).reshape(1) if n.axis is None else x.sum(axis=n.axis, keepdims=True)
    return v[id(root)]


_CACHE = {}


def _compiler():
    for cc in ("clang", "gcc", "cc"):
        if shutil.which(cc):
            return cc
    raise RuntimeError("no C compiler (clang/gcc/cc) on PATH")


def _load(csrc, n_inputs):  # compile C to a .so, load run() via ctypes, cache by source hash
    key = hashlib.sha1(csrc.encode()).hexdigest()
    if key in _CACHE:
        return _CACHE[key]
    d = os.path.join(tempfile.gettempdir(), "tinytorchcompile")
    os.makedirs(d, exist_ok=True)
    cpath, sopath = os.path.join(d, key + ".c"), os.path.join(d, key + ".so")
    if not os.path.exists(sopath):
        open(cpath, "w").write(csrc)
        subprocess.run([_compiler(), "-O2", "-shared", "-fPIC", "-o", sopath, cpath], check=True)
    fn = ctypes.CDLL(sopath).run
    fn.argtypes = [ctypes.POINTER(ctypes.c_double)] * (n_inputs + 1)
    fn.restype = None
    _CACHE[key] = fn
    return fn


def _build(fn, arrays, fuse):  # trace -> lower -> schedule -> codegen -> compile
    ins = [input(a.shape) for a in arrays]
    root = fn(*ins)
    order, bufs = lower(root)
    mats = schedule(order, bufs, fuse=fuse)
    csrc = codegen(order, bufs, ins, root)
    run, out_shape, out_size = _load(csrc, len(ins)), root.shape, prod(root.shape)

    def call(*arr):
        ca = [np.ascontiguousarray(a, np.float64).ravel() for a in arr]
        out = np.empty(out_size, np.float64)
        run(*[a.ctypes.data_as(ctypes.POINTER(ctypes.c_double)) for a in ca],
            out.ctypes.data_as(ctypes.POINTER(ctypes.c_double)))
        return out.reshape(out_shape)

    call.csrc, call.num_kernels, call.num_intermediates = csrc, len(mats), len(mats) - 1
    return call


def compile_graph(fn, example_inputs, fuse=True):
    arrays = [np.ascontiguousarray(a, np.float64) for a in example_inputs]
    return _build(fn, arrays, fuse)


def compile(fn=None, *, fuse=True):
    """ torch.compile-style: trace on first call, guard on shape/dtype, cache the artifact """

    def wrap(fn):
        cache = {}

        def runner(*arrays):
            a = [np.ascontiguousarray(x, np.float64) for x in arrays]
            key = tuple((x.shape, x.dtype.str) for x in a)  # the guard: a new shape recompiles
            if key not in cache:
                cache[key] = _build(fn, a, fuse)
                runner.csrc, runner.num_kernels = cache[key].csrc, cache[key].num_kernels
            return cache[key](*a)

        runner.cache = cache
        runner.__name__ = getattr(fn, "__name__", "compiled")
        return runner

    return wrap(fn) if callable(fn) else wrap


if __name__ == "__main__":
    import time

    np.random.seed(0)
    w, x, b = (np.random.randn(2_000_000) for _ in range(3))

    def f(w, x, b):
        return (w * x + b).relu().sum()

    def eager(w, x, b):
        return np.maximum(w * x + b, 0.0).sum()

    fused, unfused = compile(f, fuse=True), compile(f, fuse=False)
    ref = eager(w, x, b)
    rf, ru = fused(w, x, b), unfused(w, x, b)
    print("fused kernels:   %d" % fused.num_kernels)
    print("unfused kernels: %d" % unfused.num_kernels)
    print("correct:         %s" % (np.allclose(rf, ref) and np.allclose(ru, ref)))

    def bench(g):
        for _ in range(3):
            g(w, x, b)
        t = time.perf_counter()
        for _ in range(30):
            g(w, x, b)
        return (time.perf_counter() - t) / 30 * 1e3

    print("eager numpy: %.2f ms   unfused: %.2f ms   ->   fused: %.2f ms" % (
        bench(eager), bench(unfused), bench(fused)))
    print("\n--- generated fused C ---\n" + fused.csrc)
