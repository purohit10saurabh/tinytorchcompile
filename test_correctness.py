"""tinytorchcompile must match numpy and torch, and actually fuse: each case pins the exact
fused/unfused kernel counts, so a fusion regression - or fusing across a reduction/matmul
boundary - fails."""
import numpy as np
import pytest
import torch

import tinytorchcompile as ttc

TOL = dict(rtol=1e-9, atol=1e-9)
R = np.random.default_rng(0).standard_normal

CASES = [
    ("pointwise chain", lambda a, b: (a * b + a).relu(), [R(200), R(200)], 1, 3, None),
    ("scalar consts inline", lambda a, b: (a * 2.0 - b + 1.0).relu(), [R(128), R(128)], 1, 4, None),
    ("chain into reduction", lambda a, b: (a * b + a).relu().sum(), [R(300), R(300)], 1, 4, None),
    ("div exp", lambda a, b: a.exp() / (b.exp() + 1.0), [R(100), R(100)], 1, 4, None),
    ("softmax breaks at reduction", lambda x: x.exp() / x.exp().sum(axis=1), [R((16, 32))], 2, 4,
     lambda x: x.exp() / x.exp().sum(dim=1, keepdim=True)),
    ("neg sub", lambda a, b: (-(a - b)).relu(), [R(150), R(150)], 1, 3, None),
    ("matmul boundary, epilogue fuses", lambda x, w, b: (x @ w + b).relu(), [R((8, 4)), R((4, 5)), R(5)], 2, 3, None),
    ("mlp fuses each epilogue", lambda x, w1, b1, w2, b2: (x @ w1 + b1).relu() @ w2 + b2,
     [R((16, 8)), R((8, 12)), R(12), R((12, 4)), R(4)], 4, 5, None),
]


@pytest.mark.parametrize("name, fn, arrays, n_fused, n_unfused, torchfn", CASES, ids=[c[0] for c in CASES])
def test_fusion(name, fn, arrays, n_fused, n_unfused, torchfn):
    ins = [ttc.input(a.shape) for a in arrays]
    ref = ttc.eval_numpy(fn(*ins), ins, arrays)
    fused, unfused = ttc.compile_graph(fn, arrays, fuse=True), ttc.compile_graph(fn, arrays, fuse=False)
    torch_out = (torchfn or fn)(*[torch.from_numpy(a) for a in arrays]).numpy().reshape(ref.shape)
    assert np.allclose(fused(*arrays), ref, **TOL)
    assert np.allclose(unfused(*arrays), ref, **TOL)
    assert np.allclose(torch_out, ref, **TOL)
    assert (fused.num_kernels, unfused.num_kernels) == (n_fused, n_unfused)


def test_decorator_recompiles_on_new_shape():
    @ttc.compile
    def g(a, b):
        return (a * b).relu().sum()

    a, b = R(50), R(50)
    assert np.allclose(g(a, b), np.maximum(a * b, 0.0).sum(), **TOL)
    g(a, b)
    assert len(g.cache) == 1
    g(R(20), R(20))
    assert len(g.cache) == 2
