# `torch.compile` in a nutshell, showing its main idea: **operator fusion**. 

*TL;DR: `torch.compile` traces a lazy tensor expression and fuses the sequence of ops into a [highly optimized code](demo.ipynb#X10sZmlsZQ==) in Triton for GPUs and C++ for CPUs without creating large intermediate tensors. Hence, there are no memory transfers of intermediate tensors from RAM to processor and back for each pytorch operation. This is how it reduces the runtime significantly. Play with it in the attached [demo notebook](demo.ipynb).*

![license](https://img.shields.io/badge/license-MIT-green)
![python](https://img.shields.io/badge/python-3.9%2B-blue)
![deps](https://img.shields.io/badge/runtime%20deps-numpy-orange)

## Why is operator fusion the heart of `torch.compile`?

In the attached [demo notebook](demo.ipynb), the same expression is run and timedin four different ways:
- torch eager: 11.7 ms (4 kernels) --> runs the expression in eager mode (running each python line sequentially) without any compilation.
- torch.compile: 1.9 ms (1 kernel) --> compiles the expression into a highly optimized single C++ loop and runs it.
- tinytorchcompile unfused: 29.9 ms (4 kernels) --> compiled into C code but NOT fused into one loop. It has 4 loops transfering the intermediate arrays between RAM and processor. This is even slower than eager mode as shown in the PyTorch 2 paper.
- tinytorchcompile fused: 7.8 ms (1 kernel) --> compiled into C code and fused into one nested loop, removing intermediate arrays and transfering them between RAM and processor.

This shows that without operator fusion, compiling is no faster than eager because it has the same memory transfers but without the optimized numpy operations. Operator fusion removes the memory transfers of intermediate arrays from RAM to processor (like on GPU's HBM to SRAM) making the runtime no longer memory-bound. It alone led to a 3.8x (tinytorchcompile fused vs tinytorchcompile unfused) speedup. On top of this, other optimizations in `torch.compile` like multithreading and SIMD instructions work only because the memory bottleneck is removed by operator fusion. In above example, they led to an additional 2.3x speedup (torch.compile is 11.7ms/1.9ms = 6.1x faster than torch eager, 3.8x is due to operator fusion and 2.3x is due to other optimizations). To understand it theoretically, suppose that half of the total runtime is taken by memory transfers and other half is taken in computation like matrix multiplications, and operator fusion completely removes the time in memory transfer. This gives a speedup of 2x. Moreover, an optimization like multithreading on 10 threads would speed up the computation by 10x. Without operator fusion, it would only give a speedup of 5x because half of the time is taken by memory transfers. But with operator fusion, multithreading speeds up the total time by 10x. Hence, operator fusion speeds up each other optimization as well.

Example of operator fusion:
Unfused, each of the 4 ops is its own kernel writing a full intermediate array back to RAM:

![unfused: 4 ops, 4 separate kernels with intermediate arrays](assets/fusion_unfused.jpg)

tinytorchcompile fuses those 4 ops into 1 kernel, so memory transfer of intermediate arrays is removed:

![tinytorchcompile fuses 4 ops into 1 reduce kernel](assets/fusion_tinytorchcompile.jpg)

`torch.compile` fuses the same chain into one kernel too:

![torch.compile fuses the same chain into one kernel](assets/fusion_torchcompile.jpg)

Note: The above times are measured on Macbook M1 Pro CPU and for a single function call. The actual runtime will be faster on a GPU or if the function is called multiple times due to caching. If you want to see the difference in runtime on a GPU, you can run the [demo notebook](demo.ipynb) on a GPU.

## Run it

```python
import numpy as np, tinytorchcompile as ttc

@ttc.compile                      # works just like torch.compile
def f(w, x, b):
    return (w * x + b).relu().sum()

w, x, b = (np.random.randn(1_000_000) for _ in range(3))
print(f(w, x, b))      # the fused, compiled result
print(f.num_kernels)   # 1, the four ops fused into one loop
print(f.csrc)          # the C code generated, compiled, and ran
```

The four ops (`mul, add, relu, sum`) become **one** loop with no intermediate arrays in RAM:

```c
static void kernel_b6(double* in0, double* in1, double* in2, double* out) {
  double acc=0.0; for(long k=0;k<1000000;k++){ acc = acc + (fmax(((in0[k] * in1[k]) + in2[k]), 0.0)); } out[0]=acc;
}
```

`torch.compile` fuses the same chain, then vectorizes it with SIMD and parallelizes it with OpenMP. `clamp_min(x, 0)` is the relu. This is the [highly optimized single C++ code](demo.ipynb#X10sZmlsZQ==) the demo prints in full:

```cpp
extern "C" void kernel(const double* in_ptr0, const double* in_ptr1,
                       const double* in_ptr2, double* out_ptr0) {
  double tmp_acc0 = 0;
  at::vec::VectorizedN<double,2> tmp_acc0_vec(0);
  #pragma omp parallel num_threads(8)
  {
    auto tmp_acc0_vec_local = at::vec::VectorizedN<double,2>(0);
    #pragma omp for
    for (int64_t x0 = 0; x0 < 8000000LL; x0 += 4LL) {
      auto tmp0 = at::vec::VectorizedN<double,2>::loadu(in_ptr0 + x0, 4);
      auto tmp1 = at::vec::VectorizedN<double,2>::loadu(in_ptr1 + x0, 4);
      auto tmp3 = at::vec::VectorizedN<double,2>::loadu(in_ptr2 + x0, 4);
      auto tmp2 = tmp0 * tmp1;
      auto tmp4 = tmp2 + tmp3;
      auto tmp5 = at::vec::clamp_min(tmp4, decltype(tmp4)(0));
      tmp_acc0_vec_local = tmp_acc0_vec_local + tmp5;
    }
    tmp_acc0_vec = tmp_acc0_vec + tmp_acc0_vec_local;
  }
  out_ptr0[0] = at::vec::vec_reduce_all<double, 2>(
      [](auto& x, auto& y) { return x + y; }, tmp_acc0_vec);
}
```

To run the demo notebook, you can use the following commands:
```bash
pip install numpy                      # torch optional, for comparison with torch.compile
python tinytorchcompile.py                 # the algorithm of operator fusion in a single file
jupyter lab demo.ipynb    # the demo notebook
```

## How tinytorchcompile works

The pipeline of tinytorchcompile is similar to TorchInductor: **trace -> lower -> fuse -> codegen -> run**.

1. **Trace.** Operator overloading on `Tensor` records a lazy graph of ops.
2. **Lower.** Each node becomes a `Buffer` whose loop body is a closure `inner(index)` calling a *virtualized* ops namespace `V.ops`. Swap the handler and the same body changes job: `Analysis` records reads, `Codegen` emits C — TorchInductor's central trick.
3. **Fuse.** Just closure inlining: the scheduler marks pointwise producers `inlined`, so a consumer recomputes them in place instead of reading a materialized array.
4. **Codegen + run.** One C kernel per materialized buffer, compiled with `clang`/`gcc`, called via `ctypes`.

Attached [demo notebook](demo.ipynb) demonstrates the above pipeline of operator fusion with a simple example and a complex example:

- Operator fusion on `(w * x + b).relu().sum()` — ~8ms in fused vs ~30ms in unfused kernels for linear+relu operation on an 8M sized input vector demonstrating a 3.75x speedup. Prints the highly optimized C++ code of this fused kernel used in torch.compile.
- ResNet layer via `torch.compile` — times in eager(unfused) vs compiled(fused), prints the highly optimized C++ code of this fused kernel generated by TorchInductor under the hood of torch.compile. TorchInductor fuses the conv-bn-relu epilogues into a handful of kernels:

![torch.compile fuses the ResNet block's conv-bn-relu into a few kernels](assets/fusion_resnet_torchcompile.jpg)

## Reference

[PyTorch 2 paper](https://dl.acm.org/doi/pdf/10.1145/3620665.3640366). Its ablation shows that other optimizations without fusion are slower than eager.

## Contributing

If you found this repo useful, consider giving it a ⭐ as it helps others discover it. Also, if you have any suggestions or feedback, please feel free to open an issue or a pull request.