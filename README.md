# C vs Python for neural nets: MiniAlexNet, seven ways

One AlexNet-shaped CNN (5 conv + 3 FC layers, ReLU, max-pool after conv1/2/5,
146k parameters, 32×32 RGB input) implemented seven ways. Every implementation
starts from the **same initial weights** and sees the **same images in the same
order**, and they produce the same losses to ~6 significant digits. Any speed
difference comes from how the math is executed, not what is computed.

| name | what it is | file |
|---|---|---|
| `python_pure` | CPython, lists + nested loops, no libraries | `python/pure_python.py` |
| `c_O0` | direct-loop C, `-O0` | `c/alexnet.c` |
| `c_O3` | the same C, `-O3 -mcpu=native` | `c/alexnet.c` |
| `python_numpy` | Python + NumPy, im2col + matmul (Accelerate BLAS) | `python/numpy_impl.py` |
| `python_torch_cpu` | PyTorch eager, CPU | `python/torch_impl.py` |
| `python_torch_mps` | PyTorch eager, Apple GPU (Metal) | `python/torch_impl.py` |
| `c_blas` | C, im2col + `cblas_sgemm` (Accelerate) | `c/alexnet.c -DUSE_BLAS` |

## Run

```bash
python3.14 -m venv .venv && .venv/bin/pip install numpy torch
.venv/bin/python python/make_data.py        # synthetic 10-class shapes dataset + shared init weights
.venv/bin/python run_benchmarks.py --budget 60
.venv/bin/python report/build_report.py     # -> report/index.html
```

`run_benchmarks.py` builds the C binaries (`make -C c`) and runs, for each
implementation, one after another:

1. **parity**: 3 SGD steps at batch 4, losses written to `results/<name>/parity.json`
2. **train**: a fixed wall-clock budget at batch 16, SGD with momentum 0.9 and lr 0.005;
   20 weight snapshots are later scored on the test set by one shared evaluator
3. **infer**: batch-1 forward latency (median)

Any single implementation can be run directly, e.g.
`./c/alexnet_blas --mode train --budget 10 --name try` (from the project root) or
`.venv/bin/python python/numpy_impl.py --mode steps --name try`.
