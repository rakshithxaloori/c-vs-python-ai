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

## Results

Apple M1 (8 cores), Python 3.14.10, NumPy 2.5.5, PyTorch 2.15.0. Each
implementation trained for 60 seconds of wall-clock time (batch 16); test
accuracy is scored by the shared evaluator on the last snapshot. Chance is 10%.

| implementation | train images/s | vs pure Python | images seen in 60 s | test accuracy | batch-1 inference |
|---|---:|---:|---:|---:|---:|
| Pure Python | 0.97 | 1× | 64 | 11.1% | 617 ms |
| C, `-O0` | 29.6 | 31× | 1,776 | 23.2% | 26.9 ms |
| C, `-O3` (direct loops) | 140 | 145× | 8,400 | 74.8% | 6.01 ms |
| Python + NumPy | 515 | 533× | 30,928 | 94.4% | 0.48 ms |
| Python + PyTorch (GPU/MPS) | 1,227 | 1,269× | 73,616 | 98.8% | 2.18 ms |
| Python + PyTorch (CPU) | 1,455 | 1,505× | 87,328 | 98.8% | 0.31 ms |
| **C + BLAS (im2col + GEMM)** | **2,979** | **3,082×** | **178,768** | 98.5% | **0.16 ms** |

**Parity.** All seven produce the same first three SGD losses
(1.828216, 4.103791, 2.615286) to within ±2 in the sixth decimal place, so the
gaps above are purely execution speed.

**What it shows.**

- The language matters less than the algorithm. Naive C at `-O3` is about 145×
  faster than pure Python, but Python calling NumPy's BLAS beats that C by
  3.7×, and PyTorch on CPU beats it by about 10×.
- The fastest result is C doing what NumPy and PyTorch do (im2col and a tuned
  GEMM) without the interpreter or framework overhead: about 2× PyTorch CPU
  for training and 1.9× for batch-1 inference.
- Throughput turns into accuracy under a fixed time budget. The slow
  implementations barely move from chance in 60 seconds, while everything
  BLAS-backed reaches 94–99%.
- On a network this small, the Apple GPU (MPS) is slower than the CPU: kernel
  launch and sync overhead outweighs the parallelism, especially at batch 1.

Raw numbers are in `results/summary.json`; per-run logs, curves and snapshots
are in `results/<name>/`. `report/index.html` is an interactive report with the
accuracy-over-time and loss curves.
