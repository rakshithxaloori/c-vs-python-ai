"""Run every implementation through the same three tests, one at a time, then
write results/summary.json for the report.

  1. parity  - 3 identical SGD steps; all losses must agree
  2. train   - a fixed wall-clock training budget (default 60 s) each
  3. infer   - batch-1 latency (the real-time / serving case)

Usage:  .venv/bin/python run_benchmarks.py [--budget 60] [--only name,name]
"""

import argparse
import csv
import glob
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
PY = os.path.join(ROOT, ".venv", "bin", "python")
sys.path.insert(0, os.path.join(ROOT, "python"))

IMPLS = [
    # name, label, group, command
    ("python_pure", "Pure Python", "python", [PY, "python/pure_python.py"]),
    ("c_O0", "C, -O0 (unoptimised)", "c", ["c/alexnet_O0"]),
    ("c_O3", "C, -O3 (direct loops)", "c", ["c/alexnet_O3"]),
    ("python_numpy", "Python + NumPy", "python", [PY, "python/numpy_impl.py"]),
    ("python_torch_cpu", "Python + PyTorch (CPU)", "python", [PY, "python/torch_impl.py", "--device", "cpu"]),
    ("python_torch_mps", "Python + PyTorch (GPU/MPS)", "python", [PY, "python/torch_impl.py", "--device", "mps"]),
    ("c_blas", "C + BLAS (im2col+GEMM)", "c", ["c/alexnet_blas"]),
]


def sh(cmd):
    print("  $", " ".join(os.path.relpath(c, ROOT) if c.startswith(ROOT) else c for c in cmd), flush=True)
    out = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    if out.returncode != 0:
        print(out.stderr)
        raise SystemExit(f"command failed: {cmd}")
    return json.loads(out.stdout.strip().splitlines()[-1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=float, default=60)
    ap.add_argument("--infer-budget", type=float, default=5)
    ap.add_argument("--only", default="")
    a = ap.parse_args()

    subprocess.run(["make", "-s", "-C", "c"], cwd=ROOT, check=True)
    if not os.path.exists(os.path.join(ROOT, "data", "init_weights.bin")):
        subprocess.run([PY, "python/make_data.py"], cwd=ROOT, check=True)

    impls = [i for i in IMPLS if not a.only or i[0] in a.only.split(",")]
    try:
        import torch
        if not torch.backends.mps.is_available():
            impls = [i for i in impls if i[0] != "python_torch_mps"]
    except ImportError:
        impls = [i for i in impls if "torch" not in i[0]]

    for name, label, _, cmd in impls:
        print(f"\n=== {label} ===", flush=True)
        sh(cmd + ["--mode", "steps", "--steps", "3", "--batch", "4", "--name", name])
        sh(cmd + ["--mode", "train", "--budget", str(a.budget), "--name", name])
        sh(cmd + ["--mode", "infer", "--budget", str(a.infer_budget), "--name", name])

    build_summary()


def build_summary():
    from common import load_data, load_params, RESULTS_DIR
    from numpy_impl import accuracy

    _, _, test_x, test_y = load_data()
    summary = {"impls": [], "machine": machine_info()}
    for name, label, group, _ in IMPLS:
        d = os.path.join(RESULTS_DIR, name)
        if not os.path.exists(os.path.join(d, "infer.json")):
            continue
        with open(os.path.join(d, "train_log.csv")) as f:
            log = [(int(r["step"]), float(r["elapsed"]), float(r["loss"])) for r in csv.DictReader(f)]
        with open(os.path.join(d, "snaps.csv")) as f:
            snaps = list(csv.DictReader(f))
        curve = []
        for s in snaps:
            acc = accuracy(load_params(os.path.join(d, f"snap_{int(s['snap']):03d}.bin")), test_x, test_y)
            curve.append({"t": float(s["elapsed"]), "images": int(s["images"]), "acc": acc})
        with open(os.path.join(d, "parity.json")) as f:
            parity = json.load(f)
        with open(os.path.join(d, "infer.json")) as f:
            infer = json.load(f)
        # thin the loss log to <= 400 points for the chart
        stride = max(1, len(log) // 400)
        steps, elapsed = log[-1][0], log[-1][1]
        summary["impls"].append({
            "name": name, "label": label, "group": group,
            "steps": steps, "elapsed": elapsed, "images": steps * 16,
            "images_per_sec": steps * 16 / elapsed,
            "final_acc": curve[-1]["acc"], "acc_curve": curve,
            "loss_curve": [{"t": t, "step": s, "loss": l} for s, t, l in log[::stride]],
            "parity_losses": parity["losses"], "infer_ms": infer["median_ms"], "infer_n": infer["n"],
        })
        print(f"{label:30s} {steps * 16 / elapsed:10.1f} img/s   acc {curve[-1]['acc']:.3f}   "
              f"infer {infer['median_ms']:9.3f} ms")
    with open(os.path.join(RESULTS_DIR, "summary.json"), "w") as f:
        json.dump(summary, f)
    print("\nwrote results/summary.json")


def machine_info():
    def q(k):
        try:
            return subprocess.run(["sysctl", "-n", k], capture_output=True, text=True).stdout.strip()
        except Exception:
            return ""
    import platform
    import numpy
    info = {"cpu": q("machdep.cpu.brand_string") or platform.processor(), "cores": q("hw.ncpu"),
            "python": platform.python_version(), "numpy": numpy.__version__}
    try:
        import torch
        info["torch"] = torch.__version__
    except ImportError:
        pass
    return info


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--summary-only":
        build_summary()
    else:
        main()
