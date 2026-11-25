"""Shared pieces for every Python implementation: the network spec, file I/O,
and the timed training / inference driver. The C implementation mirrors the
same spec, file formats and log formats exactly (see c/alexnet.c)."""

import argparse
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
RESULTS_DIR = os.path.join(ROOT, "results")

# MiniAlexNet: AlexNet's shape (5 conv + 3 FC, ReLU, max-pool after conv1/2/5)
# scaled to 32x32 inputs so even the pure-Python version can run.
#   kind, in_ch, out_ch, kernel, pad, pool_after
CONVS = [
    ("conv1", 3, 16, 5, 2, True),    # 32x32 -> pool -> 16x16
    ("conv2", 16, 32, 5, 2, True),   # 16x16 -> pool -> 8x8
    ("conv3", 32, 48, 3, 1, False),  # 8x8
    ("conv4", 48, 48, 3, 1, False),  # 8x8
    ("conv5", 48, 32, 3, 1, True),   # 8x8 -> pool -> 4x4
]
FCS = [
    ("fc6", 32 * 4 * 4, 128, True),
    ("fc7", 128, 128, True),
    ("fc8", 128, 10, False),
]
IMG = 32
NUM_CLASSES = 10


def param_shapes():
    """Ordered (name, shape) list. Weight files are these arrays, float32,
    concatenated in this order."""
    shapes = []
    for name, cin, cout, k, _, _ in CONVS:
        shapes.append((name + ".W", (cout, cin, k, k)))
        shapes.append((name + ".b", (cout,)))
    for name, fin, fout, _ in FCS:
        shapes.append((name + ".W", (fout, fin)))
        shapes.append((name + ".b", (fout,)))
    return shapes


def load_params(path):
    flat = np.fromfile(path, dtype=np.float32)
    out, off = [], 0
    for _, shp in param_shapes():
        n = int(np.prod(shp))
        out.append(flat[off:off + n].reshape(shp).copy())
        off += n
    assert off == flat.size, "weight file size mismatch"
    return out


def save_params(path, params):
    np.concatenate([np.asarray(p, dtype=np.float32).ravel() for p in params]).tofile(path)


def load_data():
    with open(os.path.join(DATA_DIR, "meta.json")) as f:
        meta = json.load(f)
    def x(name, n):
        return np.fromfile(os.path.join(DATA_DIR, name), dtype=np.float32).reshape(n, 3, IMG, IMG)
    def y(name):
        return np.fromfile(os.path.join(DATA_DIR, name), dtype=np.int32)
    return (x("train_x.bin", meta["n_train"]), y("train_y.bin"),
            x("test_x.bin", meta["n_test"]), y("test_y.bin"))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["train", "steps", "infer"], default="train")
    p.add_argument("--budget", type=float, default=60.0, help="seconds of training/inference time")
    p.add_argument("--steps", type=int, default=3, help="for --mode steps")
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--lr", type=float, default=0.005)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--snaps", type=int, default=20, help="weight snapshots over the budget")
    p.add_argument("--name", required=True, help="results/<name>/")
    p.add_argument("--device", default="cpu", help="torch only")
    return p.parse_args()


def run(model, args):
    """Drive a model object that provides:
         prepare(x_np) -> backend data (called once, untimed)
         train_step(xb, yb) -> float loss   (forward + backward + SGD update)
         forward(xb) -> logits              (any array-like)
         sync()                             (wait for async devices; optional)
         get_params() -> list of numpy arrays in param_shapes() order
    """
    out_dir = os.path.join(RESULTS_DIR, args.name)
    os.makedirs(out_dir, exist_ok=True)
    train_x, train_y, test_x, _ = load_data()
    sync = getattr(model, "sync", lambda: None)

    if args.mode == "steps":
        # Deterministic parity check: same first N batches for every implementation.
        losses = []
        for s in range(args.steps):
            i0 = s * args.batch
            xb = model.prepare(train_x[i0:i0 + args.batch])
            losses.append(float(model.train_step(xb, train_y[i0:i0 + args.batch])))
        with open(os.path.join(out_dir, "parity.json"), "w") as f:
            json.dump({"batch": args.batch, "losses": losses}, f)
        print(json.dumps({"name": args.name, "losses": losses}))
        return

    if args.mode == "infer":
        # Batch-1 latency, the "real-time" case. First call is a warm-up.
        xs = [model.prepare(test_x[i:i + 1]) for i in range(min(200, len(test_x)))]
        model.forward(xs[0]); sync()
        times, i = [], 0
        t_start = time.perf_counter()
        while time.perf_counter() - t_start < args.budget and i < 2000:
            t0 = time.perf_counter()
            model.forward(xs[i % len(xs)]); sync()
            times.append(time.perf_counter() - t0)
            i += 1
        res = {"name": args.name, "n": len(times),
               "median_ms": float(np.median(times) * 1e3),
               "mean_ms": float(np.mean(times) * 1e3)}
        with open(os.path.join(out_dir, "infer.json"), "w") as f:
            json.dump(res, f)
        print(json.dumps(res))
        return

    # --- timed training race ---
    # Pre-convert all training batches outside the clock.
    n = len(train_x)
    nb = n // args.batch
    batches = [(model.prepare(train_x[b * args.batch:(b + 1) * args.batch]),
                train_y[b * args.batch:(b + 1) * args.batch]) for b in range(nb)]

    snap_every = args.budget / args.snaps
    next_snap, snap_idx = 0.0, 0
    elapsed, step = 0.0, 0
    log = open(os.path.join(out_dir, "train_log.csv"), "w")
    log.write("step,elapsed,images,loss\n")
    snaps = open(os.path.join(out_dir, "snaps.csv"), "w")
    snaps.write("snap,elapsed,step,images\n")

    def snapshot():
        nonlocal snap_idx
        save_params(os.path.join(out_dir, f"snap_{snap_idx:03d}.bin"), model.get_params())
        snaps.write(f"{snap_idx},{elapsed:.6f},{step},{step * args.batch}\n"); snaps.flush()
        snap_idx += 1

    while elapsed < args.budget:
        if elapsed >= next_snap:
            snapshot()
            next_snap += snap_every
        xb, yb = batches[step % nb]
        t0 = time.perf_counter()
        loss = model.train_step(xb, yb)
        loss = float(loss)  # forces device sync for torch
        elapsed += time.perf_counter() - t0
        step += 1
        log.write(f"{step},{elapsed:.6f},{step * args.batch},{loss:.6f}\n"); log.flush()
        if step == 1 or step % 50 == 0:
            print(f"[{args.name}] step {step:6d}  t={elapsed:7.2f}s  loss={loss:.4f}", file=sys.stderr)
    snapshot()
    log.close(); snaps.close()
    print(json.dumps({"name": args.name, "steps": step, "elapsed": elapsed,
                      "images_per_sec": step * args.batch / elapsed}))
