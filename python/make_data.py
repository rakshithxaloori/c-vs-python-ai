"""Generate a synthetic 10-class 32x32 RGB "shapes" dataset and one shared set
of initial weights. Every implementation (Python, NumPy, C, PyTorch) starts
from these exact bytes, so their loss curves are directly comparable.

No downloads: everything is procedurally generated with a fixed seed."""

import json
import os

import numpy as np

from common import DATA_DIR, IMG, param_shapes, save_params

CLASSES = ["circle", "square", "triangle", "cross", "ring",
           "h-stripes", "v-stripes", "diagonal", "checker", "dots"]


def draw(cls, rng):
    yy, xx = np.mgrid[0:IMG, 0:IMG].astype(np.float32)
    cx, cy = rng.uniform(11, 21, size=2)
    r = rng.uniform(6, 10)
    dx, dy = xx - cx, yy - cy
    if cls == 0:   m = dx**2 + dy**2 < r**2
    elif cls == 1: m = (np.abs(dx) < r * .8) & (np.abs(dy) < r * .8)
    elif cls == 2: m = (dy < r * .7) & (dy > -r) & (np.abs(dx) < (dy + r) * .6)
    elif cls == 3: m = ((np.abs(dx) < r * .3) | (np.abs(dy) < r * .3)) & (np.abs(dx) < r) & (np.abs(dy) < r)
    elif cls == 4: d = np.sqrt(dx**2 + dy**2); m = (d < r) & (d > r * .55)
    elif cls == 5: m = (np.floor(yy / rng.uniform(3, 5)) % 2 == 0)
    elif cls == 6: m = (np.floor(xx / rng.uniform(3, 5)) % 2 == 0)
    elif cls == 7: m = (np.floor((xx + yy * rng.choice([-1, 1])) / rng.uniform(3, 5)) % 2 == 0)
    elif cls == 8: s = rng.uniform(4, 7); m = ((np.floor(xx / s) + np.floor(yy / s)) % 2 == 0)
    else:
        m = np.zeros((IMG, IMG), bool)
        for _ in range(rng.integers(4, 8)):
            px, py = rng.uniform(4, 28, size=2)
            m |= (xx - px)**2 + (yy - py)**2 < rng.uniform(2, 5)
    fg, bg = rng.uniform(0, 1, size=3), rng.uniform(0, 1, size=3)
    while np.abs(fg - bg).sum() < 0.6:          # keep the shape visible
        fg, bg = rng.uniform(0, 1, size=3), rng.uniform(0, 1, size=3)
    img = np.where(m[None], fg[:, None, None], bg[:, None, None])
    img += rng.normal(0, 0.08, size=img.shape)  # sensor noise
    return img.astype(np.float32)


def make(n, rng):
    y = rng.integers(0, 10, size=n).astype(np.int32)
    x = np.stack([draw(c, rng) for c in y])
    return (x - 0.5) / 0.3, y                    # roughly zero-mean, unit-ish var


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    rng = np.random.default_rng(1234)
    tx, ty = make(4096, rng)
    vx, vy = make(1000, rng)
    tx.astype(np.float32).tofile(os.path.join(DATA_DIR, "train_x.bin"))
    ty.tofile(os.path.join(DATA_DIR, "train_y.bin"))
    vx.astype(np.float32).tofile(os.path.join(DATA_DIR, "test_x.bin"))
    vy.tofile(os.path.join(DATA_DIR, "test_y.bin"))

    # He-normal init, zero biases.
    params = []
    for name, shp in param_shapes():
        if name.endswith(".b"):
            params.append(np.zeros(shp, np.float32))
        else:
            fan_in = int(np.prod(shp[1:]))
            params.append(rng.normal(0, np.sqrt(2 / fan_in), size=shp).astype(np.float32))
    save_params(os.path.join(DATA_DIR, "init_weights.bin"), params)

    # A few sample images for the report (as uint8 RGB).
    samples = []
    for c in range(10):
        idx = np.where(ty == c)[0][:4]
        for i in idx:
            img = np.clip(tx[i] * 0.3 + 0.5, 0, 1)
            samples.append({"cls": c, "px": (img.transpose(1, 2, 0) * 255).astype(np.uint8).ravel().tolist()})

    with open(os.path.join(DATA_DIR, "meta.json"), "w") as f:
        json.dump({"n_train": len(ty), "n_test": len(vy), "classes": CLASSES,
                   "n_params": int(sum(p.size for p in params))}, f)
    with open(os.path.join(DATA_DIR, "samples.json"), "w") as f:
        json.dump({"classes": CLASSES, "samples": samples}, f)
    print(f"train {len(ty)}  test {len(vy)}  params {sum(p.size for p in params):,}")


if __name__ == "__main__":
    main()
