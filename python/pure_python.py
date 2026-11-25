"""MiniAlexNet in pure Python: no NumPy, no libraries, just lists and loops.

This is the exact same algorithm as c/alexnet.c (direct convolution, line for
line), so the speed gap you see is the cost of the CPython interpreter itself:
every multiply-add is a bytecode dispatch, two boxed-float allocations and a
couple of list index lookups, instead of one fused CPU instruction."""

import math
import os

from common import CONVS, FCS, IMG, DATA_DIR, load_params, parse_args, run

import numpy as np  # only for loading/saving files outside the timed region


def conv_forward(x, C, H, W, w, b, cout, k, pad):
    Hp, Wp = H + 2 * pad, W + 2 * pad
    xp = [0.0] * (C * Hp * Wp)
    for c in range(C):
        for i in range(H):
            src = (c * H + i) * W
            dst = (c * Hp + i + pad) * Wp + pad
            xp[dst:dst + W] = x[src:src + W]
    y = [0.0] * (cout * H * W)
    kk = k * k
    for co in range(cout):
        bias = b[co]
        wbase = co * C * kk
        for i in range(H):
            for j in range(W):
                s = bias
                for ci in range(C):
                    for ki in range(k):
                        xrow = (ci * Hp + i + ki) * Wp + j
                        wrow = wbase + (ci * k + ki) * k
                        for kj in range(k):
                            s += w[wrow + kj] * xp[xrow + kj]
                y[(co * H + i) * W + j] = s if s > 0.0 else 0.0  # fused ReLU
    return y, xp


def conv_backward(dy, a, xp, C, H, W, w, dw, db, cout, k, pad, need_dx):
    Hp, Wp = H + 2 * pad, W + 2 * pad
    dxp = [0.0] * (C * Hp * Wp) if need_dx else None
    kk = k * k
    for co in range(cout):
        wbase = co * C * kk
        for i in range(H):
            for j in range(W):
                idx = (co * H + i) * W + j
                if a[idx] <= 0.0:          # ReLU gradient
                    continue
                g = dy[idx]
                db[co] += g
                for ci in range(C):
                    for ki in range(k):
                        xrow = (ci * Hp + i + ki) * Wp + j
                        wrow = wbase + (ci * k + ki) * k
                        for kj in range(k):
                            dw[wrow + kj] += g * xp[xrow + kj]
                            if need_dx:
                                dxp[xrow + kj] += g * w[wrow + kj]
    if not need_dx:
        return None
    dx = [0.0] * (C * H * W)
    for c in range(C):
        for i in range(H):
            src = (c * Hp + i + pad) * Wp + pad
            dst = (c * H + i) * W
            dx[dst:dst + W] = dxp[src:src + W]
    return dx


def pool_forward(x, C, H, W):
    Ho, Wo = H // 2, W // 2
    y = [0.0] * (C * Ho * Wo)
    arg = [0] * (C * Ho * Wo)
    for c in range(C):
        for i in range(Ho):
            for j in range(Wo):
                best, bi = -1e30, 0
                for di in range(2):
                    for dj in range(2):
                        idx = (c * H + 2 * i + di) * W + 2 * j + dj
                        if x[idx] > best:
                            best, bi = x[idx], idx
                o = (c * Ho + i) * Wo + j
                y[o], arg[o] = best, bi
    return y, arg


def pool_backward(dy, arg, n_in):
    dx = [0.0] * n_in
    for o in range(len(dy)):
        dx[arg[o]] += dy[o]
    return dx


def fc_forward(x, w, b, fout, relu):
    fin = len(x)
    y = [0.0] * fout
    for o in range(fout):
        s = b[o]
        base = o * fin
        for i in range(fin):
            s += w[base + i] * x[i]
        y[o] = (s if s > 0.0 else 0.0) if relu else s
    return y


def fc_backward(dy, x, a, w, dw, db, relu):
    fin = len(x)
    dx = [0.0] * fin
    for o in range(len(dy)):
        g = dy[o]
        if relu and a[o] <= 0.0:
            continue
        db[o] += g
        base = o * fin
        for i in range(fin):
            dw[base + i] += g * x[i]
            dx[i] += g * w[base + i]
    return dx


class Model:
    def __init__(self, params, lr=0.01, momentum=0.9):
        self.shapes = [p.shape for p in params]
        self.p = [p.ravel().tolist() for p in params]
        self.v = [[0.0] * len(p) for p in self.p]
        self.lr, self.mu = lr, momentum

    def prepare(self, x):
        return [img.ravel().tolist() for img in x]

    def get_params(self):
        return [np.array(p, dtype=np.float32).reshape(s) for p, s in zip(self.p, self.shapes)]

    def _forward(self, img, cache):
        h, C, H, W = img, 3, IMG, IMG
        for li, (_, cin, cout, k, pad, pool) in enumerate(CONVS):
            a, xp = conv_forward(h, C, H, W, self.p[2 * li], self.p[2 * li + 1], cout, k, pad)
            cache.append((a, xp, C, H, W))
            C = cout
            if pool:
                a, arg = pool_forward(a, C, H, W)
                cache.append((arg, C * H * W))
                H, W = H // 2, W // 2
            h = a
        off = 2 * len(CONVS)
        for fi, (_, _, fout, relu) in enumerate(FCS):
            a = fc_forward(h, self.p[off + 2 * fi], self.p[off + 2 * fi + 1], fout, relu)
            cache.append((h, a))
            h = a
        return h

    def forward(self, xs):
        return [self._forward(img, []) for img in xs]

    def train_step(self, xs, ys):
        B = len(xs)
        grads = [[0.0] * len(p) for p in self.p]
        total = 0.0
        for img, label in zip(xs, ys):
            cache = []
            logits = self._forward(img, cache)
            m = max(logits)
            e = [math.exp(z - m) for z in logits]
            s = sum(e)
            prob = [v / s for v in e]
            total += -math.log(prob[label] + 1e-12)
            d = [v / B for v in prob]
            d[label] -= 1.0 / B

            off = 2 * len(CONVS)
            for fi in reversed(range(len(FCS))):
                x_in, a = cache.pop()
                d = fc_backward(d, x_in, a, self.p[off + 2 * fi],
                                grads[off + 2 * fi], grads[off + 2 * fi + 1], FCS[fi][3])
            for li in reversed(range(len(CONVS))):
                _, cin, cout, k, pad, pool = CONVS[li]
                if pool:
                    arg, n_in = cache.pop()
                    d = pool_backward(d, arg, n_in)
                a, xp, C, H, W = cache.pop()
                d = conv_backward(d, a, xp, C, H, W, self.p[2 * li], grads[2 * li],
                                  grads[2 * li + 1], cout, k, pad, need_dx=li > 0)

        lr, mu = self.lr, self.mu
        for p, v, g in zip(self.p, self.v, grads):
            for i in range(len(p)):
                v[i] = mu * v[i] + g[i]
                p[i] -= lr * v[i]
        return total / B


if __name__ == "__main__":
    args = parse_args()
    params = load_params(os.path.join(DATA_DIR, "init_weights.bin"))
    run(Model(params, args.lr, args.momentum), args)
