"""MiniAlexNet in Python + NumPy.

Python is only the orchestrator here: every heavy op is one call into NumPy,
whose matmuls run in compiled C/assembly (Apple Accelerate on this Mac).
Convolutions use im2col so they become a single big matrix multiply.

Also used as the shared evaluator (accuracy of saved snapshots)."""

import numpy as np

from common import CONVS, FCS, load_params, DATA_DIR, parse_args, run
import os


def im2col(x, k, pad):
    B, C, H, W = x.shape
    xp = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)))
    cols = np.empty((B, C, k, k, H, W), dtype=x.dtype)
    for i in range(k):
        for j in range(k):
            cols[:, :, i, j] = xp[:, :, i:i + H, j:j + W]
    return cols.reshape(B, C * k * k, H * W)


def col2im(dcols, shape, k, pad):
    B, C, H, W = shape
    dcols = dcols.reshape(B, C, k, k, H, W)
    dxp = np.zeros((B, C, H + 2 * pad, W + 2 * pad), dtype=dcols.dtype)
    for i in range(k):
        for j in range(k):
            dxp[:, :, i:i + H, j:j + W] += dcols[:, :, i, j]
    return dxp[:, :, pad:pad + H, pad:pad + W]


class Model:
    def __init__(self, params, lr=0.01, momentum=0.9):
        self.p = [p.astype(np.float32) for p in params]
        self.v = [np.zeros_like(p) for p in self.p]
        self.lr, self.mu = lr, momentum

    def prepare(self, x):
        return np.ascontiguousarray(x, dtype=np.float32)

    def get_params(self):
        return self.p

    def forward(self, x, cache=None):
        h = x
        for li, (_, cin, cout, k, pad, pool) in enumerate(CONVS):
            W, b = self.p[2 * li], self.p[2 * li + 1]
            B, _, H, Wd = h.shape
            cols = im2col(h, k, pad)                                  # [B, cin*k*k, H*W]
            y = np.matmul(W.reshape(cout, -1), cols) + b[None, :, None]
            y = y.reshape(B, cout, H, Wd)
            a = np.maximum(y, 0)
            if cache is not None:
                cache.append(("conv", h.shape, cols, a))
            if pool:
                r = a.reshape(B, cout, H // 2, 2, Wd // 2, 2)
                m = r.max(axis=(3, 5))
                if cache is not None:
                    cache.append(("pool", r, m))
                a = m
            h = a
        h = h.reshape(h.shape[0], -1)
        off = 2 * len(CONVS)
        for fi, (_, _, _, relu) in enumerate(FCS):
            W, b = self.p[off + 2 * fi], self.p[off + 2 * fi + 1]
            y = h @ W.T + b
            a = np.maximum(y, 0) if relu else y
            if cache is not None:
                cache.append(("fc", h, a, relu))
            h = a
        return h

    def train_step(self, x, y):
        cache = []
        logits = self.forward(x, cache)
        B = x.shape[0]
        z = logits - logits.max(axis=1, keepdims=True)
        e = np.exp(z)
        prob = e / e.sum(axis=1, keepdims=True)
        loss = -np.mean(np.log(prob[np.arange(B), y] + 1e-12))

        grads = [None] * len(self.p)
        d = prob.copy()
        d[np.arange(B), y] -= 1
        d /= B
        off = 2 * len(CONVS)
        fi = len(FCS) - 1
        li = len(CONVS) - 1
        for entry in reversed(cache):
            if entry[0] == "fc":
                _, h, a, relu = entry
                if relu:
                    d = d * (a > 0)
                W = self.p[off + 2 * fi]
                grads[off + 2 * fi] = d.T @ h
                grads[off + 2 * fi + 1] = d.sum(axis=0)
                d = d @ W
                if fi == 0:
                    d = d.reshape(B, CONVS[-1][2], 4, 4)
                fi -= 1
            elif entry[0] == "pool":
                _, r, m = entry
                mask = (r == m[:, :, :, None, :, None])
                d = (mask * d[:, :, :, None, :, None]).reshape(
                    r.shape[0], r.shape[1], r.shape[2] * 2, r.shape[4] * 2)
            else:
                _, xshape, cols, a = entry
                _, cin, cout, k, pad, _ = CONVS[li]
                W = self.p[2 * li]
                d = (d * (a > 0)).reshape(B, cout, -1)                # [B, cout, H*W]
                grads[2 * li] = np.einsum("bok,bck->oc", d, cols).reshape(W.shape)
                grads[2 * li + 1] = d.sum(axis=(0, 2))
                if li > 0:
                    dcols = np.matmul(W.reshape(cout, -1).T, d)       # [B, cin*k*k, H*W]
                    d = col2im(dcols, xshape, k, pad)
                li -= 1

        for i in range(len(self.p)):
            self.v[i] = self.mu * self.v[i] + grads[i]
            self.p[i] -= self.lr * self.v[i]
        return loss


def accuracy(params, x, y, bs=250):
    m = Model(params)
    correct = 0
    for i in range(0, len(x), bs):
        correct += int((m.forward(x[i:i + bs]).argmax(axis=1) == y[i:i + bs]).sum())
    return correct / len(x)


if __name__ == "__main__":
    args = parse_args()
    params = load_params(os.path.join(DATA_DIR, "init_weights.bin"))
    run(Model(params, args.lr, args.momentum), args)
