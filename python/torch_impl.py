"""MiniAlexNet in PyTorch: what people actually use. Python on top, C++/ATen
kernels (CPU) or Metal shaders (--device mps, Apple GPU) underneath."""

import os

import numpy as np
import torch
import torch.nn.functional as F

from common import CONVS, FCS, DATA_DIR, load_params, parse_args, run


class Model:
    def __init__(self, params, lr=0.01, momentum=0.9, device="cpu"):
        self.dev = torch.device(device)
        self.p = [torch.tensor(p, device=self.dev, requires_grad=True) for p in params]
        self.opt = torch.optim.SGD(self.p, lr=lr, momentum=momentum)

    def prepare(self, x):
        return torch.tensor(np.ascontiguousarray(x), device=self.dev)

    def get_params(self):
        return [p.detach().cpu().numpy() for p in self.p]

    def sync(self):
        if self.dev.type == "mps":
            torch.mps.synchronize()

    def _forward(self, x):
        h = x
        for li, (_, _, _, _, pad, pool) in enumerate(CONVS):
            h = F.relu(F.conv2d(h, self.p[2 * li], self.p[2 * li + 1], padding=pad))
            if pool:
                h = F.max_pool2d(h, 2)
        h = h.flatten(1)
        off = 2 * len(CONVS)
        for fi, (_, _, _, relu) in enumerate(FCS):
            h = F.linear(h, self.p[off + 2 * fi], self.p[off + 2 * fi + 1])
            if relu:
                h = F.relu(h)
        return h

    def forward(self, x):
        with torch.no_grad():
            return self._forward(x)

    def train_step(self, x, y):
        if not torch.is_tensor(y):
            y = torch.tensor(np.asarray(y, dtype=np.int64), device=self.dev)
        self.opt.zero_grad(set_to_none=True)
        loss = F.cross_entropy(self._forward(x), y)
        loss.backward()
        self.opt.step()
        return loss.item()


if __name__ == "__main__":
    args = parse_args()
    torch.manual_seed(0)
    params = load_params(os.path.join(DATA_DIR, "init_weights.bin"))
    run(Model(params, args.lr, args.momentum, args.device), args)
