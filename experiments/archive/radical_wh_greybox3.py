import sys
import time

import numpy as np
import torch
from torch import nn

from vguitar import metrics as M
from vguitar.data import Dataset
from vguitar.models.archive.circe3 import _segments
from vguitar.models.archive.losses import esr_loss, preemph_esr_loss
from vguitar.models.base import pick_device

SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 0
NB = int(sys.argv[2]) if len(sys.argv) > 2 else 8
torch.manual_seed(SEED); np.random.seed(SEED)
dev = pick_device()
sw = Dataset.load("data/crossover_classb_edge_sweep.npz"); sr = sw.sr
x = np.ascontiguousarray(sw.x, np.float32); y = np.ascontiguousarray(sw.y, np.float32)
g = np.ascontiguousarray(sw.controls[:,0], np.float32)
xin = (x*g).astype(np.float32)
SEQ, HOP = 4096, 2048
starts = np.arange(0, len(xin)-SEQ, HOP)
idx = starts[:,None] + np.arange(SEQ)[None,:]
X = torch.from_numpy(xin[idx]).to(dev); Y = torch.from_numpy(y[idx]).to(dev)


def causal_fir_multi(x, W):
    B, T = x.shape; C, K = W.shape
    xp = torch.nn.functional.pad(x, (K - 1, 0)).unsqueeze(1)
    return torch.nn.functional.conv1d(xp, W.flip(1).unsqueeze(1))


def causal_fir_grouped(x, W):
    B, C, T = x.shape; K = W.shape[1]
    xp = torch.nn.functional.pad(x, (K - 1, 0))
    return torch.nn.functional.conv1d(xp, W.flip(1).unsqueeze(1), groups=C)


class PWLBank(nn.Module):
    def __init__(self, n, knots=49, rng=2.2):
        super().__init__()
        self.knots = knots; self.n = n
        xs = torch.linspace(-rng, rng, knots)
        self.register_buffer("xs", xs); self.dx = float(xs[1] - xs[0])
        base = xs.view(1, -1).repeat(n, 1)
        self.ys = nn.Parameter(base + 0.01 * torch.randn(n, knots))

    def forward(self, x):
        pos = ((x - self.xs[0]) / self.dx).clamp(0, self.knots - 1 - 1e-4)
        i0 = pos.floor().long(); frac = pos - i0.float()
        ys = self.ys; B, N, T = x.shape
        ie = i0.permute(1, 0, 2).reshape(N, -1)
        y0 = torch.gather(ys, 1, ie).reshape(N, B, T).permute(1, 0, 2)
        y1 = torch.gather(ys, 1, (ie + 1).clamp(max=self.knots - 1)).reshape(N, B, T).permute(1, 0, 2)
        yin = y0 + frac * (y1 - y0)
        lo_s = (ys[:, 1] - ys[:, 0]).view(1, N, 1) / self.dx
        hi_s = (ys[:, -1] - ys[:, -2]).view(1, N, 1) / self.dx
        below = x < self.xs[0]; above = x > self.xs[-1]
        yin = torch.where(below, ys[:, 0].view(1, N, 1) + (x - self.xs[0]) * lo_s, yin)
        yin = torch.where(above, ys[:, -1].view(1, N, 1) + (x - self.xs[-1]) * hi_s, yin)
        return yin


class RadicalModel(nn.Module):
    def __init__(self, n=8, pre_k=64, post_k=64, knots=49, rng=2.2):
        super().__init__()
        self.n = n
        wpre = torch.zeros(n, pre_k); wpre[:, 0] = 1.0
        self.w_pre = nn.Parameter(wpre + 0.01 * torch.randn(n, pre_k))
        self.shaper = PWLBank(n, knots=knots, rng=rng)
        wpost = torch.zeros(n, post_k); wpost[:, 0] = 1.0 / n
        self.w_post = nn.Parameter(wpost + 0.01 * torch.randn(n, post_k))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        u = causal_fir_multi(x, self.w_pre)
        v = self.shaper(u)
        z = causal_fir_grouped(v, self.w_post)
        return z.sum(1) + self.bias


m = RadicalModel(n=NB).to(dev)
opt = torch.optim.Adam(m.parameters(), 3e-3)
sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 80, eta_min=3e-5)
ema = {k: v.detach().clone() for k, v in m.state_dict().items()}
t0=time.time()
for ep in range(80):
    perm = torch.randperm(X.shape[0])
    for i in range(0, len(perm), 12):
        sel = perm[i:i+12]; pred = m(X[sel])[:,2048:]; tgt = Y[sel][:,2048:]
        loss = esr_loss(pred, tgt) + preemph_esr_loss(pred, tgt)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
    sch.step()
    if ep >= 60:  # EMA over last fifth to settle cosine jitter
        with torch.no_grad():
            for k, v in m.state_dict().items():
                ema[k].mul_(0.8).add_(v.detach(), alpha=0.2)

ts = Dataset.load("data/crossover_classb_edge_test.npz"); s,e = _segments(ts.controls)[0]
xt = torch.from_numpy(ts.x[s:e].astype(np.float32)).to(dev).view(1,-1)
with torch.no_grad():
    pe = m(xt).view(-1).cpu().numpy()
held_final = float(M.esr(ts.y[s:e][2048:], pe[2048:]))
m.load_state_dict(ema)
with torch.no_grad():
    pe2 = m(xt).view(-1).cpu().numpy()
held_ema = float(M.esr(ts.y[s:e][2048:], pe2[2048:]))
nparam = sum(p.numel() for p in m.parameters())
print(f"SEED {SEED} NB {NB} held_final {round(held_final,4)} held_ema {round(held_ema,4)} params {nparam} train_s {round(time.time()-t0)}", flush=True)
