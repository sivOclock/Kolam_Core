"""Where does training time go? Run on the GPU box (scripts/gpu/probe.sh), ~5 minutes.

    gpu   train steps on one batch drawn once and reused: how many images/s the GPU can train when
          pictures cost nothing (seq up to --max-dots, and grid B-M for comparison), and its memory
    draw  how many images/s the CPU workers can draw (same `real` look as training), for each
          worker count in --workers

Training runs at the smaller of the two: if draw < gpu, faster drawing makes training faster.
"""
from __future__ import annotations

import argparse
import time

import torch
from torch.utils.data import DataLoader, Subset

from kolam_feed.data import KolamDataset
from kolam_feed.model import build
from kolam_feed.train import batch_items, grid_size, to_device, to_item


def pick(ds, max_dots, n, seed=0):
    g = torch.Generator().manual_seed(seed)
    ok = [i for i in range(len(ds)) if max(grid_size(ds.encodings[i].split("|"))) <= max_dots]
    return [ok[j] for j in torch.randperm(len(ok), generator=g)[:n].tolist()]


def gpu(ds, kind, max_dots, batch, steps, extra, dev="cuda"):
    dev = torch.device(dev)
    b = to_device(batch_items([ds[i] for i in pick(ds, max_dots, batch)]), dev)
    b["neg"] = torch.rand(round(batch * 0.15), 3, ds.size, ds.size, device=dev)    # the not-kolams in each batch
    model = build({"kind": kind, "max_rows": max_dots, "max_cols": max_dots, "d": 256, **extra}, pretrained=False).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scaler = torch.amp.GradScaler()
    torch.cuda.reset_peak_memory_stats()
    for i in range(steps + 5):
        if i == 5:                                                    # 5 warm-up steps
            torch.cuda.synchronize()
            t = time.time()
        with torch.autocast(dev.type):
            loss = model.loss(b)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
    torch.cuda.synchronize()
    rate = steps * batch / (time.time() - t)
    print(f"gpu  {kind:4s} up to {max_dots}x{max_dots}: {rate:5.1f} images/s, "
          f"peak {torch.cuda.max_memory_allocated() / 2**30:.2f} GB", flush=True)
    del model, opt


def draw(ds, max_dots, batch, workers, seconds):
    idx = pick(ds, max_dots, 100_000, seed=1)
    dl = DataLoader(Subset(ds, idx), batch_size=batch, num_workers=workers, collate_fn=batch_items, prefetch_factor=2)
    n, t = 0, None
    for k, b in enumerate(dl):
        if k == 2 * workers:                                          # every worker has started
            t, n = time.time(), 0
        elif t:
            n += len(b["encoding"])
            if time.time() - t > seconds:
                break
    print(f"draw {workers:2d} workers, up to {max_dots}x{max_dots}: {n / (time.time() - t):5.1f} images/s", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("shard", help="one kolam_gen .jsonl shard (sizes 3 to 20)")
    p.add_argument("--max-dots", type=int, default=12)
    p.add_argument("--batch", type=int, default=24)
    p.add_argument("--steps", type=int, default=40)
    p.add_argument("--workers", default="4,8,10,14,16")
    p.add_argument("--seconds", type=float, default=30)
    a = p.parse_args()
    ds = KolamDataset(a.shard, size=384, augment="real", transform=to_item)
    gpu(ds, "seq", a.max_dots, a.batch, a.steps, {"backbone": "resnet34"})
    gpu(ds, "grid", 20, a.batch, a.steps, {"backbone": "resnet34", "ctx_layers": 4})
    for w in map(int, a.workers.split(",")):
        draw(ds, a.max_dots, a.batch, w, a.seconds)
