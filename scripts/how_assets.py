"""Assets for the How it works page (kolam_studio/public/how/), from the served model:
  decode.json    the model reading one example photo, step by step: each digit it wrote, its raw scores
                 (logits) and chances (softmax) for all 11 tokens, the ones the kolam rules allowed, and
                 where it looked on the photo (the last decoder layer's cross-attention, averaged over its
                 heads, as 0-255 per image token)
  training.json  its training record: loss along the way and the held-out exact readings
usage: PYTHONPATH=. python scripts/how_assets.py runs/night_AM2/best_head.pt runs/night_AM2/log.jsonl [kolam_1.jpg]"""
import base64
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from kolam_feed.encoding import VOCAB, allowed_next
from kolam_feed.model import TOK, BOS
from kolam_feed.train import load, prepare

OUT = Path("kolam_studio/public/how")
EXAMPLE = sys.argv[3] if len(sys.argv) > 3 else "kolam_1.jpg"


def attend(mha, q, kv, mask=None):
    """train._attend, also returning the attention weights averaged over the heads."""
    import torch.nn.functional as F
    d, h = q.shape[-1], mha.num_heads
    W, b = mha.in_proj_weight, mha.in_proj_bias
    q, k, v = (F.linear(t, W[j * d:(j + 1) * d], b[j * d:(j + 1) * d]).reshape(1, -1, h, d // h).transpose(1, 2)
               for j, t in enumerate((q, kv, kv)))
    att = q @ k.transpose(-1, -2) / math.sqrt(d // h)
    if mask is not None:
        att = att + mask
    w = att.softmax(-1)
    return mha.out_proj((w @ v).transpose(1, 2).reshape(1, -1, d)), w.mean(1)[0]


@torch.no_grad()
def decode(ckpt: str) -> dict:
    m, state = load(ckpt)
    S = state["args"]["size"]
    img = prepare(Image.open(f"kolam_studio/public/examples/{EXAMPLE}"), S)
    x = torch.from_numpy(np.asarray(img, np.float32) / 255).permute(2, 0, 1)[None]
    f = m.enc(x)[1]
    gh, gw = f.shape[2:]
    mem, _ = m.encode(x)
    ids, rows, cols, steps = [BOS], [0], [0], []
    seq = []
    while True:
        t = torch.tensor([ids])
        h = m.tok(t) + m.row(torch.tensor([rows]).clamp(max=m.row.num_embeddings - 1)) + \
            m.col(torch.tensor([cols]).clamp(max=m.col.num_embeddings - 1))
        n = len(ids)
        i = torch.arange(n)
        mask = torch.where(i[None, :] > i[:, None], float("-inf"), 0.0)
        for layer in m.dec.layers:
            a = layer.norm1(h)
            h = h + attend(layer.self_attn, a, a, mask)[0]
            o, cross = attend(layer.multihead_attn, layer.norm2(h), mem)
            h = h + o
            h = h + layer.linear2(torch.relu(layer.linear1(layer.norm3(h))))
        logits = m.out(m.norm(h))[0, -1]
        p = logits.softmax(-1)
        ok = [TOK[v] for v in allowed_next([VOCAB[s] for s in seq])] or [TOK["<end>"]]
        best = max(ok, key=lambda j: logits[j].item())
        look = cross[-1]                                     # the newest position, last layer
        look = (255 * look / look.max()).round().byte().numpy()
        steps.append({"tok": best, "p": [round(float(v), 4) for v in p], "l": [round(float(v), 2) for v in logits], "ok": ok,
                      "look": base64.b64encode(look.tobytes()).decode()})
        seq.append(best)
        if best == TOK["<end>"] or len(seq) > 1000:
            break
        bars = sum(1 for s in seq if s == TOK["|"])
        last = max([k + 1 for k, s in enumerate(seq) if s == TOK["|"]], default=0)
        ids.append(best)
        rows.append(bars)
        cols.append(len(seq) - last)
    enc, r = [], ""
    for s in seq:
        if s == TOK["<end>"]:
            break
        if s == TOK["|"]:
            enc.append(r)
            r = ""
        else:
            r += VOCAB[s]
    truth = next(e for e in json.loads(Path("kolam_studio/public/examples/examples.json").read_text()) if e["file"] == EXAMPLE)
    return {"photo": f"./examples/{EXAMPLE}", "size": S, "grid": [int(gh), int(gw)], "d": int(mem.shape[-1]),
            "layers": len(m.dec.layers), "heads": m.dec.layers[0].self_attn.num_heads,
            "vocab": VOCAB, "encoding": enc, "exact": enc == truth["encoding"], "steps": steps}


def training(log: str, batch: int) -> dict:
    loss, val = [], []
    for line in open(log):
        r = json.loads(line)
        if "val" in r:
            val.append([r["step"], round(r["val"]["exact"], 3)])
        elif "loss" in r:
            loss.append([int(r["step"].split("/")[0]), r["loss"]])
    k = max(1, len(loss) // 200)                             # ~200 points, each the mean of its stretch
    loss = [[loss[i + k - 1][0], round(sum(v for _, v in loss[i:i + k]) / k, 4)] for i in range(0, len(loss) - k + 1, k)]
    return {"loss": loss, "val": val, "images_per_step": batch}


if __name__ == "__main__":
    ckpt, log = sys.argv[1:3]
    d = decode(ckpt)
    (OUT / "decode.json").write_text(json.dumps(d, separators=(",", ":")))
    print("decode.json:", len(d["steps"]), "steps, exact" if d["exact"] else "steps, NOT exact", d["grid"])
    batch = torch.load(ckpt, map_location="cpu", weights_only=False)["args"]["batch"]
    (OUT / "training.json").write_text(json.dumps(training(log, batch), separators=(",", ":")))
    print("training.json written, batch", batch)
