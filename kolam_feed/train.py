"""Train and evaluate the image -> encoding models (kolam_feed/model.py) on kolam_gen shards.

    python -m kolam_feed train data/enc_1m --model seq  --out runs/seq  --max-dots 5
    python -m kolam_feed train data/enc_1m --model grid --out runs/grid --resume   # continue from runs/grid/last.pt
    python -m kolam_feed eval  runs/seq/last.pt data/enc_1m --n 2000 --aug photo

Images are rendered on the fly (KolamDataset), a fresh look every epoch. Every 100th kolam is held out
for validation. Checkpoints: <out>/last.pt every --save-every steps (resume from it), and a kept
<out>/step_<n>.pt at every evaluation. Scores: exact (the whole encoding right, i.e. the exact label), valid (the answer reads
back into a kolam), digits (gap accuracy where the predicted grid has the right size).
"""
from __future__ import annotations

import json
import math
import os
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from .data import KolamDataset
from .encoding import KolamError, encoding_grid, from_encoding
from .model import PAD, TOK, build, gap_points

ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("TORCH_HOME", str(ROOT / ".cache" / "torch"))     # pretrained weights stay in the repo


def grid_size(enc: list[str]) -> tuple[int, int]:
    return (len(enc) - 1) // 2, len(enc[0])


def to_item(it: dict) -> dict:
    """Dataset transform (runs in the workers): image -> float tensor, the label in model form."""
    a = np.asarray(it["image"].convert("RGB"), np.float32) / 255
    enc = it["encoding"]
    rows, cols = grid_size(enc)
    g = encoding_grid(enc)
    pts = gap_points(rows, cols)
    y = g[pts[:, 1].astype(int), pts[:, 0].astype(int)]
    return {"image": torch.from_numpy(a).permute(2, 0, 1), "encoding": enc, "rows": rows, "cols": cols,
            "tokens": torch.tensor([TOK[t] for t in it["tokens"]]), "M": torch.tensor(it["M"], dtype=torch.float32),
            "gap_pts": torch.from_numpy(pts), "gap_y": torch.from_numpy(y.astype(np.int64))}


def batch_items(items: list[dict]) -> dict:
    L = max(len(it["tokens"]) for it in items)
    tok = torch.full((len(items), L), PAD, dtype=torch.long)
    for i, it in enumerate(items):
        tok[i, :len(it["tokens"])] = it["tokens"]
    return {"image": torch.stack([it["image"] for it in items]), "tokens": tok,
            "rows": torch.tensor([it["rows"] for it in items]), "cols": torch.tensor([it["cols"] for it in items]),
            "M": torch.stack([it["M"] for it in items]),
            "gap_pts": torch.cat([it["gap_pts"] for it in items]),
            "gap_y": torch.cat([it["gap_y"] for it in items]),
            "gap_b": torch.cat([torch.full((len(it["gap_y"]),), i) for i, it in enumerate(items)]),
            "encoding": [it["encoding"] for it in items]}


def to_device(b: dict, dev) -> dict:
    return {k: v.to(dev, non_blocking=True) if torch.is_tensor(v) else v for k, v in b.items()}


def split(ds: KolamDataset, max_dots: int | None, min_dots: int | None = None) -> tuple[list[int], list[int]]:
    keep = [i for i, e in enumerate(ds.encodings)
            if (max_dots is None or max(grid_size(e.split("|"))) <= max_dots)
            and (min_dots is None or max(grid_size(e.split("|"))) >= min_dots)]
    return [i for i in keep if i % 100], [i for i in keep if i % 100 == 0]


def score(pred: list[list[str]], true: list[list[str]]) -> Counter:
    c = Counter()
    for p, t in zip(pred, true):
        c["n"] += 1
        c["exact"] += p == t
        try:
            from_encoding(p)
            c["valid"] += 1
        except (KolamError, ValueError, IndexError):
            pass
        if p and grid_size(p) == grid_size(t):
            c["sized"] += 1
            pg, tg = encoding_grid(p), encoding_grid(t)
            m = (np.add.outer(np.arange(pg.shape[0]), np.arange(pg.shape[1])) % 2) == 1
            c["digits_ok"] += int((pg[m] == tg[m]).sum())
            c["digits"] += int(m.sum())
    return c


def evaluate(model, loader, dev, limit: int | None = None) -> dict:
    model.eval()
    c, by_size, secs = Counter(), {}, 0.0
    for b in loader:
        img = b["image"].to(dev)
        t = time.time()
        with torch.autocast(dev.type, enabled=dev.type == "cuda"):
            pred = model.predict(img)
        secs += time.time() - t
        for p, t in zip(pred, b["encoding"]):
            s = score([p], [t])
            c += s
            k = "x".join(map(str, grid_size(t)))
            by_size[k] = by_size.get(k, Counter()) + s
        if limit and c["n"] >= limit:
            break
    model.train()
    n = max(c["n"], 1)
    return {"n": c["n"], "exact": c["exact"] / n, "valid": c["valid"] / n, "sized": c["sized"] / n,
            "digits": c["digits_ok"] / max(c["digits"], 1), "ms_per_image": round(1000 * secs / n, 1),
            "exact_by_size": {k: round(v["exact"] / v["n"], 3) for k, v in sorted(by_size.items())}}


def _hms(sec: float) -> str:
    sec = int(sec)
    return f"{sec // 3600}h{sec % 3600 // 60:02d}m{sec % 60:02d}s"


def _device(name: str):
    if name != "auto":
        return torch.device(name)
    return torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


def _loader(ds, idx, a, shuffle):
    return DataLoader(Subset(ds, idx), batch_size=a.batch, shuffle=shuffle, num_workers=a.workers,
                      collate_fn=batch_items, pin_memory=True, drop_last=shuffle,
                      persistent_workers=False)


def train(a) -> None:
    dev = _device(a.device)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    ds = KolamDataset(a.src, size=a.size, augment=a.aug, seed=a.seed, transform=to_item)
    tr, va = split(ds, a.max_dots)
    sizes = [grid_size(ds.encodings[i].split("|")) for i in tr]
    cfg = {"kind": a.model, "max_rows": max(r for r, _ in sizes), "max_cols": max(c for _, c in sizes), "d": a.width,
           "backbone": a.backbone}
    step, epoch = 0, 0
    ck = out / "last.pt"
    if a.resume and ck.exists():
        state = torch.load(ck, map_location="cpu")
        cfg = state["cfg"]
    model = build(cfg, pretrained=not (a.resume and ck.exists())).to(dev)
    enc = [p for n, p in model.named_parameters() if n.startswith("enc.") and not n.startswith(("enc.p8", "enc.p16"))]
    rest = [p for n, p in model.named_parameters() if not (n.startswith("enc.") and not n.startswith(("enc.p8", "enc.p16")))]
    k = 0.1 if cfg["backbone"] == "resnet18" else 1.0                  # pretrained layers move gently
    opt = torch.optim.AdamW([{"params": enc, "k": k}, {"params": rest, "k": 1.0}], lr=a.lr, weight_decay=0.05)
    if a.resume and ck.exists():
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["opt"])
        step, epoch = state["step"], state["epoch"]
    print(f"{a.model}: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M params, {len(tr)} train / "
          f"{len(va)} val kolams, grids up to {cfg['max_rows']}x{cfg['max_cols']}, on {dev}")
    scaler = torch.amp.GradScaler(enabled=dev.type == "cuda")
    val = _loader(KolamDataset(a.src, size=a.size, augment=a.aug, seed=a.seed + 1, transform=to_item), va, a, False)
    log = open(out / "log.jsonl", "a")
    def save(path):
        """Write to a temp file, then rename: a crash mid-save never leaves a broken checkpoint."""
        tmp = path.with_suffix(".tmp")
        torch.save({"cfg": cfg, "model": model.state_dict(), "opt": opt.state_dict(), "step": step,
                    "epoch": epoch, "args": vars(a) | {"fn": None}}, tmp)
        os.replace(tmp, path)

    if step:
        print(f"resumed from {ck} at step {step}", flush=True)
    t0, step0 = time.time(), step
    while step < a.steps:
        ds.set_epoch(epoch)
        for b in _loader(ds, tr, a, True):
            lr = a.lr * min(1, (step + 1) / a.warmup) * 0.5 * (1 + math.cos(math.pi * min(step / a.steps, 1)))
            for g in opt.param_groups:
                g["lr"] = lr * g["k"]
            with torch.autocast(dev.type, enabled=dev.type == "cuda"):
                loss = model.loss(to_device(b, dev))
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            step += 1
            if step % a.log_every == 0:
                el = time.time() - t0
                rate = (step - step0) / el
                r = {"step": f"{step}/{a.steps}", "done": f"{100 * step / a.steps:.0f}%", "epoch": epoch,
                     "loss": round(loss.item(), 4), "lr": round(lr, 6), "img_s": round(rate * a.batch, 1),
                     "elapsed": _hms(el), "left": _hms((a.steps - step) / max(rate, 1e-9))}
                print(json.dumps(r), flush=True)
                log.write(json.dumps(r) + "\n")
                log.flush()
            if step % a.eval_every == 0 or step == a.steps:
                r = {"step": step, "val": evaluate(model, val, dev, a.val_n)}
                print(json.dumps(r), flush=True)
                log.write(json.dumps(r) + "\n")
                log.flush()
                save(step_ck(out, step))
            if step % a.save_every == 0 or step == a.steps:
                save(ck)
            if step >= a.steps:
                break
        epoch += 1


def step_ck(out: Path, step: int) -> Path:
    return out / f"step_{step:07d}.pt"


def load(path, dev="cpu"):
    state = torch.load(path, map_location="cpu")
    model = build(state["cfg"], pretrained=False)
    model.load_state_dict(state["model"])
    return model.to(dev).eval(), state


def eval_cmd(a) -> None:
    dev = _device(a.device)
    model, state = load(a.ckpt, dev)
    ds = KolamDataset(a.src, size=a.size or state["args"]["size"], augment=a.aug, seed=a.seed, transform=to_item)
    _, va = split(ds, a.max_dots, a.min_dots)
    print(json.dumps(evaluate(model, _loader(ds, va, a, False), dev, a.n), indent=1))


def prepare(img, size: int):
    """Any picture -> the model's input: upright (EXIF), padded to a square with its edge colour, resized."""
    from PIL import Image, ImageOps
    img = ImageOps.exif_transpose(img).convert("RGB")
    w, h = img.size
    a = np.asarray(img)
    edge = np.concatenate([a[0], a[-1], a[:, 0], a[:, -1]]).mean(0).astype(int)
    sq = Image.new("RGB", (max(w, h),) * 2, tuple(edge))
    sq.paste(img, ((max(w, h) - w) // 2, (max(w, h) - h) // 2))
    return sq.resize((size, size), Image.LANCZOS)


def _redraw(enc, size: int):
    """The kolam an encoding describes, drawn clean (or a note if it doesn't read back)."""
    from PIL import Image, ImageDraw
    from .compiler import compile_hypergraph
    from .render import Style, render_kolam
    try:
        return render_kolam(compile_hypergraph(from_encoding(enc), check=False), Style(size=size)).convert("RGB")
    except Exception as e:                                   # an answer that isn't a drawable kolam
        im = Image.new("RGB", (size, size), (40, 20, 20))
        ImageDraw.Draw(im).text((8, 8), f"not drawable:\n{type(e).__name__}", fill=(255, 200, 200))
        return im


def predict_cmd(a) -> None:
    """Read kolams from image files (or from held-out data with --data) and save a sheet:
    input | what the model read (| the truth, for --data)."""
    from PIL import Image, ImageDraw
    dev = _device(a.device)
    model, state = load(a.ckpt, dev)
    size = state["args"]["size"]
    rows = []                                                # (name, input image, true encoding or None)
    for f in a.images:
        rows.append((Path(f).name, prepare(Image.open(f), size), None))
    if a.data:
        ds = KolamDataset(a.data, size=size, augment=a.aug, seed=a.seed)
        _, va = split(ds, state["cfg"]["max_rows"])
        for i in va[:a.n]:
            r = ds.render(i)
            rows.append((f"kolam {i}", r["image"].convert("RGB"), r["encoding"]))
    if not rows:
        raise SystemExit("give image files and/or --data <shards>")
    x = torch.stack([torch.from_numpy(np.asarray(im, np.float32) / 255).permute(2, 0, 1) for _, im, _ in rows])
    preds = []
    for k in range(0, len(x), 16):
        preds += model.predict(x[k:k + 16].to(dev))
    T, top = 256, 18
    cols = 3 if a.data else 2
    sheet = Image.new("RGB", (cols * T, len(rows) * (T + top)), (10, 10, 12))
    d = ImageDraw.Draw(sheet)
    right = 0
    for k, ((name, im, true), pred) in enumerate(zip(rows, preds)):
        y = k * (T + top)
        ok = "" if true is None else ("  RIGHT" if pred == true else "  wrong")
        right += pred == true
        d.text((6, y + 3), f"{name}{ok}   read: {len(pred) // 2}x{len(pred[0]) if pred else 0}", fill=(230, 230, 230))
        sheet.paste(im.resize((T, T)), (0, y + top))
        sheet.paste(_redraw(pred, T), (T, y + top))
        if true is not None:
            sheet.paste(_redraw(true, T), (2 * T, y + top))
        print(f"{name}{ok}: " + " ".join(pred) if pred else f"{name}: (empty)")
    sheet.save(a.out)
    if a.data:
        print(f"{right}/{len(rows) - len(a.images)} held-out kolams read exactly right")
    print(f"saved {a.out}: input | model's reading" + (" | truth" if a.data else ""))


def serve_cmd(a) -> None:
    """A tiny local HTTP server for kolam_studio's predict page: POST an image to /api/predict?model=seq
    and get back the encoding; GET /api/models lists the loaded checkpoints."""
    import base64
    import io
    import json
    import time
    from http.server import BaseHTTPRequestHandler, HTTPServer
    from urllib.parse import parse_qs, urlparse
    from PIL import Image
    dev = _device(a.device)
    models = {}                                              # kind -> (model, size, ckpt path)
    for ck in a.ckpts:
        m, state = load(ck, dev)
        models[state["cfg"]["kind"]] = (m, state["args"]["size"], ck)
        print(f"loaded {state['cfg']['kind']}: {ck}")

    class Handler(BaseHTTPRequestHandler):
        def reply(self, code, body):
            data = json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if urlparse(self.path).path != "/api/models":
                return self.reply(404, {"error": "not found"})
            self.reply(200, [{"model": k, "size": s, "ckpt": c} for k, (_, s, c) in models.items()])

        def do_POST(self):
            u = urlparse(self.path)
            kind = parse_qs(u.query).get("model", [next(iter(models))])[0]
            if u.path != "/api/predict" or kind not in models:
                return self.reply(404, {"error": f"no model {kind!r}"})
            model, size, _ = models[kind]
            try:
                im = prepare(Image.open(io.BytesIO(self.rfile.read(int(self.headers["Content-Length"])))), size)
            except Exception as e:
                return self.reply(400, {"error": f"not an image: {e}"})
            x = torch.from_numpy(np.asarray(im, np.float32) / 255).permute(2, 0, 1)[None].to(dev)
            t = time.perf_counter()
            with torch.no_grad():
                enc = model.predict(x)[0]
            ms = (time.perf_counter() - t) * 1000
            buf = io.BytesIO()
            im.save(buf, "PNG")                              # what the model saw, after prepare()
            print(f"{kind}: {len(enc) // 2}x{len(enc[0]) if enc else 0} in {ms:.0f} ms")
            self.reply(200, {"model": kind, "encoding": enc, "ms": round(ms),
                             "seen": "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()})

        def do_OPTIONS(self):
            self.send_response(204)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Headers", "*")
            self.end_headers()

        def log_message(self, *args):
            pass

    print(f"serving on http://{a.host}:{a.port}  (open kolam_studio: npm run dev -> predict page)")
    HTTPServer((a.host, a.port), Handler).serve_forever()


def add_parsers(sub) -> None:
    t = sub.add_parser("train", help="train the seq (option A) or grid (option B) model")
    t.add_argument("src")
    t.add_argument("--model", choices=["seq", "grid"], default="seq")
    t.add_argument("--out", default="runs/seq")
    t.add_argument("--size", type=int, default=384, help="image side, px")
    t.add_argument("--width", type=int, default=256, help="model width")
    t.add_argument("--backbone", choices=["resnet18", "small"], default="resnet18",
                   help="resnet18: ImageNet-pretrained; small: from scratch")
    t.add_argument("--batch", type=int, default=32)
    t.add_argument("--steps", type=int, default=100_000)
    t.add_argument("--lr", type=float, default=3e-4)
    t.add_argument("--warmup", type=int, default=1000)
    t.add_argument("--aug", default="light", help="augmentation preset: none, light, photo, real")
    t.add_argument("--max-dots", type=int, default=None, help="only grids up to this many rows/cols")
    t.add_argument("--workers", type=int, default=6)
    t.add_argument("--device", default="auto")
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--log-every", type=int, default=50)
    t.add_argument("--eval-every", type=int, default=2000)
    t.add_argument("--val-n", type=int, default=500, help="val kolams per evaluation")
    t.add_argument("--save-every", type=int, default=100, help="steps between saves of last.pt")
    t.add_argument("--resume", action="store_true", help="continue from <out>/last.pt if it exists")
    t.set_defaults(fn=train)
    e = sub.add_parser("eval", help="score a checkpoint on the held-out kolams")
    e.add_argument("ckpt")
    e.add_argument("src")
    e.add_argument("--n", type=int, default=2000)
    e.add_argument("--aug", default="light")
    e.add_argument("--max-dots", type=int, default=None)
    e.add_argument("--min-dots", type=int, default=None, help="only grids with rows or cols at least this")
    e.add_argument("--size", type=int, default=None, help="image side (default: the training size)")
    e.add_argument("--batch", type=int, default=32)
    e.add_argument("--workers", type=int, default=6)
    e.add_argument("--device", default="auto")
    e.add_argument("--seed", type=int, default=1)
    e.set_defaults(fn=eval_cmd)
    q = sub.add_parser("predict", help="read kolams from your own images (and/or held-out data) into a sheet")
    q.add_argument("ckpt")
    q.add_argument("images", nargs="*", help="image files: photos, screenshots, renders")
    q.add_argument("--data", default=None, help="also read --n held-out kolams from these shards")
    q.add_argument("--n", type=int, default=8)
    q.add_argument("--aug", default="light")
    q.add_argument("--seed", type=int, default=3)
    q.add_argument("-o", "--out", default="runs/predict.png")
    q.add_argument("--device", default="auto")
    q.set_defaults(fn=predict_cmd)
    v = sub.add_parser("serve", help="local server for kolam_studio's predict page (drop an image, see the kolam)")
    v.add_argument("ckpts", nargs="+", help="checkpoints to serve, e.g. runs/pilot_seq/last.pt runs/pilot_grid/last.pt")
    v.add_argument("--host", default="127.0.0.1")
    v.add_argument("--port", type=int, default=8765)
    v.add_argument("--device", default="auto")
    v.set_defaults(fn=serve_cmd)
