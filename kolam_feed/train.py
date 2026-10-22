"""Train and evaluate the image -> encoding models (kolam_feed/model.py) on kolam_gen shards.

    python -m kolam_feed train data/enc_1m --model seq  --out runs/seq  --max-dots 5
    python -m kolam_feed train data/enc_1m --model grid --out runs/grid --resume   # continue from runs/grid/last.pt
    python -m kolam_feed eval  runs/seq/last.pt data/enc_1m --n 2000 --aug photo

Images are rendered on the fly (KolamDataset), a fresh look every epoch. Every 100th kolam is held out
for validation. Each batch also gets --not-kolam-share non-kolams (NotKolamDataset: photos from
--not-kolam, empty floors, dot grids, scribbles) for the model's "is this a kolam?" head.
Checkpoints: <out>/last.pt every --save-every steps (resume from it) and <out>/best.pt (best held-out
exact). Scores: exact (the whole encoding right, i.e. the exact label), valid (the answer reads back
into a kolam), digits (gap accuracy where the predicted grid has the right size), keeps_kolams /
rejects (the kolam head on held-out kolams / non-kolams). Live view: python -m kolam_feed.watch <out>.
"""
from __future__ import annotations

import copy
import json
import math
import os
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from .data import KolamDataset, NotKolamDataset
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


def split(ds: KolamDataset, max_dots: int | None, min_dots: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Train / val kolam indices (every 100th is val), as int arrays: DataLoader workers share an array,
    while a list of millions of ints gets copied into each of them (and ran a 2M-kolam run out of RAM)."""
    keep = np.array([i for i, e in enumerate(ds.encodings)
                     if (max_dots is None or max(grid_size(e.split("|"))) <= max_dots)
                     and (min_dots is None or max(grid_size(e.split("|"))) >= min_dots)], np.int64)
    return keep[keep % 100 != 0], keep[keep % 100 == 0]


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


def evaluate(model, loader, dev, limit: int | None = None, neg=None, real=None) -> dict:
    """Reading scores on held-out kolams; with `neg` (a loader of non-kolam image batches) also the
    kolam head: keeps_kolams (kolams it says are kolams) and rejects (non-kolams it turns away); with
    `real` (a batch of real kolam photos, no labels needed) keeps_real: the real photos it keeps."""
    model.eval()
    c, by_size, by_band, secs = Counter(), {}, {}, 0.0
    for b in loader:
        img = b["image"].to(dev)
        t = time.time()
        with torch.autocast(dev.type, enabled=dev.type == "cuda"):
            pred = model.predict(img)
            if neg is not None:
                c["kept"] += int((model.is_kolam(img) >= 0.5).sum())
        secs += time.time() - t
        for p, t in zip(pred, b["encoding"]):
            s = score([p], [t])
            c += s
            k = "x".join(map(str, grid_size(t)))
            by_size[k] = by_size.get(k, Counter()) + s
            m = max(grid_size(t))
            lo = 3 * ((m - 3) // 3) + 3 if m >= 3 else 1
            band = f"{lo}-{lo + 2}" if m >= 3 else "1-2"              # 3-5, 6-8, ... by the longer side
            by_band[band] = by_band.get(band, Counter()) + s
        if limit and c["n"] >= limit:
            break
    r = {}
    if neg is not None:
        nn_, rejected = 0, 0
        with torch.autocast(dev.type, enabled=dev.type == "cuda"):
            for img in neg:
                rejected += int((model.is_kolam(img.to(dev)) < 0.5).sum())
                nn_ += len(img)
                if limit and nn_ >= limit:
                    break
        r = {"keeps_kolams": c["kept"] / max(c["n"], 1), "rejects": rejected / max(nn_, 1), "n_not_kolam": nn_}
    if real is not None:
        with torch.autocast(dev.type, enabled=dev.type == "cuda"):
            r["keeps_real"] = float((model.is_kolam(real.to(dev)) >= 0.5).float().mean())
    model.train()
    n = max(c["n"], 1)
    return {"n": c["n"], "exact": c["exact"] / n, "valid": c["valid"] / n, "sized": c["sized"] / n,
            "digits": c["digits_ok"] / max(c["digits"], 1), "ms_per_image": round(1000 * secs / n, 1),
            "exact_by_size": {k: round(v["exact"] / v["n"], 3) for k, v in sorted(by_size.items())},
            "by_band": {k: {"n": v["n"], "exact": round(v["exact"] / v["n"], 3),
                            "digits": round(v["digits_ok"] / max(v["digits"], 1), 4), "sized": round(v["sized"] / v["n"], 3)}
                        for k, v in sorted(by_band.items(), key=lambda kv: int(kv[0].split("-")[0]))}} | r


def real_photos(folder, size: int) -> torch.Tensor:
    """Every picture in a folder of real kolam photos, prepared like any input (prepare)."""
    from PIL import Image
    from .data import PHOTO_EXT
    fs = sorted(f for f in Path(folder).rglob("*") if f.suffix.lower() in PHOTO_EXT)
    return torch.stack([torch.from_numpy(np.asarray(prepare(Image.open(f), size), np.float32) / 255).permute(2, 0, 1)
                        for f in fs])


def neg_images(items: list[dict]) -> torch.Tensor:
    """collate for NotKolamDataset: just the pictures, as one float batch."""
    return torch.stack([torch.from_numpy(np.asarray(it["image"].convert("RGB"), np.float32) / 255).permute(2, 0, 1)
                        for it in items])


def load_weights(model, sd: dict) -> list[str]:
    """Load a checkpoint's weights; one from before the kolam head loads too (the head starts fresh).
    Returns the missing keys."""
    missing, unexpected = model.load_state_dict(sd, strict=False)
    bad = [k for k in missing if not k.startswith("kolam.")] + list(unexpected)
    if bad:
        raise RuntimeError(f"checkpoint does not fit the model: {bad[:5]}")
    return missing


def _hms(sec: float) -> str:
    sec = int(sec)
    return f"{sec // 3600}h{sec % 3600 // 60:02d}m{sec % 60:02d}s"


def _device(name: str):
    if name != "auto":
        return torch.device(name)
    return torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


def _loader(ds, idx, a, shuffle, workers: int | None = None):
    """One batch queued per worker (prefetch 1): each queued batch of 384 px images is ~40 MB of shared
    memory, which counts on a small-RAM box (WSL's 7.7 GB)."""
    n = a.workers if workers is None else workers
    return DataLoader(Subset(ds, idx), batch_size=a.batch, shuffle=shuffle, num_workers=n,
                      collate_fn=batch_items, pin_memory=True, drop_last=shuffle,
                      persistent_workers=False, prefetch_factor=1 if n else None)


def _give_back_memory() -> None:
    """Return what loading millions of kolams freed to the OS before the workers fork, so they don't
    each start with it (glibc keeps freed heap otherwise)."""
    import ctypes
    import gc
    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except OSError:
        pass


def train(a) -> None:
    dev = _device(a.device)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    floors = NotKolamDataset(a.not_kolam, split="train").files if a.not_kolam and a.photo_floor else None
    ds = KolamDataset(a.src, size=a.size, augment=a.aug, seed=a.seed, transform=to_item, photo_floors=floors,
                      p_photo_floor=a.photo_floor)
    tr, va = split(ds, a.max_dots)
    sizes = [grid_size(ds.encodings[i].split("|")) for i in tr]
    cfg = {"kind": a.model, "max_rows": max(r for r, _ in sizes), "max_cols": max(c for _, c in sizes), "d": a.width,
           "backbone": a.backbone} | ({"ctx_layers": a.ctx_layers} if a.model == "grid" else {})
    step, epoch = 0, 0
    ck = out / "last.pt"
    if a.resume and ck.exists():
        state = torch.load(ck, map_location="cpu")
        cfg = state["cfg"]
    model = build(cfg, pretrained=not (a.resume and ck.exists())).to(dev)
    enc = [p for n, p in model.named_parameters() if n.startswith("enc.") and not n.startswith(("enc.p8", "enc.p16"))]
    rest = [p for n, p in model.named_parameters() if not (n.startswith("enc.") and not n.startswith(("enc.p8", "enc.p16")))]
    k = 0.1 if cfg["backbone"].startswith("resnet") else 1.0           # pretrained layers move gently
    opt = torch.optim.AdamW([{"params": enc, "k": k}, {"params": rest, "k": 1.0}], lr=a.lr, weight_decay=0.05)
    if a.resume and ck.exists():
        if not load_weights(model, state["model"]):
            opt.load_state_dict(state["opt"])                         # (a model that grew the kolam head restarts AdamW)
        step, epoch = state["step"], state["epoch"]
    print(f"{a.model}: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M params, {len(tr)} train / "
          f"{len(va)} val kolams, grids up to {cfg['max_rows']}x{cfg['max_cols']}, on {dev}")
    scaler = torch.amp.GradScaler(enabled=dev.type == "cuda")
    vds = copy.copy(ds)                                               # same kolams (shared, not loaded again),
    vds.seed = a.seed + 1                                             # its own looks
    val = _loader(vds, va, a, False, workers=min(2, a.workers))
    real = real_photos(a.real_kolams, a.size) if a.real_kolams else None
    n_neg = round(a.batch * a.not_kolam_share)
    neg_ds, neg_val, neg_it = None, None, iter(())
    if n_neg:                                                         # non-kolams for the kolam head
        neg_ds = NotKolamDataset(a.not_kolam, a.size, a.seed, a.aug)
        nv = NotKolamDataset(a.not_kolam, a.size, a.seed + 1, a.aug, split="val")
        neg_val = DataLoader(nv, batch_size=a.batch, num_workers=min(2, a.workers // 4), collate_fn=neg_images)
        print(f"not-kolams: {n_neg} per batch, {len(neg_ds.files)} photos" + (f" from {a.not_kolam}" if a.not_kolam else ""))

    def next_neg(epoch):
        nonlocal neg_it
        b = next(neg_it, None)
        if b is None:
            neg_ds.set_epoch(epoch)
            neg_it = iter(DataLoader(neg_ds, batch_size=n_neg, shuffle=True, drop_last=True, collate_fn=neg_images,
                                     num_workers=max(1, a.workers // 4), pin_memory=True))
            b = next(neg_it)
        return b

    log = open(out / "log.jsonl", "a")
    best = max((json.loads(l)["val"]["exact"] for l in open(out / "log.jsonl") if '"val"' in l), default=-1)
    look = va[:a.samples]
    def save(path):
        """Write to a temp file, then rename: a crash mid-save never leaves a broken checkpoint."""
        tmp = path.with_suffix(".tmp")
        torch.save({"cfg": cfg, "model": model.state_dict(), "opt": opt.state_dict(), "step": step,
                    "epoch": epoch, "args": vars(a) | {"fn": None}}, tmp)
        os.replace(tmp, path)

    if step:
        print(f"resumed from {ck} at step {step}", flush=True)
    _give_back_memory()
    t0, step0 = time.time(), step
    while step < a.steps:
        ds.set_epoch(epoch)
        for b in _loader(ds, tr, a, True):
            lr = a.lr * min(1, (step + 1) / a.warmup) * 0.5 * (1 + math.cos(math.pi * min(step / a.steps, 1)))
            for g in opt.param_groups:
                g["lr"] = lr * g["k"]
            if n_neg:
                b["neg"] = next_neg(epoch)
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
                r = {"step": step, "val": evaluate(model, val, dev, a.val_n, neg_val, real)}
                print(json.dumps(r), flush=True)
                log.write(json.dumps(r) + "\n")
                log.flush()
                if len(look):
                    samples(model, val.dataset.dataset, look, dev, out, step)
                if r["val"]["exact"] > best:                          # keep only last.pt and the best so far
                    best = r["val"]["exact"]
                    save(out / "best.pt")
            if step % a.save_every == 0 or step == a.steps:
                save(ck)
            if step >= a.steps:
                break
        epoch += 1


def load(path, dev="cpu"):
    state = torch.load(path, map_location="cpu")
    model = build(state["cfg"], pretrained=False)
    load_weights(model, state["model"])
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


def _sheet(rows, preds, T: int = 256):
    """One row per kolam: input | the model's reading, drawn clean | the truth (when known)."""
    from PIL import Image, ImageDraw
    top = 18
    cols = 3 if any(t is not None for _, _, t in rows) else 2
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
    return sheet, right


def samples(model, ds, idx, dev, out: Path, step: int, T: int = 160) -> None:
    """What the model reads now, for the live view (`python -m kolam_feed.watch`): fixed held-out kolams
    under <out>/samples/: input_<k>.png and truth_<k>.png once, read_<step>_<k>.png at every eval, and
    index.json (per eval: step and, per kolam, right or not and the grid size read)."""
    d = out / "samples"
    d.mkdir(exist_ok=True)
    rows = [ds.render(i) for i in idx]
    x = torch.stack([torch.from_numpy(np.asarray(r["image"].convert("RGB"), np.float32) / 255).permute(2, 0, 1)
                     for r in rows])
    model.eval()
    with torch.no_grad(), torch.autocast(dev.type, enabled=dev.type == "cuda"):
        preds = model.predict(x.to(dev))
    model.train()
    size = lambda e: f"{len(e) // 2}x{len(e[0])}" if e else "0x0"
    items = []
    for k, (i, r, p) in enumerate(zip(idx, rows, preds)):
        if not (d / f"input_{k}.png").exists():
            r["image"].convert("RGB").resize((T, T)).save(d / f"input_{k}.png")
            _redraw(r["encoding"], T).save(d / f"truth_{k}.png")
        _redraw(p, T).save(d / f"read_{step:07d}_{k}.png")
        items.append({"kolam": int(i), "right": p == r["encoding"], "read": size(p), "true": size(r["encoding"])})
    f = d / "index.json"
    index = [e for e in (json.loads(f.read_text()) if f.exists() else []) if e["step"] != step]
    index.append({"step": step, "items": items})
    tmp = f.with_suffix(".tmp")
    tmp.write_text(json.dumps(index))
    os.replace(tmp, f)


def predict_cmd(a) -> None:
    """Read kolams from image files (or from held-out data with --data) and save a sheet:
    input | what the model read (| the truth, for --data)."""
    from PIL import Image
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
    sheet, right = _sheet(rows, preds)
    for (name, _, true), pred in zip(rows, preds):
        ok = "" if true is None else ("  RIGHT" if pred == true else "  wrong")
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


def export_cmd(a) -> None:
    """A seq checkpoint -> two ONNX files for the browser (kolam_studio/index.html, no server):
    encoder.onnx  image (1, 3, S, S) in [0, 1] -> mem (1, N, d) and kolam (1,), the chance it is a kolam
                  at all, run once per picture;
    decoder.onnx  mem, ids, row, col (1, max_len) padded, pos (1,) -> logits of the token after pos (1, 11),
                  run per token (the causal mask keeps the padding out).
    The page works out row/col (model.token_slots) and the constrained decoding (encoding.allowed_next)."""
    import torch.nn as nn
    model, state = load(a.ckpt)
    assert state["cfg"]["kind"] == "seq", "export takes a seq checkpoint"
    size, out = state["args"]["size"], Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    class Wrap(nn.Module):
        def __init__(self):
            super().__init__()
            self.m = model

    class Enc(Wrap):
        def forward(self, img):
            mem, k = self.m.encode(img)
            return mem, torch.sigmoid(k)

    class Dec(Wrap):
        def forward(self, mem, ids, row, col, pos):
            m = self.m
            x = m.tok(ids) + m.row(row.clamp(max=m.row.num_embeddings - 1)) + \
                m.col(col.clamp(max=m.col.num_embeddings - 1))
            i = torch.arange(ids.shape[1])
            mask = torch.where(i[None, :] > i[:, None], float("-inf"), 0.0)
            return m.out(m.norm(m.dec(x, mem, tgt_mask=mask)))[0].index_select(0, pos)

    img = torch.rand(1, 3, size, size)
    mem = model.memory(img)
    ids = torch.full((1, model.max_len), len(TOK))
    rc = torch.zeros_like(ids)
    with torch.no_grad():
        torch.onnx.export(Enc(), (img,), out / "encoder.onnx", input_names=["image"], output_names=["mem", "kolam"],
                          opset_version=17, dynamo=False)
        torch.onnx.export(Dec(), (mem, ids, rc, rc, torch.tensor([0])), out / "decoder.onnx",
                          input_names=["mem", "ids", "row", "col", "pos"], output_names=["logits"],
                          opset_version=17, dynamo=False)
    meta = {"size": size, "max_len": model.max_len, "bos": len(TOK), "ckpt": str(a.ckpt)}
    (out / "model.json").write_text(json.dumps(meta))
    print(f"wrote {out}/encoder.onnx, decoder.onnx, model.json ({size}px)")


def add_parsers(sub) -> None:
    t = sub.add_parser("train", help="train the seq (option A) or grid (option B) model")
    t.add_argument("src")
    t.add_argument("--model", choices=["seq", "grid"], default="seq")
    t.add_argument("--out", default="runs/seq")
    t.add_argument("--size", type=int, default=384, help="image side, px")
    t.add_argument("--width", type=int, default=256, help="model width")
    t.add_argument("--backbone", choices=["resnet18", "resnet34", "small"], default="resnet18",
                   help="resnet18 / resnet34: ImageNet-pretrained; small: from scratch")
    t.add_argument("--ctx-layers", type=int, default=2, help="grid model: layers of the context transformer "
                   "(it sees the whole picture and counts the grid)")
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
    t.add_argument("--not-kolam", default=None, help="folder of photos that are not kolams (e.g. Imagenette), "
                   "mixed with drawn non-kolams (floors, dot grids, scribbles) for the kolam head")
    t.add_argument("--photo-floor", type=float, default=0.2, help="share of kolams drawn on a --not-kolam photo "
                   "as the floor, so photo-ness can't mean 'not a kolam' (0: off)")
    t.add_argument("--real-kolams", default=None, help="folder of real kolam photos (no labels): every eval scores "
                   "how many the kolam head keeps")
    t.add_argument("--not-kolam-share", type=float, default=0.15, help="non-kolams per batch, as a share of --batch (0: off)")
    t.add_argument("--samples", type=int, default=6, help="held-out kolams the live view shows at each eval (0: none)")
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
    x = sub.add_parser("export", help="seq checkpoint -> ONNX for the in-browser reader (kolam_studio/index.html)")
    x.add_argument("ckpt")
    x.add_argument("-o", "--out", default="kolam_studio/public/model")
    x.set_defaults(fn=export_cmd)
