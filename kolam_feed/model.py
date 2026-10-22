"""The two models: image -> kolam encoding (spec §12). Both share one CNN encoder.

    SeqModel   option A: a transformer decoder writes the encoding token by token (encoding.to_tokens),
               greedy and constrained by encoding.allowed_next, so every answer is well formed.
    GridModel  option B: every gap at once. Heads give the dot grid's rows and cols and the image
               positions of the gap lattice's 4 corners; the features under each gap (placed by the
               homography of those corners) are classified into its digit 0-8.

Both also have a "kolam" head: one logit, is this picture a kolam at all? (trained with non-kolam
pictures, data.NotKolamDataset; only kolam pictures train the reading). is_kolam(img) -> probability.

Inputs are float images (B, 3, S, S) scaled to [0, 1]. Lattice points are (lj, li) = (x, y) order,
like render_kolam's frame M.
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoding import VOCAB, allowed_next, from_tokens

BOS, PAD = len(VOCAB), len(VOCAB) + 1
TOK = {t: i for i, t in enumerate(VOCAB)}
BAR, END = TOK["|"], TOK["<end>"]


# ------------------------------------------------------------------ shared encoder
class _Block(nn.Module):
    def __init__(self, cin: int, cout: int, stride: int):
        super().__init__()
        self.c1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.b1 = nn.BatchNorm2d(cout)
        self.c2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.b2 = nn.BatchNorm2d(cout)
        self.skip = (nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False), nn.BatchNorm2d(cout))
                     if stride != 1 or cin != cout else nn.Identity())

    def forward(self, x):
        return F.relu(self.b2(self.c2(F.relu(self.b1(self.c1(x))))) + self.skip(x))


class Encoder(nn.Module):
    """Features at stride 8 (f8, for reading gaps) and stride 16 (f16, for context), both `width` wide.
    backbone "resnet18" / "resnet34": torchvision's ResNet up to layer3, ImageNet weights when pretrained
    (cached under TORCH_HOME; resnet34 is the deeper one, same channels); "small": a small ResNet from scratch."""

    def __init__(self, width: int = 256, backbone: str = "resnet18", pretrained: bool = True):
        super().__init__()
        self.backbone = backbone
        if backbone in ("resnet18", "resnet34"):
            import torchvision
            r = getattr(torchvision.models, backbone)(weights="IMAGENET1K_V1" if pretrained else None)
            self.stem = nn.Sequential(r.conv1, r.bn1, r.relu, r.maxpool, r.layer1)
            self.s8, self.s16 = r.layer2, r.layer3
            self.p8, self.p16 = nn.Conv2d(128, width, 1), nn.Conv2d(256, width, 1)
            self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
            self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
            return
        w = width
        self.stem = nn.Sequential(nn.Conv2d(3, w // 8, 3, 2, 1, bias=False), nn.BatchNorm2d(w // 8), nn.ReLU())
        self.s4 = nn.Sequential(_Block(w // 8, w // 4, 2), _Block(w // 4, w // 4, 1))
        self.s8 = nn.Sequential(_Block(w // 4, w // 2, 2), _Block(w // 2, w // 2, 1), _Block(w // 2, w, 1))
        self.s16 = nn.Sequential(_Block(w, w, 2), _Block(w, w, 1), _Block(w, w, 1))
        self.register_buffer("mean", torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1))

    def forward(self, x):
        if self.backbone != "small":
            f8 = self.s8(self.stem((x - self.mean) / self.std))
            return self.p8(f8), self.p16(self.s16(f8))
        f8 = self.s8(self.s4(self.stem((x - self.mean) * 2)))
        return f8, self.s16(f8)


def kolam_loss(logit: torch.Tensor, n_kolams: int) -> torch.Tensor:
    """The "is this a kolam?" head: the first n_kolams pictures of the batch are kolams, the rest are not."""
    y = torch.zeros_like(logit)
    y[:n_kolams] = 1
    return 0.5 * F.binary_cross_entropy_with_logits(logit.float(), y.float())


def _pos2d(h: int, w: int, d: int, device) -> torch.Tensor:
    """Fixed 2D sine position code, (h*w, d)."""
    q = d // 4
    f = torch.exp(torch.arange(q, device=device) * (-math.log(1e4) / q))
    y, x = torch.meshgrid(torch.arange(h, device=device), torch.arange(w, device=device), indexing="ij")
    y, x = y.reshape(-1, 1) * f, x.reshape(-1, 1) * f
    return torch.cat([y.sin(), y.cos(), x.sin(), x.cos()], 1)


# ------------------------------------------------------------------ option A: sequence
def token_slots(ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """For each position p of the decoder input (BOS + targets[:-1]): the encoding row and the column
    of target p, read off the prefix ids[1..p] (rows end at "|"; BOS counts as the start of row 0)."""
    bar = ids == BAR
    row = bar.long().cumsum(1)
    idx = torch.arange(ids.shape[1], device=ids.device).expand_as(ids)
    last = torch.where(bar | (idx == 0), idx, torch.zeros_like(idx)).cummax(1).values
    return row, idx - last


class SeqModel(nn.Module):
    def __init__(self, max_rows: int = 16, max_cols: int = 16, d: int = 256, layers: int = 4, heads: int = 8,
                 backbone: str = "resnet18", pretrained: bool = True):
        super().__init__()
        self.cfg = dict(kind="seq", max_rows=max_rows, max_cols=max_cols, d=d, layers=layers, heads=heads,
                        backbone=backbone)
        self.max_len = (2 * max_rows + 1) * (max_cols + 2) + 1
        self.enc = Encoder(d, backbone, pretrained)
        self.tok = nn.Embedding(len(VOCAB) + 2, d)
        self.row = nn.Embedding(2 * max_rows + 3, d)
        self.col = nn.Embedding(max_cols + 3, d)
        layer = nn.TransformerDecoderLayer(d, heads, 4 * d, 0.1, batch_first=True, norm_first=True)
        self.dec = nn.TransformerDecoder(layer, layers)
        self.norm = nn.LayerNorm(d)
        self.out = nn.Linear(d, len(VOCAB))
        self.kolam = nn.Linear(d, 1)

    def encode(self, img):
        """-> decoder memory (B, N, d) and the is-a-kolam logit (B,)."""
        f = self.enc(img)[1]
        B, D, h, w = f.shape
        return f.flatten(2).transpose(1, 2) + _pos2d(h, w, D, f.device), self.kolam(f.mean((2, 3)))[:, 0]

    def memory(self, img):
        return self.encode(img)[0]

    @torch.no_grad()
    def is_kolam(self, img) -> torch.Tensor:
        return torch.sigmoid(self.encode(img)[1].float())

    def logits(self, mem, ids):
        row, col = token_slots(ids)
        x = self.tok(ids) + self.row(row.clamp(max=self.row.num_embeddings - 1)) + \
            self.col(col.clamp(max=self.col.num_embeddings - 1))
        L = ids.shape[1]
        mask = torch.triu(torch.full((L, L), float("-inf"), device=ids.device), 1)
        return self.out(self.norm(self.dec(x, mem, tgt_mask=mask, tgt_key_padding_mask=ids == PAD)))

    def loss(self, batch):
        """Reading loss on the kolams; plus the kolam head when the batch has non-kolams ("neg")."""
        tgt = batch["tokens"]                                         # (B, L) padded with PAD
        inp = torch.cat([torch.full_like(tgt[:, :1], BOS), tgt[:, :-1]], 1)
        inp = inp.masked_fill(inp == PAD, END)                        # positions past <end> are ignored anyway
        img, neg, B = batch["image"], batch.get("neg"), len(tgt)
        mem, k = self.encode(img if neg is None else torch.cat([img, neg]))
        lg = self.logits(mem[:B], inp)
        return F.cross_entropy(lg.flatten(0, 1), tgt.flatten(), ignore_index=PAD) + kolam_loss(k, B)

    @torch.no_grad()
    def predict(self, img) -> list[list[str]]:
        """Greedy, constrained decoding -> one encoding (list of digit rows) per image."""
        mem = self.memory(img)
        B = img.shape[0]
        ids = torch.full((B, 1), BOS, dtype=torch.long, device=img.device)
        seqs, done = [[] for _ in range(B)], [False] * B
        for _ in range(self.max_len):
            lg = self.logits(mem, ids)[:, -1].float().cpu()
            nxt = []
            for b in range(B):
                if done[b]:
                    nxt.append(END)
                    continue
                ok = allowed_next(seqs[b]) if len(seqs[b]) < self.max_len - 1 else []
                ok = ok or ["<end>"]
                m = torch.full((len(VOCAB),), float("-inf"))
                m[[TOK[t] for t in ok]] = 0
                t = int((lg[b] + m).argmax())
                seqs[b].append(VOCAB[t])
                done[b] = t == END
                nxt.append(t)
            if all(done):
                break
            ids = torch.cat([ids, torch.tensor(nxt, device=img.device)[:, None]], 1)
        return [from_tokens(s) for s in seqs]


# ------------------------------------------------------------------ option B: grid
def gap_points(rows: int, cols: int) -> np.ndarray:
    """Every gap of a rows x cols dot grid in encoding order: (lj, li, orientation), orientation 0 for
    an up/down gap (even li), 1 for a left/right gap (odd li)."""
    pts = []
    for li in range(2 * rows + 1):
        for lj in range(1 - li % 2, 2 * cols + 1, 2):
            pts.append((lj, li, li % 2))
    return np.array(pts, np.float32)


def homography(src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
    """(B, 4, 2) -> (B, 4, 2) point pairs -> (B, 3, 3) H with dst ~ H src."""
    x, y, u, v = src[..., 0], src[..., 1], dst[..., 0], dst[..., 1]
    o, z = torch.ones_like(x), torch.zeros_like(x)
    A = torch.cat([torch.stack([x, y, o, z, z, z, -u * x, -u * y], -1),
                   torch.stack([z, z, z, x, y, o, -v * x, -v * y], -1)], 1)
    h = torch.linalg.solve(A, torch.cat([u, v], 1))
    return torch.cat([h, torch.ones_like(h[:, :1])], 1).view(-1, 3, 3)


def lattice_corners(rows, cols) -> torch.Tensor:
    """(B,) rows, cols -> (B, 4, 2) lattice corners (lj, li): top-left, top-right, bottom-right, bottom-left."""
    z = torch.zeros_like(rows, dtype=torch.float32)
    R, C = 2 * rows.float(), 2 * cols.float()
    return torch.stack([torch.stack([z, z], -1), torch.stack([C, z], -1),
                        torch.stack([C, R], -1), torch.stack([z, R], -1)], 1)


class GridModel(nn.Module):
    def __init__(self, max_rows: int = 16, max_cols: int = 16, d: int = 256, backbone: str = "resnet18",
                 pretrained: bool = True, ctx_layers: int = 2):
        super().__init__()
        self.cfg = dict(kind="grid", max_rows=max_rows, max_cols=max_cols, d=d, backbone=backbone, ctx_layers=ctx_layers)
        self.enc = Encoder(d, backbone, pretrained)
        self.ctx = nn.TransformerEncoder(nn.TransformerEncoderLayer(d, 8, 4 * d, 0.1, batch_first=True,
                                                                    norm_first=True), ctx_layers)
        self.size_head = nn.Linear(d, max_rows + max_cols)            # rows 1..max_rows, cols 1..max_cols
        self.corner_head = nn.Conv2d(d, 4, 1)                         # soft-argmax heatmaps on f8
        self.orient = nn.Embedding(2, 2 * d)
        self.gap_head = nn.Sequential(nn.Linear(2 * d, d), nn.ReLU(), nn.Linear(d, len(VOCAB) - 2))
        self.kolam = nn.Linear(d, 1)

    def features(self, img):
        f8, f16 = self.enc(img)
        B, D, h, w = f16.shape
        t = self.ctx(f16.flatten(2).transpose(1, 2) + _pos2d(h, w, D, f16.device))
        f16 = t.transpose(1, 2).reshape(B, D, h, w)
        R = self.cfg["max_rows"]
        sz = self.size_head(t.mean(1))
        hm = self.corner_head(f8)                                     # (B, 4, h8, w8)
        B, _, h8, w8 = hm.shape
        p = hm.flatten(2).softmax(-1).view(B, 4, h8, w8)
        S = img.shape[-1]
        xs = (torch.arange(w8, device=img.device) + 0.5) * S / w8
        ys = (torch.arange(h8, device=img.device) + 0.5) * S / h8
        corners = torch.stack([(p.sum(2) * xs).sum(-1), (p.sum(3) * ys).sum(-1)], -1)   # (B, 4, 2) pixels
        return f8, f16, sz[:, :R], sz[:, R:], corners, self.kolam(t.mean(1))[:, 0]

    @torch.no_grad()
    def is_kolam(self, img) -> torch.Tensor:
        return torch.sigmoid(self.features(img)[5].float())

    def read_gaps(self, f8, f16, M, pts, bidx, S):
        """Digit logits for gaps pts (N, 3) of images bidx (N,), placed in pixels by M (B, 3, 3)."""
        P = torch.cat([pts[:, :2], torch.ones_like(pts[:, :1])], 1)
        q = torch.einsum("nij,nj->ni", M[bidx], P)
        xy = q[:, :2] / q[:, 2:].clamp(min=1e-6)
        g = xy / S * 2 - 1
        x = torch.zeros(len(bidx), 2 * f8.shape[1], device=f8.device)    # float32: under autocast f8 and f16
        for b in bidx.unique().tolist():                                   # can come out in different precisions
            sel = bidx == b
            gb = g[sel].view(1, 1, -1, 2)
            x[sel] = torch.cat([F.grid_sample(f[b:b + 1], gb.to(f.dtype), align_corners=False)[0, :, 0].T.float()
                                for f in (f8, f16)], 1)
        x = x + self.orient(pts[:, 2].long())
        return self.gap_head(x)

    def loss(self, batch):
        img, neg = batch["image"], batch.get("neg")
        S, B = img.shape[-1], len(img)
        f8, f16, lr, lc, corners, k = self.features(img if neg is None else torch.cat([img, neg]))
        lr, lc, corners = lr[:B], lc[:B], corners[:B]                 # gaps below index only the kolams
        rows, cols = batch["rows"], batch["cols"]
        M = batch["M"]
        lat = lattice_corners(rows, cols)
        true_c = torch.einsum("bij,bkj->bki", M, torch.cat([lat, torch.ones_like(lat[..., :1])], -1))
        true_c = true_c[..., :2] / true_c[..., 2:]
        l_size = F.cross_entropy(lr, rows - 1) + F.cross_entropy(lc, cols - 1)
        l_corner = 10 * F.l1_loss(corners / S, true_c / S)            # 0.1 = a tenth of the picture off
        lg = self.read_gaps(f8, f16, M, batch["gap_pts"], batch["gap_b"], S)     # read at the true positions
        l_gap = F.cross_entropy(lg, batch["gap_y"])
        return l_gap + l_size + l_corner + kolam_loss(k, B)

    @torch.no_grad()
    def predict(self, img) -> list[list[str]]:
        S = img.shape[-1]
        f8, f16, lr, lc, corners, _ = self.features(img)
        rows, cols = lr.argmax(1) + 1, lc.argmax(1) + 1
        M = homography(lattice_corners(rows, cols), corners)
        pts, bidx = [], []
        for b in range(img.shape[0]):
            g = gap_points(int(rows[b]), int(cols[b]))
            pts.append(torch.from_numpy(g))
            bidx += [b] * len(g)
        pts = torch.cat(pts).to(img.device)
        bidx = torch.tensor(bidx, device=img.device)
        digits = self.read_gaps(f8, f16, M, pts, bidx, S).argmax(1).cpu().tolist()
        out, k = [], 0
        for b in range(img.shape[0]):
            R, C = int(rows[b]), int(cols[b])
            enc = []
            for li in range(2 * R + 1):
                n = C if li % 2 == 0 else C + 1
                enc.append("".join(str(d) for d in digits[k:k + n]))
                k += n
            out.append(enc)
        return out


def build(cfg: dict, pretrained: bool = True) -> nn.Module:
    """A model from its cfg; pretrained=False skips the ImageNet download (when loading a checkpoint)."""
    cfg = dict(cfg, pretrained=pretrained)
    kind = cfg.pop("kind")
    return {"seq": SeqModel, "grid": GridModel}[kind](**cfg)
