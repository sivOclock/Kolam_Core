# Model experiments: image → kolam encoding

A learning log for the model side of Kolam Labs. It explains what we built, why, how each piece works,
and what each experiment showed. Code: `kolam_feed/model.py` (the models), `kolam_feed/train.py`
(training and scoring), `kolam_feed/augment.py` (the images the models see).

---

## 1. The task

**Input:** a picture of a kolam, eventually a phone photo: tilted, glare, any floor, any ink.
**Output:** its **kolam encoding** (spec §12): one digit per gap between neighbouring dots.

```
0 wall   1 cross   2 tube   3 4 5  plain / left / right horn facing E or S   6 7 8  same facing W or N
```

Rows alternate `cols` digits (up/down gaps) and `cols+1` digits (left/right gaps), `2*rows+1` rows in
all. A 3×3 kolam:

```
000        ← gaps above row 0 (border: walls or horns pointing out)
0110       ← gaps left/right of the dots in row 0
111        ← gaps between row 0 and row 1
...
```

The encoding converts **exactly** to and from the ordered hypergraph (`kolam_feed/encoding.py`), so
reading the digits right means reading the kolam right. Every training image is rendered from a known
encoding, so every label is exact: no human labelling, unlimited data.

What "right" means:
- **exact**: the whole encoding is right, which means the whole kolam. The real score.
- **digits**: the share of gaps read right (only counted when the grid size is right).
- **valid**: the answer reads back into a real kolam.
- **sized**: the grid size (rows × cols) is right.

---

## 2. The data the model sees

```
kolam_gen (Rust)  →  shards of encodings  →  KolamDataset (Python): encoding → hypergraph → strokes → image
```

Images are **rendered on the fly**: every epoch, every kolam gets a new look, seeded by
`(seed, epoch, index)` so any image can be made again. The model never sees the same picture twice,
so it can't memorise pixels; it has to learn the structure.

### Augmentation (`kolam_feed/augment.py`)

Three layers:

| Layer | What | Label |
|---|---|---|
| **style** | palette (chalk / ink / colour), line width, dot size, margin, blur, noise, hand jitter | unchanged |
| **turn** | one of 8 rotations/mirrors of the dot grid | **turned with it** (mirrors swap LEFT_HORN ↔ RIGHT_HORN) |
| **image** | angle, camera tilt, floors, inks, glare, blur, ... | unchanged |

Image effects: rotation ±30° (shrunk to fit), **camera tilt** (foreshortening: the far side shrinks),
perspective, 10 procedural **floors** (cement, red oxide, mud, sand, tiles, granite, wood, paper,
chalkboard, asphalt), real **inks** (rice flour, chalk, turmeric, kumkum, rangoli colours,
multi-colour), thicker lines, petals and dirt, **glare** (flash hotspot / sun sheen / window wash),
vignette, lighting gradient, shadow, colour/gamma, broken chalk, motion blur *or* defocus *or* low-res
(never stacked), sensor noise, JPEG.

Presets: `none`, `light` (synthetic variety), `photo`, `real` (every image a new angle, 85% real
floors, 70% glare).

**Why rotation stops at ±30°:** turns by 90° are already covered by the turn layer, which turns the label
too. Past 45° a rotated kolam looks the same as its 90°-turned twin, and the label would be ambiguous.

**Why the effects keep the label:** they change pixels, never the kolam's structure. Every geometric
change also returns its 3×3 pixel transform `H`, so we always know where each gap moved to (`M` below).

See them: `python -m kolam_feed augs data/enc_demo_100k -o data/augs.png` (each effect on its own) and
`python -m kolam_feed preview data/enc_demo_100k --aug real` (random mixes).

### The lattice-to-pixel matrix `M`

`render_kolam(..., frame=True)` returns `M`, a 3×3 matrix taking a lattice point `(lj, li, 1)` to its
pixel `(x, y, 1)`. The dataset composes it with the augmentation: `M = H @ M_render`. So for any
training image we know exactly where every gap sits, even after tilt and rotation. Model B trains on
this.

---

## 3. The shared encoder (the "eye")

Both models start the same way: a CNN turns the image into **feature maps**, grids of vectors, each
describing one patch of the picture.

```
image 384×384×3
  │ ResNet-18 stem (conv 7×7 s2, maxpool, layer1)   → 96×96×64      stride 4
  │ layer2                                          → 48×48×128     stride 8   → 1×1 conv → f8  (48×48×256)
  │ layer3                                          → 24×24×256     stride 16  → 1×1 conv → f16 (24×24×256)
```

- **Convolution:** slides small learned filters over the image. Early layers find edges and curves,
  later ones find crossings, loop ends and horn tips.
- **Stride:** how much the map shrinks. Stride 8 means one vector per 8×8-pixel patch.
- **Residual block:** `out = layers(x) + x`. The skip connection keeps gradients flowing in deep nets.
- **f8** (fine detail) is for reading individual gaps; **f16** (wider view) is for context.

### Pretrained vs from scratch

**Pretrained doesn't change the structure, only the starting weights.**

- `--backbone resnet18` (default): ResNet-18 with **ImageNet** weights, trained on 1.2M real photos.
  It already understands real lighting, texture, blur and noise, which is the hard part of reading
  phone photos. Our synthetic data then only has to teach it kolams. Weights are cached in
  `.cache/torch/` (inside the repo).
- `--backbone small`: our own small ResNet with random starting weights.

When fine-tuning, the pretrained layers get a **10× smaller learning rate** than the new parts, so they
adjust gently instead of forgetting what they know.

Only the encoder is pretrained. The decoder (A) and heads (B) are kolam-specific and start from
scratch: no pretrained model speaks our 11-token digit language.

---

## 4. Model A: SeqModel, "the writer"

It looks at the image and writes the encoding **one token at a time**, like image captioning.

```
f16 ──flatten──► 576 memory vectors (+ 2D sine position code)
                         │ cross-attention
<bos> 0 0 0 | 0 1 1 … ──► TRANSFORMER DECODER (4 layers, d=256, 8 heads) ──► next token
```

**Vocabulary:** 11 tokens: `0`–`8`, `|` (end of row), `<end>`. Plus `<bos>` (start) and `<pad>`
internally. The 3×3 example becomes `0 0 0 | 0 1 1 0 | 1 1 1 | ... <end>`.

### The transformer decoder: same block as an LLM, much smaller

| | GPT-3 | Our decoder |
|---|---|---|
| Layers | 96 | 4 |
| Width d | 12,288 | 256 |
| Heads | 96 | 8 |
| Vocabulary | ~50k | 11 |
| Sees an image | no | yes (cross-attention) |

Each layer, in order (pre-norm: LayerNorm before each sublayer, residual `x + …` after):

1. **Masked self-attention:** each token looks back at the tokens already written.
2. **Cross-attention:** each token looks at the image patches.
3. **Feed-forward MLP:** 256 → 1024 → 256 per token.

#### Embedding and position
A token ID → a learned 256-vector. Transformers see all positions at once and have no built-in order,
so position must be added. GPT adds "I'm token #17". We add **"I'm writing row 3, column 5"**
(`token_slots` in `model.py`: the row is the number of `|` so far, the column is the tokens since the
last `|`). That tells attention exactly which gap is being written, so it knows where to look.

#### Attention, the core maths
Every token makes three vectors from its own vector: **Query** ("what am I looking for?"), **Key**
("what do I contain?"), **Value** ("what I hand over").

```
attention(Q, K, V) = softmax( Q · Kᵀ / √d_head ) · V
```

Q·K scores how well each pair matches. Softmax turns the scores into weights that sum to 1, and the
output is the weighted mix of Values. With **8 heads**, d=256 splits into 8 × 32: eight attentions in
parallel, each free to track something different ("the digit above me", "the start of this row", ...).

- **Self-attention** uses Q, K and V from the tokens. The **causal mask** (`-inf` above the diagonal)
  stops a token seeing future tokens, so it can't peek at the answer. GPT uses exactly this mask.
- **Cross-attention** takes Q from the tokens and K, V from the **image memory**. Writing gap (3, 5),
  the query learns to match the patches where that gap sits and pulls in what's drawn there. This is how
  the model reads the picture. GPT has no cross-attention; the original 2017 Transformer did (it
  attended to the source sentence when translating).

#### Feed-forward
Attention moves information between positions, and the MLP processes it within each position.

#### Glue
- **Residuals** let information skip around each block, so deep stacks still train.
- **LayerNorm** keeps each vector's scale stable.
- **Dropout 0.1** randomly zeroes 10% of values in training, against overfitting.

### Training: teacher forcing
Input = `<bos>` + the true tokens shifted right; target = the true tokens. Thanks to the causal mask,
every position predicts its next token in **one parallel pass**. Loss = **cross-entropy** ("how
surprised was the model by the right token"), averaged over positions, ignoring padding. This is how
GPT is pre-trained.

### Reading a new image: greedy, constrained decoding
Start at `<bos>`; repeat: run the decoder, take the most likely **allowed** token, append it; stop at
`<end>`. `encoding.allowed_next` blocks illegal tokens (wrong row length, a horn facing inward on a
border gap), so the output is always well formed. LLMs use the same trick to force valid JSON.

One full decoder pass per token, so it's slow, and big kolams are ~200+ tokens. A **KV cache** (reuse
earlier tokens' K/V instead of recomputing) is the standard speed-up; we don't have it yet.

**Strengths:** simple, flexible, a well-known recipe. **Weaknesses:** long outputs for big kolams; one
mistake ruins exact match; slow on phones.

---

## 5. Model B: GridModel, "the reader"

It finds the dot grid first, then reads **every gap at once**.

```
f16 ──► transformer encoder (2 layers, no mask) ──► mean ──► rows? (1..R)   cols? (1..C)
f8  ──► 1×1 conv → 4 heatmaps ──► soft-argmax ──► pixel (x, y) of the lattice's 4 corners
           corners + rows/cols ──► homography ──► pixel position of every gap
           sample f8 ⊕ f16 there (grid_sample) + orientation embedding ──► MLP ──► digit 0–8
```

### The pieces
- **Context transformer:** 2 layers of self-attention over the 24×24 patches, **without** a causal mask:
  every patch sees every other patch, so each knows about the whole kolam. (The same block as BERT/ViT.)
- **Size heads:** classifiers for "how many dot rows?" and "how many columns?".
- **Corner heatmaps + soft-argmax:** 4 heatmaps, one per lattice corner. A softmax over all pixels turns
  each into a probability map; its average position (Σ p·x, Σ p·y) is the corner. Unlike a hard argmax
  it's **differentiable**, so the network learns sub-pixel positions. (From pose estimation:
  "Integral Human Pose Regression".)
- **Homography:** a 3×3 matrix mapping the flat lattice onto the tilted photo, solved from 4 point
  pairs (8 equations, 8 unknowns: `homography()` in `model.py`). The same maths as the camera-tilt
  augmentation.
- **grid_sample:** reads a feature map at exact sub-pixel positions (bilinear interpolation). From
  "Spatial Transformer Networks".
- **Orientation embedding:** tells the gap MLP whether it's an up/down or a left/right gap, because the
  same digit (e.g. 3 = horn facing E *or* S) depends on it.

### Training
```
loss = CE(gap digits)  +  CE(rows) + CE(cols)  +  10 · L1(corners / image size)
```
The trick: the gap classifier is trained at the **true** gap positions (from `M`), so it learns to read
gaps even while the corner heads are still learning to find them. When reading a new image it uses the
**predicted** size and corners.

**Strengths:** one pass, so fast (phone-friendly); each gap is a small local decision, which should
scale to big kolams. **Weakness:** if the size or corners are wrong, every gap is read in the wrong place.

---

## 6. Training setup (`kolam_feed/train.py`)

| | |
|---|---|
| Optimiser | AdamW, weight decay 0.05 |
| Learning rate | 3e-4, linear warmup, then cosine decay to 0; pretrained encoder ×0.1 |
| Stability | gradient clipping at 1.0; mixed precision (fp16) on GPU |
| Split | every 100th kolam held out for validation |
| Curriculum | `--max-dots N` trains only on grids up to N×N |
| Checkpoints | `runs/<name>/last.pt` every evaluation; `--resume` continues |
| Log | `runs/<name>/log.jsonl` |

```bash
python -m kolam_feed train data/enc_demo_100k --model grid --out runs/grid --max-dots 5 --size 192
python -m kolam_feed eval  runs/grid/last.pt data/enc_demo_100k --aug real
python -m kolam_feed eval  runs/grid/last.pt data/enc_demo_100k --min-dots 6   # bigger than trained on
```

---

## 7. Experiments

### Exp 0: smoke test (laptop CPU, 2026-10-18)

30 steps each, 192px, grids ≤ 5×5, batch 16. Purpose: does everything run end to end?

| | B: grid | A: seq |
|---|---|---|
| Parameters | 4.6M | 7.1M |
| Training speed | ~11 img/s | ~11 img/s |
| **Time to read one image (CPU)** | **53 ms** | **408 ms** |
| Exact after 30 steps | 0% | 0% |

Both run. Neither has learned anything yet (both guess the most common answer), as expected after 30
steps. Training speed is limited by rendering images on the CPU, not by the model. Reading a new image
is ~8× faster with B, because A runs one full decoder pass per token.

Fix after this: B's corner loss started at ~14 and drowned out the gap loss; it was rescaled to
`10 · L1` on normalised coordinates.

### Exp 1: pilot A vs B (laptop CPU, 2026-10-18)

**Question:** do both learn, and which learns faster on the same budget?

**Setup:** same for both: `data/enc_demo_100k`, grids ≤ 5×5 (21,625 train / 228 val kolams), 192px,
pretrained ResNet-18, batch 16, 2,000 steps (32k images), warmup 100, `light` augmentation, scored every
500 steps on 128 held-out kolams. Script: `runs/pilot.sh`; log: `runs/pilot.log`.

**Decision rule (set before looking):** prefer B if it's within ~2 points of A on exact match, or
better, because it's faster on phones. Pick A only if it clearly wins.

**Results** (128 held-out kolams, `light` augmentation, same as training):

| step | B exact | B digits | A exact | A digits |
|---|---|---|---|---|
| 500 | 18.0% | 87.4% | 19.5% | 92.2% |
| 1000 | 29.7% | 91.1% | 50.0% | 96.2% |
| 1500 | 34.4% | 91.8% | 66.4% | 98.0% |
| 2000 | **33.6%** | 91.9% | **69.5%** | **98.3%** |

| | B: grid | A: seq |
|---|---|---|
| Grid size right (sized) | 100% | 98.4% |
| Valid answers | 100% | 100% |
| Time to read one image (CPU, 192px) | **~25 ms** | ~265 ms |
| Training time (2,000 steps) | 47 min | 43 min |
| Curve at the end | flat since step 1500 | still climbing slowly |

**A wins on accuracy, clearly:** 69.5% vs 33.6% exact. B is ~10× faster at reading an image.

**Why B lost (diagnosis):** B's gap reader, given the **true** gap positions, reads **98.2%** of gaps
right, the same as A. Its grid size is 100% right. What fails is **finding the corners**: median error
6.1 px, worst 10% over 20 px on a 192 px image, where neighbouring gaps are only ~15 px apart. So B reads
the right digits from slightly wrong places. B's idea works; its corner finder is the weak part, and
that's fixable:
- predict every dot's position (a dense heatmap of dots) instead of 4 corners, and fit the grid to them
- train the gap reader on slightly jittered positions, so small placement errors don't hurt
- a second refinement pass: re-predict corners from features sampled around the first guess

**Phone-like images** (`--aug real`, never seen in training):

| | B: grid | A: seq |
|---|---|---|
| exact | 0% | 5.5% |
| digits | 62.5% | 76.7% |
| sized | 52.3% | 48.4% |

Both collapse: they were trained on `light` only. Expected, and the reason training must include `real`
(and why real photos are the true test).

**Verdict:** by the decision rule, **A wins**. A is the model to scale up next. B stays worth a second
try with a better grid finder, because it's ~10× faster on phones and its gap reading is already as good
as A's.

**Limits:** tiny compared to the real task: small grids, small images, a short run. A usually needs
more data than B, so it may look worse here than it would at full scale; check whether its curve is
still climbing. This can't answer "big kolams" or "real phone photos"; that's Exp 2 on the gaming PC.

### Next

- **Exp 2 (gaming PC):** bigger grids (up to 15×15+), 384–512px, long runs, A vs B for real.
- **"Not a kolam":** `<none>` token (A) / yes-no head (B), trained with ~20% negatives (COCO photos plus
  rendered hard negatives: empty floors, dots only, scribbles, broken kolams).
- **Phone framing:** one shared `prepare()` (fix rotation, pad to square, resize) used in training and
  when reading new photos; kolam 30–90% of a 4:3 / 16:9 frame; crop-and-zoom second pass for big kolams.
- **Real test set:** ~100 phone photos labelled in `kolam_studio`. The true score.
- **Phone encoder:** MobileNetV3 / EfficientNet-Lite once the model works.

---

## 8. Further reading, in order

1. *The Illustrated Transformer*, Jay Alammar: pictures of Q/K/V and multi-head attention.
2. Andrej Karpathy, *Let's build GPT: from scratch, in code* (YouTube): our decoder minus cross-attention.
3. *Attention Is All You Need* (2017), §3.1–3.3: the original encoder-decoder with cross-attention.
4. *Deep Residual Learning* (ResNet, He et al. 2015) and the CS231n notes: CNNs, residual blocks.
5. *Pix2Seq* (Chen et al. 2021): image → token sequence, the closest relative of Model A.
6. *Spatial Transformer Networks* (2015): grid_sample and learned warps (Model B).
7. *Integral Human Pose Regression* (2018): soft-argmax heatmaps (Model B's corners).
8. *Domain randomization* (Tobin et al. 2017): why heavy synthetic variety transfers to real photos.
