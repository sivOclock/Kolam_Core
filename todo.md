> kolam data generation pipeline
> train the model
> VLLm Baselineing
> UI to show

# Plan: train model A (seq), ship it in the browser

Goal: a model that reads a photo of a real-life kolam, **up to 12 x 12 dots for now** (20 x 20 later),
writes its encoding (spec §12), says when a picture is not a kolam at all, and runs **in the browser**
(no server).

## Decisions (and why)

**Model A (seq), settled 2026-10-25.** A plain sequence decoder (`SeqModel`): the encoder's feature map,
then a 4-layer transformer decoder writes the encoding digit by digit, constrained by `allowed_next`.
Why A over B (grid): A won the only fair test (69.5% vs 33.6% exact up to 5 x 5, same data and steps);
it needs no corner guesses; each digit sees the ones before it, so answers fit together. B night 1
(`runs/night_BM`, 12 h, 1.44M images, 3 to 20 dots) ended at 3.7% exact, 78% of gaps right, and kept only
32% of real kolam photos. Reasoning laid out in `explainer/index.html`.

A's cost is length: a 20 x 20 is 882 tokens (10.4 s on the GPU, 3.95 of 4 GB while training), so the
cap is 12 x 12 (312 tokens) for now. 20 x 20 later by reading big kolams in overlapping tiles, not by
longer sequences.

Measured head-to-head on the GPU laptop (3x3 to 20x20, ResNet-18, 384 px, batch 24):

| | A, seq (writes digit by digit) | B, grid (reads every gap at once) |
|---|---|---|
| training speed | 18.8 images/s | 44 images/s |
| GPU memory | 3.95 of 4 GB (no headroom) | 2.15 GB |
| reading one kolam (GPU) | 10,400 ms | 41 ms |

**Size: A-M**, ResNet-34 encoder + 4-layer decoder. Not bigger on a 4 GB GPU.

**Data: 3-5M training images over 2-3 nights.** 2M unique kolams (kolam_gen, ~30 min), each drawn
fresh every time it is used (angle, floor, ink, glare). ~1.5M images a night. Stop when
the held-out checks stop improving for a night, especially the 9-11 and 12 bands.

**Not a kolam.** One yes/no head on the shared encoder (no change to the encoding). 15% of every
batch: Imagenette photos (13,394) plus drawn hard cases (empty floors, dot grids with no lines,
scribbles). Those photos are only ever non-kolam examples, never a kolam's background (kolams are drawn
on real-looking floors: cement, red oxide, tiles, granite, sand). The first try learned "looks like a
photo = not a kolam" (it rejected the real coaster photo, 0.002); kolams in the `real` photo look fixed
it, and it is checked at
every eval on 31 real line-kolam photos (`data/real_kolams_line`: 30 from Wikimedia Commons, licences
in its `SOURCES.json`, plus our coaster; local only, not committed). Rangoli: left out for now.

**One picture size (384 px), read twice.** Pass 1 finds the kolam in the whole photo; pass
2 crops the kolam out of the full-resolution photo and reads it again, so a far-away or 4000 px photo
still gives each gap enough pixels. Train with a wider zoom range (kolam 40-100% of the frame) so
pass 1 finds small kolams. 448-512 px only if the 18-20 band stays weak.

## Steps

- [x] Browser reader: ONNX export (`python -m kolam_feed export`), onnxruntime-web, static `dist/`
      (cPanel ok, `.htaccess` for .wasm/.onnx). Seq model only so far.
- [x] Live training page: `python -m kolam_feed.watch runs/<x>` (loss, accuracy by size band,
      not-kolam checks, GPU/CPU/RAM, replay of what it reads)
- [x] Not-a-kolam head, non-kolam data, real-photo check
- [x] GPU laptop set up: everything in `~/kolam_core_model` (WSL), `scripts/gpu/launch.sh` + `jobs.sh`
- [x] 400k kolams up to 20 x 20 generated there (5.7 min)
- [x] B-M: `--backbone resnet34 --ctx-layers 4` = 11.6M params; probe: 44 images/s, 2.9 of 4 GB
- [x] Wider zoom range: `real` puts a quarter of the kolams at 40-90% of the frame (`p_far`)
- [x] Generate 2M unique kolams (3 to 20 dots; `--max-dots 12` keeps the ones up to 12 x 12)
- [x] Night 1: B-M (grid) on 1.44M images -> 3.7% exact; B dropped for A (see Decisions)
- [x] B exported to the browser anyway (`export` takes grid checkpoints; border gaps constrained)
- [x] `kolam_gen render`: pictures + labels drawn ahead onto disk (Rust; effects on the CPU or, with
      `--features gpu` / `--gpu`, in CUDA — identical pictures). GPU laptop: ~750 pictures/s (Python ~60),
      1.44M in ~32 min, ~25 KB each (~36 GB). `train --rendered DIR` trains from it; checks still draw live.
- [x] Not-a-kolam head refit on night 2's `best.pt` (only its last Linear, on cached features: real kolam
      photos as positives) -> `best_head.pt`: keeps 74% of held-out real kolams (was 29%), lets 15% of
      random photos through
- [x] 2M pictures rendered on the laptop (`data/img_2m_20`, 3 to 20 dots, 88 GB, 61 min); milder look in
      Rust and Python; non-kolams: photos 25% and only lightly touched, drawn ones get the kolams' look
- [x] Night 2 (started 2026-10-25 10:28): `night_AM2`, seq A-M up to 12 x 12 from the rendered pictures,
      sized for 20 x 20, size-bucketed batches (~83 pictures/s). 100k steps, 10 h 25 min. On 100 new kolams
      `best.pt` (step 72k) 64% exact (3-5: 38%, 6-8: 91%, 9-11: 33%), `last.pt` 36%; in the site as `night_am2`
- [ ] Night 3: continue from `last.pt` with `--max-dots 20 --batch 16`
- [ ] Nights 3+: continue from `last.pt` while the checks improve
- [ ] KV cache in the decoder (Python predict and the ONNX export): each digit costs one step, not the
      whole sequence again; needed for 12 x 12 in the browser
- [ ] Label 20-30 real photos in the studio: the first real-photo accuracy number
- [ ] Rebuild the site, upload `dist/` to cPanel

Later: fine-tune on real kolam photos with labels (draw them in the studio); diamond (1-3-5-7) dot
layouts; rangoli as its own class; a large teacher model distilled into the browser model.

## Running on the GPU laptop

Before a night run: laptop plugged in, lid open, Windows Update paused. `scripts/gpu/jobs.sh` lists
the runs (one per line); the Windows scheduled task `kolam_core_model` runs `launch.sh`, which keeps
WSL up, opens the live page and asks Windows not to sleep until the jobs end. Runs resume from
`last.pt`; `best.pt` is the best held-out exact. Delete the scheduled task when the laptop is done.
