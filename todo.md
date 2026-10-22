> kolam data generation pipeline
> train the model
> VLLm Baselineing
> UI to show

# Plan: train model B, ship it in the browser

Goal: a model that reads a photo of a real-life kolam, **up to 20 x 20 dots**, writes its encoding
(spec §12), says when a picture is not a kolam at all, and runs **in the browser** (no server).

## Decisions (and why)

**Model B (grid), not A (seq).** Head-to-head on the GPU laptop: same data (3x3 to 20x20), same
ResNet-18 encoder, 384 px, batch 24, `real` augmentation, 20-step probes:

| | A, seq (writes digit by digit) | B, grid (reads every gap at once) |
|---|---|---|
| training speed | 18.8 images/s | 44 images/s |
| GPU memory | 3.95 of 4 GB (no headroom) | 2.15 GB |
| reading one kolam (GPU) | 10,400 ms | 41 ms |

A's answer for a 20 x 20 kolam is 903 digits written one after another: slow, memory-bound, and one
early slip carries on. B reads each gap at its own spot, so a 20 x 20 kolam is just more gaps. A won
on small kolams in the CPU pilot (69.5% vs 33.6% exact up to 5 x 5, both undertrained); the bake-off
was stopped before an accuracy verdict and B was chosen on speed, memory and scaling.

**Size: B-M, about 12M parameters** (ResNet-34 encoder 8.2M + 4-layer context transformer 3.2M).
The laptop is CPU-bound drawing kolams (~48 images/s, GPU ~60% busy), so a model this size trains at
about the same speed. The encoder reads gaps; the context transformer counts the grid (B's weak spot).
Not 50M: ~3x slower to train on a 4 GB GPU, ~200 MB in the browser, and undertrained on this compute.
Bigger only if the loss flattens while still high.

**Data: 3-5M training images over 2-3 nights.** 2M unique kolams (kolam_gen, ~30 min), each drawn
fresh every time it is used (angle, floor, ink, glare). ~1.5M images a night. Stop when
the held-out checks stop improving for a night, especially the 15-17 and 18-20 bands.

**Not a kolam.** One yes/no head on the shared encoder (no change to the encoding). 15% of every
batch: Imagenette photos (13,394) plus drawn hard cases (empty floors, dot grids with no lines,
scribbles). Those photos are only ever non-kolam examples, never a kolam's background (kolams are drawn
on real-looking floors: cement, red oxide, tiles, granite, sand). The first try learned "looks like a
photo = not a kolam" (it rejected the real coaster photo, 0.002); kolams in the `real` photo look fixed
it, and it is checked at
every eval on 31 real line-kolam photos (`data/real_kolams_line`: 30 from Wikimedia Commons, licences
in its `SOURCES.json`, plus our coaster; local only, not committed). Rangoli: left out for now.

**One picture size (384 px), read twice.** Pass 1 finds the grid's corners in the whole photo; pass
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
- [ ] Generate 2M unique kolams
- [ ] Night 1: B-M on ~1.5M images; copy `best.pt` back; look at the bands
- [ ] Nights 2-3: continue from `last.pt` while the checks improve
- [ ] Export B to ONNX (grid model: size, corners, homography, gaps) + two-pass zoom-in in `reader.ts`;
      try int8; time it on a laptop and a phone
- [ ] Constrained gaps for B (border gaps only walls or outward horns), like A's `allowed_next`
- [ ] Rebuild the site, upload `dist/` to cPanel

Later: fine-tune on real kolam photos with labels (draw them in the studio); diamond (1-3-5-7) dot
layouts; rangoli as its own class; a large teacher model distilled into the browser model.

## Running on the GPU laptop

Before a night run: laptop plugged in, lid open, Windows Update paused. `scripts/gpu/jobs.sh` lists
the runs (one per line); the Windows scheduled task `kolam_core_model` runs `launch.sh`, which keeps
WSL up, opens the live page and asks Windows not to sleep until the jobs end. Runs resume from
`last.pt`; `best.pt` is the best held-out exact. Delete the scheduled task when the laptop is done.
