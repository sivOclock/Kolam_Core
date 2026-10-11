# Dataset report — `data/kolams_v2`

**3000 samples**

## Correctness (independent re-check of every sample)

| check | failures |
|---|---|
| structure_invalid | 0 |
| not_canonical | 0 |
| program_mismatch | 0 |
| compile_not_deterministic | 0 |
| geometry | 0 |
| decode_mismatch | 0 |
| render_not_deterministic | 0 |
| stroke_count_theorem | 0 |

- re-rendered byte-identical: 200 images checked
- different labels with identical pictures: **0**
- stroke_count_theorem: tree-shaped brush lines give one stroke per tree, checked on 2208 kolams
- min dot clearance 0.466 / min strand clearance 0.2 (lattice units; dots are 2 apart)

## Diversity

- unique labels: 3000 / 3000 (duplicates: 0)
- isomorphism classes (§6, translation etc. merged): 2887 / 3000
- means: {'relations': 12.72, 'strokes': 5.34, 'crossings': 12.38, 'coverage': 0.95}
- fractions: {'one_stroke': 0.278, 'full_coverage': 0.761, 'all_three_brushes': 0.604, 'single_brush': 0.167, 'kolams_with_shared_dots': 0.524}
- relations: {'CROSSING_LOOP': 12574, 'PARALLEL_LOOP': 11283, 'HORN': 14293}, closed rings: 1000

- **grid**: 1x1: 13, 1x2: 14, 1x3: 31, 1x4: 13, 1x5: 28, 1x6: 8, 1x7: 11, 1x8: 4, 2x1: 7, 2x2: 3, 2x3: 5, 2x4: 5, 2x6: 1, 3x1: 34, 3x2: 2, 3x3: 220, 3x4: 26, 3x5: 44, 3x6: 42, 3x7: 33, 3x8: 46, 4x1: 22, 4x2: 3, 4x3: 47, 4x4: 293, 4x5: 42, 4x6: 41, 4x7: 31, 4x8: 48, 5x1: 28, 5x2: 2, 5x3: 32, 5x4: 37, 5x5: 261, 5x6: 39, 5x7: 35, 5x8: 32, 6x1: 11, 6x2: 4, 6x3: 36, 6x4: 46, 6x5: 48, 6x6: 290, 6x7: 43, 6x8: 44, 7x1: 9, 7x2: 1, 7x3: 39, 7x4: 39, 7x5: 52, 7x6: 40, 7x7: 244, 7x8: 29, 8x1: 12, 8x2: 1, 8x3: 29, 8x4: 40, 8x5: 34, 8x6: 38, 8x7: 32, 8x8: 256
- **relations**: 1: 70, 10: 150, 11: 147, 12: 158, 13: 118, 14: 128, 15: 84, 16: 112, 17: 63, 18: 72, 19: 54, 2: 71, 20: 95, 21: 61, 22: 53, 23: 37, 24: 42, 25: 23, 26: 36, 27: 17, 28: 43, 29: 15, 3: 98, 30: 26, 31: 15, 32: 25, 33: 7, 34: 12, 35: 8, 36: 20, 37: 7, 38: 6, 4: 158, 40: 11, 42: 4, 44: 8, 45: 2, 46: 3, 47: 1, 48: 1, 49: 1, 5: 170, 50: 6, 52: 1, 6: 172, 7: 212, 8: 201, 9: 176
- **strokes**: 1: 835, 10: 132, 11: 68, 12: 103, 13: 51, 14: 60, 15: 20, 16: 55, 17: 9, 18: 16, 19: 3, 2: 250, 20: 18, 22: 6, 24: 4, 28: 2, 3: 169, 32: 1, 4: 388, 5: 181, 6: 199, 7: 143, 8: 207, 9: 80
- **symmetry**: c4: 174, d2: 667, d4: 157, mirror_lr: 470, mirror_ud: 294, none: 769, rot180: 469
- **loop_len**: 10: 176, 11: 21, 12: 65, 13: 1, 14: 24, 15: 4, 16: 19, 18: 2, 2: 7727, 20: 2, 21: 3, 24: 5, 25: 4, 28: 2, 3: 5473, 30: 3, 32: 3, 35: 4, 36: 4, 4: 3637, 40: 2, 42: 3, 48: 4, 49: 3, 5: 2271, 56: 1, 6: 1978, 64: 4, 7: 1119, 8: 1083, 9: 210
- **horn_variant**: LEFT_HORN: 4509, PLAIN_HORN: 5131, RIGHT_HORN: 4653
- **brush_dots**: CROSSING_LOOP: 49153, HORN: 14293, PARALLEL_LOOP: 44104
- **style**: dark bg: 2120, light bg: 880
