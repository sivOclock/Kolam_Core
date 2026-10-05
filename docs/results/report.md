# Dataset report — `data/kolams_v1`

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
- stroke_count_theorem: tree-shaped brush lines give one stroke per tree, checked on 2350 kolams
- min dot clearance 0.443 / min strand clearance 0.226 (lattice units; dots are 2 apart)

## Diversity

- unique labels: 3000 / 3000 (duplicates: 0)
- isomorphism classes (§6, translation etc. merged): 2841 / 3000
- means: {'relations': 11.99, 'strokes': 5.19, 'crossings': 10.66, 'coverage': 0.95}
- fractions: {'one_stroke': 0.293, 'full_coverage': 0.663, 'all_three_brushes': 0.547, 'single_brush': 0.184, 'kolams_with_shared_dots': 0.495}
- relations: {'CROSSING_LOOP': 10304, 'PARALLEL_LOOP': 9288, 'HORN': 16369}, closed rings: 800

- **grid**: 2x3: 1, 2x4: 3, 3x2: 3, 3x3: 213, 3x4: 50, 3x5: 59, 3x6: 53, 3x7: 53, 3x8: 48, 4x2: 1, 4x3: 62, 4x4: 273, 4x5: 42, 4x6: 52, 4x7: 36, 4x8: 51, 5x3: 56, 5x4: 36, 5x5: 275, 5x6: 42, 5x7: 42, 5x8: 34, 6x3: 54, 6x4: 52, 6x5: 44, 6x6: 307, 6x7: 43, 6x8: 44, 7x3: 49, 7x4: 40, 7x5: 47, 7x6: 43, 7x7: 272, 7x8: 32, 8x3: 44, 8x4: 43, 8x5: 33, 8x6: 44, 8x7: 31, 8x8: 293
- **relations**: 1: 68, 10: 182, 11: 185, 12: 186, 13: 142, 14: 121, 15: 89, 16: 129, 17: 62, 18: 75, 19: 53, 2: 57, 20: 84, 21: 49, 22: 49, 23: 27, 24: 56, 25: 27, 26: 27, 27: 11, 28: 48, 29: 14, 3: 89, 30: 17, 31: 7, 32: 14, 34: 6, 35: 3, 36: 17, 37: 1, 38: 3, 4: 159, 40: 3, 41: 2, 42: 2, 44: 1, 45: 1, 46: 1, 49: 1, 5: 169, 6: 208, 7: 162, 8: 218, 9: 175
- **strokes**: 1: 878, 10: 132, 11: 56, 12: 108, 13: 37, 14: 45, 15: 21, 16: 57, 17: 9, 18: 17, 19: 4, 2: 242, 20: 15, 21: 2, 22: 3, 24: 3, 3: 169, 4: 379, 5: 163, 6: 228, 7: 122, 8: 231, 9: 79
- **symmetry**: c4: 157, d2: 641, d4: 142, mirror_lr: 478, mirror_ud: 309, none: 811, rot180: 462
- **loop_len**: 10: 178, 11: 19, 12: 57, 14: 17, 15: 5, 16: 18, 18: 2, 2: 5959, 20: 2, 21: 2, 24: 5, 25: 4, 28: 2, 3: 4360, 30: 4, 32: 3, 35: 3, 36: 4, 4: 3020, 40: 2, 42: 3, 48: 3, 49: 3, 5: 1967, 6: 1643, 64: 4, 7: 1119, 8: 971, 9: 213
- **horn_variant**: LEFT_HORN: 5023, PLAIN_HORN: 6132, RIGHT_HORN: 5214
- **brush_dots**: CROSSING_LOOP: 41819, HORN: 32738, PARALLEL_LOOP: 37356
- **style**: dark bg: 2134, light bg: 866

## Pixel probe (tiny MLP on 9x9 local patches, grid position given)

- trained on 1050 images, tested on 450
- gap state accuracy: 0.99872 over 33560 gaps
- centre (curled tip) accuracy: 0.99995 over 19722 centres
- **whole kolam decoded exactly: 0.9289** (by style: {'blur/noise': 0.9333, 'clean': 0.92})
- top confusions: [('SPLIT->TIP1', 22), ('SPLIT->TIP0', 7), ('CROSS->TIP0', 5), ('CROSS->JOIN', 3), ('TIP1->CROSS', 2)]
