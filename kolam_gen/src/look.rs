//! The look of one training picture: kolam_feed/render.py random_style plus augment.py's "real" preset,
//! toned down (2026-10-25) so a person can still read every kolam
//! (turn, camera angle, floor, ink, chalk, petals, light, glare, blur, noise, JPEG). Every random choice
//! is made here, up front, from the kolam's own Rng, and written into a flat parameter row (`P_*`), so
//! the pixel work (fx.rs on the CPU, fx.cu on the GPU) is a pure function of (mask, row) and both read
//! the same layout. Same kinds of draws as the Python augmentation, not the same random stream.

use crate::model::{horn_target, is_reflection, lat_op, transform_vertex, Kolam, Op, DIRS, SPLIT};
use crate::rng::Rng;

macro_rules! params {
    ($($name:ident),* $(,)?) => {
        #[allow(non_camel_case_types, dead_code, clippy::upper_case_acronyms)]
        #[repr(usize)]
        enum Idx { $($name),*, Count }
        $(#[allow(dead_code)] pub const $name: usize = Idx::$name as usize;)*
        /// Floats per picture in the parameter row.
        pub const NP: usize = Idx::Count as usize;
        /// `#define P_NAME index` for fx.cu: one layout for both sides.
        #[cfg_attr(not(feature = "gpu"), allow(dead_code))]
        pub fn cuda_defines() -> String {
            let mut s = String::new();
            $(s += &format!("#define {} {}\n", stringify!($name), $name);)*
            s + &format!("#define NP {}\n#define MAX_DEBRIS {}\n", NP, MAX_DEBRIS)
        }
    };
}

params!(
    P_SEED,                                                       // u32 bits: the hash noise of this picture
    P_HI0, P_HI1, P_HI2, P_HI3, P_HI4, P_HI5, P_HI6, P_HI7, P_HI8, // new pixel -> drawn pixel (inverse of H)
    P_BG_R, P_BG_G, P_BG_B, P_INK_R, P_INK_G, P_INK_B,            // floor when no surface; the line colour
    P_MASK_BLUR, P_THICK_R,                                       // line softness (sigma, px); thicker line (radius)
    P_CHALK, P_CHALK_CELLS, P_CHALK_T,                            // broken, powdery lines
    P_SURF, P_FL_R, P_FL_G, P_FL_B, P_FL2_R, P_FL2_G, P_FL2_B, P_SA, P_SB, P_SC, P_SD, // the floor
    P_MULTI, P_MC0, P_MC1, P_MC2, P_MC3, P_MC4, P_MC5, P_MC6, P_MC7, P_MC8, P_MC9, P_MC10, P_MC11,
    P_MCELLS0, P_MCELLS1, P_MCELLS2, P_MCELLS3,                   // several ink colours across the kolam
    P_POWDER, P_DEBRIS, P_TEX_CELLS,
    P_LIGHT, P_LIGHT_T,
    P_SHADOW, P_SHADOW_T, P_SHADOW_X, P_SHADOW_Y, P_SHADOW_SOFT,
    P_GLARE, P_GLARE_X, P_GLARE_Y, P_GLARE_A, P_GLARE_B, P_GLARE_T, P_GLARE_S,
    P_VIG, P_VIG_X, P_VIG_Y,
    P_COLOUR, P_COL_MEAN, P_COL_C, P_COL_B, P_COL_G0, P_COL_G1, P_COL_G2, P_COL_GAMMA,
    P_GRAY, P_MOTION_N, P_MOTION_T, P_NOISE, P_NOISE_MONO, P_DEFOCUS, P_LOWRES,
);

/// Petals and dirt per picture, at most; each (cx, cy, rx, ry, r, g, b).
pub const MAX_DEBRIS: usize = 30;
pub const DEBRIS_W: usize = 7;

/// augment.OPS, in the same order (`turn` in the labels).
pub const OP_NAMES: [&str; 8] = ["id", "rot90", "rot180", "rot270", "mirror_lr", "mirror_ud", "transpose", "antitranspose"];
const OPS: [Op; 8] = [Op::Id, Op::Rot90, Op::Rot180, Op::Rot270, Op::MirLR, Op::MirUD, Op::Tr, Op::Anti];

/// augment.SURFACES, in order (P_SURF).
pub const SURFACES: [&str; 10] = ["cement", "red_oxide", "mud", "sand", "tiles", "granite", "wood", "paper", "chalkboard", "asphalt"];
const INKS: [(f64, f64, f64); 10] = [
    (240., 236., 222.), (232., 232., 228.), (238., 186., 30.), (200., 30., 45.), (40., 160., 70.),
    (40., 90., 200.), (230., 80., 150.), (240., 120., 30.), (120., 50., 160.), (25., 22., 30.),
]; // flour, chalk, turmeric, kumkum, green, blue, pink, orange, purple, black

pub type M3 = [[f64; 3]; 3];

/// How the kolam itself is drawn (render.Style), before any image change.
pub struct Style {
    pub line_width: f64, // lattice units (dots are 2 apart)
    pub dot_radius: f64,
    pub margin: f64,
    pub show_dots: bool,
    pub jitter: f64, // hand wobble, lattice units
}

pub struct Look {
    pub op: usize, // index into OP_NAMES
    pub style: Style,
    pub h: M3,      // drawn pixel -> picture pixel (augment.image's H)
    pub thick: bool, // decided once the line coverage is known (finish)
    pub p: [f32; NP],
    pub debris: [[f32; DEBRIS_W]; MAX_DEBRIS],
    pub jpeg_quality: u8,
    floor_lum: f64,
    ink_lum: f64,
}

pub fn normal(r: &mut Rng) -> f64 {
    let (u, v) = (1.0 - r.random(), r.random());
    (-2.0 * u.ln()).sqrt() * (std::f64::consts::TAU * v).cos()
}

fn lum(c: [f64; 3]) -> f64 {
    0.3 * c[0] + 0.59 * c[1] + 0.11 * c[2]
}

fn ints(r: &mut Rng, lo: [i32; 3], hi: [i32; 3]) -> [f64; 3] {
    [0, 1, 2].map(|i| r.integers(lo[i], hi[i]) as f64)
}

fn tint(r: &mut Rng, base: [f64; 3], spread: f64) -> [f64; 3] {
    base.map(|v| v + r.uniform(-spread, spread))
}

/// augment._ink_colour: a real kolam ink that stands out from the floor.
fn ink_colour(r: &mut Rng, floor_lum: f64, coloured: bool) -> [f64; 3] {
    let x = if coloured { 0.0 } else { 1.0 };
    let p = [3.0 * x, x, 1., 1., 1., 1., 1., 1., 0.5, x];
    for _ in 0..8 {
        let (a, b, c) = INKS[r.choice_p(&p)];
        let col = [a, b, c].map(|v| (v * r.uniform(0.9, 1.05)).clamp(0.0, 255.0));
        if (lum(col) - floor_lum).abs() > 100.0 {
            return col;
        }
    }
    if floor_lum < 128.0 { [245., 240., 228.] } else { [25., 22., 30.] }
}

// ------------------------------------------------------------------ turns (augment.turn, label turned too)
/// The kolam under one of the 8 turns/mirrors of its dot grid: gaps move with lat_op, horns keep
/// pointing at the same neighbour, and a mirror swaps LEFT_HORN and RIGHT_HORN.
pub fn turn(k: &Kolam, op: Op) -> Kolam {
    let swap = matches!(op, Op::Rot90 | Op::Rot270 | Op::Tr | Op::Anti);
    let (rows, cols) = if swap { (k.cols, k.rows) } else { (k.rows, k.cols) };
    let lw = (2 * cols + 1) as usize;
    let mut gates = vec![SPLIT; ((2 * rows + 1) * (2 * cols + 1)) as usize];
    for i in 0..=2 * k.rows {
        for j in 0..=2 * k.cols {
            let (a, b) = lat_op((i, j), op, k.rows, k.cols);
            gates[a as usize * lw + b as usize] = k.gate((i, j));
        }
    }
    let horns = k
        .horns
        .iter()
        .map(|&(v, d, var)| {
            let v2 = transform_vertex(v, op, k.rows, k.cols);
            let t = transform_vertex(horn_target(v, d), op, k.rows, k.cols);
            let dir = DIRS.iter().position(|&x| x == (t.0 - v2.0, t.1 - v2.1)).unwrap() as u8;
            (v2, dir, if is_reflection(op) && var != 0 { 3 - var } else { var })
        })
        .collect();
    Kolam::from_gates(rows, cols, gates, horns)
}

// ------------------------------------------------------------------ camera (augment._geom, _tilt)
pub fn mul(a: &M3, b: &M3) -> M3 {
    let mut c = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            c[i][j] = (0..3).map(|k| a[i][k] * b[k][j]).sum();
        }
    }
    c
}

pub fn apply(h: &M3, p: (f64, f64)) -> (f64, f64) {
    let w = h[2][0] * p.0 + h[2][1] * p.1 + h[2][2];
    ((h[0][0] * p.0 + h[0][1] * p.1 + h[0][2]) / w, (h[1][0] * p.0 + h[1][1] * p.1 + h[1][2]) / w)
}

fn inverse(m: &M3) -> M3 {
    let c = |i: usize, j: usize| m[(i + 1) % 3][(j + 1) % 3] * m[(i + 2) % 3][(j + 2) % 3] - m[(i + 1) % 3][(j + 2) % 3] * m[(i + 2) % 3][(j + 1) % 3];
    let det: f64 = (0..3).map(|j| m[0][j] * c(0, j)).sum();
    let mut r = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            r[i][j] = c(j, i) / det;
        }
    }
    r
}

/// The 3x3 H with dst ~ H src, from 4 point pairs (augment._homography).
pub fn homography(src: &[(f64, f64); 4], dst: &[(f64, f64); 4]) -> M3 {
    let mut a = [[0.0f64; 9]; 8];
    for k in 0..4 {
        let ((x, y), (u, v)) = (src[k], dst[k]);
        a[2 * k] = [x, y, 1., 0., 0., 0., -u * x, -u * y, u];
        a[2 * k + 1] = [0., 0., 0., x, y, 1., -v * x, -v * y, v];
    }
    for c in 0..8 {
        let p = (c..8).max_by(|&i, &j| a[i][c].abs().total_cmp(&a[j][c].abs())).unwrap();
        a.swap(c, p);
        for r in 0..8 {
            if r != c {
                let f = a[r][c] / a[c][c];
                for k in c..9 {
                    a[r][k] -= f * a[c][k];
                }
            }
        }
    }
    let h: Vec<f64> = (0..8).map(|i| a[i][8] / a[i][i]).collect();
    [[h[0], h[1], h[2]], [h[3], h[4], h[5]], [h[6], h[7], 1.0]]
}

/// A camera looking at the floor from an angle: the far side shrinks, the picture foreshortens.
fn tilt(p: [(f64, f64); 4], s: f64, r: &mut Rng, amount: f64) -> [(f64, f64); 4] {
    let f = r.uniform(0.0, amount);
    let side = r.integers(0, 4);
    let c = s / 2.0;
    // turn so the far side is on top: side 1 -> (y, -x), side 2 -> (-x, -y), side 3 -> (-y, x)
    let fwd = |(x, y): (f64, f64)| match side { 0 => (x, y), 1 => (y, -x), 2 => (-x, -y), _ => (-y, x) };
    let back = |(x, y): (f64, f64)| match side { 0 => (x, y), 1 => (-y, x), 2 => (-x, -y), _ => (y, -x) };
    p.map(|(x, y)| {
        let (qx, qy) = fwd((x - c, y - c));
        let t = (c - qy) / (2.0 * c);
        let (bx, by) = back((qx * (1.0 - f * t), qy * (1.0 - 0.5 * f) - 0.25 * f * c));
        (bx + c, by + c)
    })
}

/// augment._geom with the "real" preset: rotation up to 30 degrees (shrunk to fit), scale (a quarter
/// shot from further away), shift, camera tilt and a perspective nudge.
fn camera(r: &mut Rng, s: f64) -> M3 {
    let (rotate, tilt_amt, persp, shift) = (15.0f64, 0.2, 0.03, 0.05);
    let c = s / 2.0;
    let t = r.uniform(-rotate, rotate).to_radians();
    let mut k = if r.random() < 0.2 { r.uniform(0.6, 0.9) } else { r.uniform(0.92, 1.1) };
    k /= t.cos().abs() + t.sin().abs();
    let (dx, dy) = (r.uniform(-shift, shift) * s, r.uniform(-shift, shift) * s);
    let a = [[k * t.cos(), -k * t.sin(), 0.], [k * t.sin(), k * t.cos(), 0.], [0., 0., 1.]];
    let tr = mul(&mul(&[[1., 0., c + dx], [0., 1., c + dy], [0., 0., 1.]], &a), &[[1., 0., -c], [0., 1., -c], [0., 0., 1.]]);
    let src = [(0.0, 0.0), (s, 0.0), (s, s), (0.0, s)];
    let mut dst = tilt(src, s, r, tilt_amt).map(|p| apply(&tr, p));
    for d in dst.iter_mut() {
        d.0 += r.uniform(-persp, persp) * s;
        d.1 += r.uniform(-persp, persp) * s;
    }
    homography(&src, &dst)
}

// ------------------------------------------------------------------ the whole look
impl Look {
    /// Every random choice for one picture of size s (the "real" preset; render.random_style).
    pub fn new(r: &mut Rng, s: f64) -> Look {
        let mut p = [0f32; NP];
        p[P_SEED] = f32::from_bits(r.next_u64() as u32);
        let op = if r.random() < 0.5 { r.below(8) } else { 0 };

        // style (render.random_style)
        let (bg, ink) = match r.choice_p(&[0.5, 0.3, 0.2]) {
            0 => (ints(r, [10, 12, 16], [70, 60, 60]), ints(r, [215, 210, 200], [256, 256, 246])), // chalk
            1 => (ints(r, [225, 218, 205], [256, 252, 245]), ints(r, [0, 0, 0], [70, 60, 90])),     // ink
            _ => (ints(r, [90, 20, 20], [200, 90, 80]), ints(r, [225, 205, 150], [256, 256, 256])), // colour
        };
        let style = Style {
            line_width: r.uniform(0.07, 0.15),
            dot_radius: r.uniform(0.08, 0.16),
            margin: r.uniform(0.5, 1.4),
            show_dots: r.random() < 0.9,
            jitter: if r.random() < 0.5 { 0.0 } else { r.uniform(0.01, 0.05) },
        };
        let style_blur = if r.random() < 2.0 / 3.0 { 0.0 } else { r.uniform(0.3, 0.7) };
        let style_noise = if r.random() < 0.5 { 0.0 } else { r.uniform(1.0, 5.0) };

        let h = camera(r, s);
        let hi = inverse(&h);
        for i in 0..9 {
            p[P_HI0 + i] = hi[i / 3][i % 3] as f32;
        }

        // broken chalk lines
        let chalk = r.random() < 0.15;
        if chalk {
            p[P_CHALK] = 1.0;
            p[P_CHALK_CELLS] = r.integers(30, 90) as f32;
            p[P_CHALK_T] = r.uniform(-0.4, 0.0) as f32;
        }
        p[P_MASK_BLUR] = (style_blur * style_blur + if chalk { 0.16 } else { 0.0 }).sqrt() as f32;

        // repaint: a real floor, real inks, several colours, a thicker line (augment._repaint)
        let (surface, new_ink, multi, thick) = (r.random() < 0.85, r.random() < 0.7, r.random() < 0.1, r.random() < 0.4);
        let mut floor = bg;
        p[P_SURF] = -1.0;
        if surface {
            let kind = r.below(SURFACES.len());
            p[P_SURF] = kind as f32;
            let mut fl2 = [0.0; 3];
            floor = match SURFACES[kind] {
                "cement" => tint(r, [150., 148., 142.], 25.),
                "red_oxide" => tint(r, [150., 52., 42.], 20.),
                "mud" => {
                    p[P_SA] = r.integers(30, 70) as f32;
                    p[P_SB] = (r.random() < 0.5) as i32 as f32; // 1: streaks run down
                    tint(r, [115., 98., 62.], 18.)
                }
                "sand" => tint(r, [196., 166., 122.], 18.),
                "tiles" => {
                    let bases = [[226., 220., 205.], [180., 90., 60.], [150., 150., 150.], [235., 235., 235.]];
                    let base = bases[r.below(4)];
                    let b = tint(r, base, 15.);
                    let size = (r.uniform(0.12, 0.35) * s).floor().max(4.0);
                    p[P_SA] = size as f32;
                    p[P_SB] = r.integers(0, size as i32) as f32;
                    p[P_SC] = r.integers(0, size as i32) as f32;
                    p[P_SD] = r.uniform(0.0, 30.0) as f32;
                    b
                }
                "granite" => tint(r, [60., 58., 62.], 20.),
                "wood" => {
                    let a = tint(r, [150., 105., 65.], 20.);
                    fl2 = tint(r, [110., 72., 42.], 15.);
                    p[P_SA] = (r.random() < 0.5) as i32 as f32; // 1: grain along y
                    p[P_SB] = r.uniform(5.0, 12.0) as f32;
                    a
                }
                "paper" => tint(r, [236., 230., 214.], 10.),
                "chalkboard" => {
                    let base = [[34., 52., 40.], [30., 32., 36.]][r.below(2)];
                    tint(r, base, 8.)
                }
                _ => tint(r, [62., 62., 64.], 12.), // asphalt
            };
            for i in 0..3 {
                p[P_FL2_R + i] = fl2[i] as f32;
            }
        }
        for i in 0..3 {
            p[P_FL_R + i] = floor[i] as f32;
            p[P_BG_R + i] = bg[i] as f32;
        }
        let floor_lum = lum(floor) + if p[P_SURF] == 1.0 { 8.0 } else { 0.0 };
        let mut line = ink;
        if multi {
            let n = r.integers(2, 5) as usize;
            p[P_MULTI] = n as f32;
            for k in 0..n {
                let c = ink_colour(r, floor_lum, true);
                for i in 0..3 {
                    p[P_MC0 + 3 * k + i] = c[i] as f32;
                }
                p[P_MCELLS0 + k] = r.integers(2, 6) as f32;
            }
            line = [0, 1, 2].map(|i| (0..n).map(|k| p[P_MC0 + 3 * k + i] as f64).sum::<f64>() / n as f64);
        } else {
            if new_ink {
                line = ink_colour(r, floor_lum, false);
            }
            if (lum(line) - floor_lum).abs() < 100.0 {
                line = if floor_lum < 128.0 { [245., 240., 228.] } else { [25., 22., 30.] };
            }
        }
        for i in 0..3 {
            p[P_INK_R + i] = line[i] as f32;
        }
        if new_ink || multi {
            p[P_POWDER] = r.uniform(0.0, 0.2) as f32;
        }

        // petals, leaves and dirt
        let mut debris = [[0f32; DEBRIS_W]; MAX_DEBRIS];
        if r.random() < 0.3 {
            let cols = [[240., 140., 20.], [250., 200., 40.], [200., 30., 50.], [240., 240., 230.], [60., 130., 50.], [70., 50., 35.]];
            let n = r.integers(3, 15) as usize;
            p[P_DEBRIS] = n as f32;
            for d in debris.iter_mut().take(n) {
                let (x, y) = (r.uniform(0.0, s), r.uniform(0.0, s));
                let rad = r.uniform(1.0, 5.0) * s / 384.0;
                let (up, down) = (r.uniform(0.4, 1.0), r.uniform(0.4, 1.0));
                let c = cols[r.below(cols.len())];
                *d = [x as f32, (y + rad * (down - up) / 2.0) as f32, rad as f32, (rad * (up + down) / 2.0) as f32,
                      c[0] as f32, c[1] as f32, c[2] as f32];
            }
        }

        // light, texture, shadow, glare, vignette, colour, gray
        if r.random() < 0.3 {
            p[P_TEX_CELLS] = r.integers(4, 24) as f32;
        }
        if r.random() < 0.6 {
            p[P_LIGHT_T] = r.uniform(0.0, std::f64::consts::TAU) as f32;
            p[P_LIGHT] = (0.25 * r.uniform(0.3, 1.0)) as f32;
        }
        if r.random() < 0.3 {
            p[P_SHADOW_T] = r.uniform(0.0, std::f64::consts::PI) as f32;
            p[P_SHADOW_X] = r.uniform(0.0, s) as f32;
            p[P_SHADOW_Y] = r.uniform(0.0, s) as f32;
            p[P_SHADOW_SOFT] = (r.uniform(0.03, 0.15) * s) as f32;
            p[P_SHADOW] = r.uniform(0.1, 0.3) as f32;
        }
        p[P_GLARE] = -1.0;
        if r.random() < 0.4 {
            let kind = r.integers(0, 3);
            p[P_GLARE] = kind as f32;
            p[P_GLARE_X] = r.uniform(0.1, 0.9) as f32;
            p[P_GLARE_Y] = r.uniform(0.1, 0.9) as f32;
            match kind {
                0 => p[P_GLARE_A] = r.uniform(0.08, 0.3) as f32,
                1 => {
                    p[P_GLARE_T] = r.uniform(0.0, std::f64::consts::PI) as f32;
                    p[P_GLARE_A] = r.uniform(0.03, 0.12) as f32;
                    p[P_GLARE_B] = r.uniform(0.3, 0.8) as f32;
                }
                _ => p[P_GLARE_T] = r.uniform(0.0, std::f64::consts::TAU) as f32,
            }
            p[P_GLARE_S] = r.uniform(0.1, 0.3) as f32;
        }
        if r.random() < 0.4 {
            p[P_VIG_X] = r.uniform(0.4, 0.6) as f32;
            p[P_VIG_Y] = r.uniform(0.4, 0.6) as f32;
            p[P_VIG] = r.uniform(0.1, 0.3) as f32;
        }
        if r.random() < 0.7 {
            p[P_COLOUR] = 1.0;
            p[P_COL_C] = r.uniform(0.85, 1.15) as f32;
            p[P_COL_B] = r.uniform(0.85, 1.15) as f32;
            for i in 0..3 {
                p[P_COL_G0 + i] = r.uniform(0.9, 1.1) as f32;
            }
            p[P_COL_GAMMA] = r.uniform(0.85, 1.2) as f32;
        }
        if r.random() < 0.1 {
            p[P_GRAY] = 1.0;
        }

        // camera shake, sensor noise, then at most one more blur (defocus or low resolution), JPEG
        let mut blurred = r.random() < 0.05;
        if blurred {
            p[P_MOTION_N] = (r.integers(2, 4) as f64 * s / 384.0).floor() as f32 + 1.0;
            p[P_MOTION_T] = r.uniform(0.0, std::f64::consts::PI) as f32;
        }
        let (mut noise, mut mono) = (0.0, true);
        if r.random() < 0.4 {
            noise = r.uniform(2.0, 6.0);
            mono = r.random() < 0.5;
        }
        p[P_NOISE] = (noise * noise + style_noise * style_noise).sqrt() as f32;
        p[P_NOISE_MONO] = mono as i32 as f32;
        if !blurred && r.random() < 0.1 {
            blurred = true;
            p[P_DEFOCUS] = (r.uniform(0.4, 1.0) * s / 384.0) as f32;
        }
        if !blurred && r.random() < 0.1 {
            p[P_LOWRES] = r.uniform(0.6, 0.85) as f32;
        }
        let jpeg_quality = if r.random() < 0.5 { r.integers(50, 90) as u8 } else { 92 };

        Look { op, style, h, thick, p, debris, jpeg_quality, floor_lum, ink_lum: lum(line) }
    }

    pub fn op(&self) -> Op {
        OPS[self.op]
    }

    /// What needs the drawn line's coverage (its mean over the picture): a thicker line only where the
    /// lines have room, and the picture's mean brightness for the colour change.
    pub fn finish(&mut self, coverage: f64, s: f64) {
        if self.thick && coverage < 0.06 {
            self.p[P_THICK_R] = if s < 300.0 { 1.0 } else { 2.0 }; // a 3 or 5 px max filter, then soften 0.6
            self.p[P_MASK_BLUR] = (self.p[P_MASK_BLUR].powi(2) + 0.36).sqrt();
        }
        let m = self.floor_lum * (1.0 - coverage) + self.ink_lum * coverage;
        self.p[P_COL_MEAN] = m as f32;
    }
}
