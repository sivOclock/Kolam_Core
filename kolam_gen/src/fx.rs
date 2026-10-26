//! The picture from the line mask and its Look, on the CPU: camera warp, broken chalk, thicker and softer
//! line, floor, ink colours, petals, texture, light, shadow, glare, vignette, colour, gray, camera shake,
//! sensor noise, defocus or low resolution. fx.cu is the same, pass for pass and formula for formula,
//! on the GPU: noise is a hash of (picture seed, stream, x, y), not a random stream, so both give the
//! same picture up to float rounding. Change one, change the other.

use crate::look::*;

// ------------------------------------------------------------------ hash noise (fx.cu: same)
#[inline]
fn mix(mut h: u32) -> u32 {
    h ^= h >> 16;
    h = h.wrapping_mul(0x7feb_352d);
    h ^= h >> 15;
    h = h.wrapping_mul(0x846c_a68b);
    h ^ (h >> 16)
}

#[inline]
fn hash(seed: u32, stream: u32, x: i32, y: i32) -> u32 {
    mix(seed ^ mix(stream.wrapping_mul(0x9e37_79b9) ^ mix((x as u32).wrapping_mul(0x85eb_ca6b) ^ mix((y as u32).wrapping_add(0xc2b2_ae35)))))
}

/// Uniform in [0, 1).
#[inline]
fn u01(seed: u32, stream: u32, x: i32, y: i32) -> f32 {
    (hash(seed, stream, x, y) >> 8) as f32 * (1.0 / 16_777_216.0)
}

#[inline]
fn gauss(seed: u32, stream: u32, x: i32, y: i32) -> f32 {
    let u = u01(seed, stream, x, y).max(1e-7);
    let v = u01(seed, stream ^ 0x5555_5555, x, y);
    (-2.0 * u.ln()).sqrt() * (6.283_185_5 * v).cos()
}

#[inline]
fn cubic(t: f32) -> [f32; 4] {
    // Catmull-Rom (PIL's bicubic, a = -0.5), weights of the samples at -1, 0, 1, 2
    let (t2, t3) = (t * t, t * t * t);
    [-0.5 * t3 + t2 - 0.5 * t, 1.5 * t3 - 2.5 * t2 + 1.0, -1.5 * t3 + 2.0 * t2 + 0.5 * t, 0.5 * t3 - 0.5 * t2]
}

/// augment._smooth_noise: a coarse grid of uniform(-1, 1), `cy` cells down and `cx` across, bicubically
/// enlarged to s x s; the value at pixel (x, y).
pub fn smooth(seed: u32, stream: u32, cy: f32, cx: f32, x: i32, y: i32, s: i32) -> f32 {
    let (cy_i, cx_i) = (cy as i32, cx as i32);
    let u = (x as f32 + 0.5) * cx / s as f32 - 0.5;
    let v = (y as f32 + 0.5) * cy / s as f32 - 0.5;
    let (i0, j0) = (u.floor(), v.floor());
    let (wu, wv) = (cubic(u - i0), cubic(v - j0));
    let mut acc = 0.0;
    for b in 0..4 {
        let j = (j0 as i32 + b - 1).clamp(0, cy_i - 1);
        let mut row = 0.0;
        for a in 0..4 {
            let i = (i0 as i32 + a - 1).clamp(0, cx_i - 1);
            row += wu[a as usize] * (u01(seed, stream, i, j) * 2.0 - 1.0);
        }
        acc += wv[b as usize] * row;
    }
    acc.clamp(-1.0, 1.0)
}

/// smooth() at every pixel of one (stream, cy, cx), computed once per picture (CPU only; the GPU
/// evaluates smooth() per pixel). Same sums in the same order, so the same values.
struct Field {
    key: (u32, f32, f32),
    v: Vec<f32>,
}

fn field(seed: u32, stream: u32, cy: f32, cx: f32, s: i32) -> Field {
    let (cy_i, cx_i, su) = (cy as i32, cx as i32, s as usize);
    let along = |c: f32, k: i32| {
        let u = (k as f32 + 0.5) * c / s as f32 - 0.5;
        let i0 = u.floor();
        (i0 as i32, cubic(u - i0))
    };
    let xs: Vec<(i32, [f32; 4])> = (0..s).map(|x| along(cx, x)).collect();
    // rows[j][x]: the cubic along x of grid row j
    let rows: Vec<Vec<f32>> = (0..cy_i)
        .map(|j| {
            let g: Vec<f32> = (0..cx_i).map(|i| u01(seed, stream, i, j) * 2.0 - 1.0).collect();
            xs.iter()
                .map(|&(i0, wu)| {
                    let mut row = 0.0;
                    for a in 0..4 {
                        row += wu[a] * g[(i0 + a as i32 - 1).clamp(0, cx_i - 1) as usize];
                    }
                    row
                })
                .collect()
        })
        .collect();
    let mut v = vec![0.0f32; su * su];
    for y in 0..s {
        let (j0, wv) = along(cy, y);
        for x in 0..su {
            let mut acc = 0.0;
            for b in 0..4 {
                acc += wv[b] * rows[(j0 + b as i32 - 1).clamp(0, cy_i - 1) as usize][x];
            }
            v[y as usize * su + x] = acc.clamp(-1.0, 1.0);
        }
    }
    Field { key: (stream, cy, cx), v }
}

/// The picture's noise fields; smooth() itself for one not made ahead.
struct Noise {
    seed: u32,
    s: i32,
    fields: Vec<Field>,
}

impl Noise {
    fn new(p: &[f32], s: i32) -> Noise {
        let seed = seed_of(p);
        let mut keys: Vec<(u32, f32, f32)> = Vec::new();
        if p[P_CHALK] > 0.0 {
            keys.push((S_CHALK, p[P_CHALK_CELLS], p[P_CHALK_CELLS]));
        }
        let (a, f) = (p[P_SA], S_FLOOR);
        keys.extend(match p[P_SURF] as i32 {
            0 => vec![(f, 6., 6.), (f + 1, 40., 40.)],
            1 => vec![(f, 5., 5.), (f + 1, 3., 3.)],
            2 => vec![if p[P_SB] > 0.5 { (f, a, 4.) } else { (f, 4., a) }, (f + 1, 12., 12.)],
            3 => vec![(f, 8., 8.)],
            4 => vec![(f, 20., 20.), (f + 1, 5., 5.), (f + 2, 17., 17.)],
            6 => vec![(f, 3., 8.), if p[P_SA] > 0.5 { (f + 1, 90., 4.) } else { (f + 1, 4., 90.) }],
            7 => vec![(f, 10., 10.), (f + 1, 80., 80.)],
            8 => vec![(f, 7., 7.), (f + 1, 30., 30.)],
            9 => vec![(f, 10., 10.)],
            _ => vec![],
        });
        for k in 0..p[P_MULTI] as usize {
            keys.push((S_MULTI + k as u32, p[P_MCELLS0 + k], p[P_MCELLS0 + k]));
        }
        if p[P_TEX_CELLS] > 0.0 {
            keys.push((S_TEX, p[P_TEX_CELLS], p[P_TEX_CELLS]));
        }
        Noise { seed, s, fields: keys.into_iter().map(|(st, cy, cx)| field(seed, st, cy, cx, s)).collect() }
    }

    #[inline]
    fn at(&self, stream: u32, cy: f32, cx: f32, x: i32, y: i32) -> f32 {
        for f in &self.fields {
            if f.key == (stream, cy, cx) {
                return f.v[(y * self.s + x) as usize];
            }
        }
        smooth(self.seed, stream, cy, cx, x, y, self.s)
    }
}

// streams: one per use of the noise
const S_CHALK: u32 = 1;
const S_CHALK_DOT: u32 = 2;
const S_POWDER: u32 = 3;
const S_MULTI: u32 = 10; // + colour
const S_TEX: u32 = 20;
const S_FLOOR: u32 = 30; // + 0..9
const S_GRAIN: u32 = 40;
const S_SPECK: u32 = 41;
const S_TILE: u32 = 42; // + channel
const S_NOISE: u32 = 50; // + channel

// ------------------------------------------------------------------ per pixel
#[inline]
fn seed_of(p: &[f32]) -> u32 {
    p[P_SEED].to_bits()
}

/// Camera warp (bilinear, nothing outside the drawing) and broken chalk.
fn warp_chalk(mask: &[u8], p: &[f32], nz: &Noise, x: i32, y: i32, s: i32) -> f32 {
    let (fx, fy) = (x as f32 + 0.5, y as f32 + 0.5);
    let w = p[P_HI6] * fx + p[P_HI7] * fy + p[P_HI8];
    let sx = (p[P_HI0] * fx + p[P_HI1] * fy + p[P_HI2]) / w - 0.5;
    let sy = (p[P_HI3] * fx + p[P_HI4] * fy + p[P_HI5]) / w - 0.5;
    let (x0, y0) = (sx.floor(), sy.floor());
    let (ax, ay) = (sx - x0, sy - y0);
    let at = |i: i32, j: i32| if i < 0 || j < 0 || i >= s || j >= s { 0.0 } else { mask[(j * s + i) as usize] as f32 / 255.0 };
    let (i, j) = (x0 as i32, y0 as i32);
    let mut a = (1.0 - ay) * ((1.0 - ax) * at(i, j) + ax * at(i + 1, j)) + ay * ((1.0 - ax) * at(i, j + 1) + ax * at(i + 1, j + 1));
    if p[P_CHALK] > 0.0 {
        let seed = seed_of(p);
        let ink = if a > 0.12 { 1.0 } else { 0.0 };
        let keep = nz.at(S_CHALK, p[P_CHALK_CELLS], p[P_CHALK_CELLS], x, y);
        let mut fade = ((p[P_CHALK_T] - keep) * 2.5).clamp(0.0, 1.0) * ink;
        if u01(seed, S_CHALK_DOT, x, y) < 0.05 {
            fade = fade.max(0.4 * ink);
        }
        a *= 1.0 - fade;
    }
    a
}

/// augment._surface: a procedural floor at pixel (x, y).
fn floor(p: &[f32], nz: &Noise, x: i32, y: i32, s: i32) -> [f32; 3] {
    let seed = seed_of(p);
    let fl = [p[P_FL_R], p[P_FL_G], p[P_FL_B]];
    let n = |k: u32, cy: f32, cx: f32| nz.at(S_FLOOR + k, cy, cx, x, y);
    let grain = |amt: f32| amt * gauss(seed, S_GRAIN, x, y);
    let add = |v: f32| fl.map(|c| c + v);
    match p[P_SURF] as i32 {
        0 => add(18.0 * n(0, 6., 6.) + 10.0 * n(1, 40., 40.) + grain(9.0) - if u01(seed, S_SPECK, x, y) < 0.004 { 40.0 } else { 0.0 }),
        1 => add(14.0 * n(0, 5., 5.) + grain(4.0) + 35.0 * n(1, 3., 3.).max(0.0)),
        2 => {
            let a = p[P_SA];
            let streak = if p[P_SB] > 0.5 { n(0, a, 4.) } else { n(0, 4., a) };
            add(16.0 * streak + 10.0 * n(1, 12., 12.) + grain(6.0))
        }
        3 => add(10.0 * n(0, 8., 8.) + grain(14.0)),
        4 => {
            let (size, ox, oy) = (p[P_SA] as i32, p[P_SB] as i32, p[P_SC] as i32);
            let line = (size / 40).max(2);
            if (y + oy) % size < line || (x + ox) % size < line {
                return fl.map(|c| c * 0.55); // grout
            }
            let (ti, tj) = ((y + oy) / size, (x + ox) / size);
            let vein = (n(1, 5., 5.) * 9.0 + n(2, 17., 17.) * 3.0).sin().abs();
            let v = 6.0 * n(0, 20., 20.) + grain(3.0) - if vein < 0.08 { p[P_SD] } else { 0.0 };
            [0, 1, 2].map(|c| fl[c] + u01(seed, S_TILE + c as u32, tj, ti) * 24.0 - 12.0 + v)
        }
        5 => {
            let speck = if u01(seed, S_SPECK, x, y) < 0.05 { 30.0 * (u01(seed, S_SPECK + 1, x, y) * 2.0 - 1.0) } else { 0.0 };
            add(grain(7.0) + speck)
        }
        6 => {
            let along_y = p[P_SA] > 0.5;
            let along = if along_y { y } else { x } as f32;
            let t = along / s as f32 * p[P_SB] + 0.6 * n(0, 3., 8.);
            let m = (0.5 + 0.5 * (t * 6.283_185_5).sin()).powi(3);
            let fine = if along_y { n(1, 90., 4.) } else { n(1, 4., 90.) };
            let v = 6.0 * fine + grain(4.0);
            [0, 1, 2].map(|c| fl[c] * (1.0 - m) + p[P_FL2_R + c] * m + v)
        }
        7 => add(5.0 * n(0, 10., 10.) + 4.0 * n(1, 80., 80.) + grain(3.0)),
        8 => add(12.0 * n(0, 7., 7.).max(0.0) + 4.0 * n(1, 30., 30.) + grain(3.0)),
        9 => add(8.0 * n(0, 10., 10.) + grain(16.0)),
        _ => [p[P_BG_R], p[P_BG_G], p[P_BG_B]],
    }
}

/// Everything from the line coverage `a` to a colour: floor, ink, powder, petals, texture, light,
/// shadow, glare, vignette, colour, gray.
fn colour(p: &[f32], debris: &[f32], nz: &Noise, mut a: f32, x: i32, y: i32, s: i32) -> [f32; 3] {
    let seed = seed_of(p);
    let fl = floor(p, nz, x, y, s);
    let mut ink = [p[P_INK_R], p[P_INK_G], p[P_INK_B]];
    let multi = p[P_MULTI] as usize;
    if multi > 0 {
        let (mut sum, mut acc) = (0.0, [0.0f32; 3]);
        for k in 0..multi {
            let c = p[P_MCELLS0 + k];
            let w = (6.0 * nz.at(S_MULTI + k as u32, c, c, x, y)).exp();
            sum += w;
            for i in 0..3 {
                acc[i] += w * p[P_MC0 + 3 * k + i];
            }
        }
        ink = acc.map(|v| v / sum);
    }
    if p[P_POWDER] > 0.0 {
        a *= 1.0 - p[P_POWDER] * u01(seed, S_POWDER, x, y);
    }
    let mut c = [0, 1, 2].map(|i| fl[i] * (1.0 - a) + ink[i] * a);
    let (fx, fy) = (x as f32 + 0.5, y as f32 + 0.5);
    for d in debris.chunks_exact(DEBRIS_W).take(p[P_DEBRIS] as usize) {
        let (dx, dy) = ((fx - d[0]) / d[2], (fy - d[1]) / d[3]);
        if dx * dx + dy * dy <= 1.0 {
            c = [d[4], d[5], d[6]];
        }
    }
    let mut f = 1.0;
    if p[P_TEX_CELLS] > 0.0 {
        f *= 1.0 + 0.08 * nz.at(S_TEX, p[P_TEX_CELLS], p[P_TEX_CELLS], x, y);
    }
    let (u, v) = (fx / s as f32, fy / s as f32);
    if p[P_LIGHT] > 0.0 {
        let t = p[P_LIGHT_T];
        f *= 1.0 + p[P_LIGHT] * (t.cos() * (u - 0.5) + t.sin() * (v - 0.5)) * 2.0;
    }
    if p[P_SHADOW] > 0.0 {
        let t = p[P_SHADOW_T];
        let d = t.cos() * (fx - p[P_SHADOW_X]) + t.sin() * (fy - p[P_SHADOW_Y]);
        f *= 1.0 - p[P_SHADOW] / (1.0 + (-d / p[P_SHADOW_SOFT]).exp());
    }
    c = c.map(|v| v * f);
    if p[P_GLARE] >= 0.0 {
        let (cx, cy, t) = (p[P_GLARE_X], p[P_GLARE_Y], p[P_GLARE_T]);
        let g = match p[P_GLARE] as i32 {
            0 => (-((u - cx).powi(2) + (v - cy).powi(2)) / (2.0 * p[P_GLARE_A].powi(2))).exp(),
            1 => {
                let d = t.cos() * (u - cx) + t.sin() * (v - cy);
                let along = -t.sin() * (u - cx) + t.cos() * (v - cy);
                (-d * d / (2.0 * p[P_GLARE_A].powi(2))).exp() * (-along * along / (2.0 * p[P_GLARE_B].powi(2))).exp()
            }
            _ => (0.5 + t.cos() * (u - 0.5) + t.sin() * (v - 0.5)).clamp(0.0, 1.0).powi(2),
        };
        let w = g * p[P_GLARE_S];
        c = [0, 1, 2].map(|i| c[i] + ([255.0, 252.0, 240.0][i] - c[i]) * w);
    }
    if p[P_VIG] > 0.0 {
        let r2 = ((u - p[P_VIG_X]).powi(2) + (v - p[P_VIG_Y]).powi(2)) * 2.0;
        c = c.map(|v| v * (1.0 - p[P_VIG] * r2));
    }
    if p[P_COLOUR] > 0.0 {
        let m = p[P_COL_MEAN];
        c = [0, 1, 2].map(|i| {
            let v = ((c[i] - m) * p[P_COL_C] + m * p[P_COL_B]) * p[P_COL_G0 + i];
            255.0 * (v.clamp(0.0, 255.0) / 255.0).powf(p[P_COL_GAMMA])
        });
    }
    if p[P_GRAY] > 0.0 {
        let m = (c[0] + c[1] + c[2]) / 3.0;
        c = [m; 3];
    }
    c
}

fn noise(p: &[f32], c: &mut [f32; 3], x: i32, y: i32) {
    if p[P_NOISE] > 0.0 {
        let seed = seed_of(p);
        let mono = p[P_NOISE_MONO] > 0.5;
        for (i, v) in c.iter_mut().enumerate() {
            *v += p[P_NOISE] * gauss(seed, S_NOISE + if mono { 0 } else { i as u32 }, x, y);
        }
    }
    for v in c.iter_mut() {
        *v = v.clamp(0.0, 255.0);
    }
}

// ------------------------------------------------------------------ neighbourhood passes (planes of s x s)
fn gauss_weights(sigma: f32) -> Vec<f32> {
    let r = ((3.0 * sigma).ceil() as usize).clamp(1, 10);
    let w: Vec<f32> = (0..=2 * r).map(|k| (-((k as f32 - r as f32).powi(2)) / (2.0 * sigma * sigma)).exp()).collect();
    let t: f32 = w.iter().sum();
    w.into_iter().map(|v| v / t).collect()
}

/// Gaussian blur of `planes` planes, in place (edges clamped).
fn blur(buf: &mut [f32], planes: usize, s: usize, sigma: f32) {
    if sigma <= 0.0 {
        return;
    }
    let w = gauss_weights(sigma);
    let r = (w.len() / 2) as isize;
    let mut tmp = vec![0.0f32; s * s];
    for pl in buf.chunks_exact_mut(s * s).take(planes) {
        for y in 0..s {
            for x in 0..s {
                tmp[y * s + x] = (0..w.len()).map(|k| w[k] * pl[y * s + (x as isize + k as isize - r).clamp(0, s as isize - 1) as usize]).sum();
            }
        }
        for y in 0..s {
            for x in 0..s {
                pl[y * s + x] = (0..w.len()).map(|k| w[k] * tmp[(y as isize + k as isize - r).clamp(0, s as isize - 1) as usize * s + x]).sum();
            }
        }
    }
}

/// Max filter of radius r (a thicker line), in place.
fn max_filter(a: &mut [f32], s: usize, r: usize) {
    if r == 0 {
        return;
    }
    let mut tmp = vec![0.0f32; s * s];
    for y in 0..s {
        for x in 0..s {
            tmp[y * s + x] = (x.saturating_sub(r)..=(x + r).min(s - 1)).map(|i| a[y * s + i]).fold(0.0, f32::max);
        }
    }
    for y in 0..s {
        for x in 0..s {
            a[y * s + x] = (y.saturating_sub(r)..=(y + r).min(s - 1)).map(|j| tmp[j * s + x]).fold(0.0, f32::max);
        }
    }
}

/// Camera shake (augment._motion): the mean of n copies shifted along angle t, wrapping like np.roll.
fn motion_px(src: &[f32], p: &[f32], x: i32, y: i32, s: i32) -> [f32; 3] {
    let n = p[P_MOTION_N] as i32;
    let t = p[P_MOTION_T];
    let mut c = [0.0f32; 3];
    for k in 0..n {
        let sh = -n as f32 / 2.0 + if n > 1 { k as f32 * n as f32 / (n - 1) as f32 } else { 0.0 };
        let (dy, dx) = ((sh * t.sin()).round() as i32, (sh * t.cos()).round() as i32);
        let (sx, sy) = ((x - dx).rem_euclid(s), (y - dy).rem_euclid(s));
        for (i, v) in c.iter_mut().enumerate() {
            *v += src[(i as i32 * s * s + sy * s + sx) as usize];
        }
    }
    c.map(|v| v / n as f32)
}

/// Low resolution: box-shrink to m x m, bilinear back up.
fn lowres_px(src: &[f32], m: i32, x: i32, y: i32, s: i32) -> [f32; 3] {
    let small = |i: i32, j: i32, ch: i32| {
        let (x0, x1) = (i * s / m, ((i + 1) * s / m).max(i * s / m + 1));
        let (y0, y1) = (j * s / m, ((j + 1) * s / m).max(j * s / m + 1));
        let mut t = 0.0;
        for yy in y0..y1 {
            for xx in x0..x1 {
                t += src[(ch * s * s + yy * s + xx) as usize];
            }
        }
        t / ((x1 - x0) * (y1 - y0)) as f32
    };
    let u = ((x as f32 + 0.5) * m as f32 / s as f32 - 0.5).clamp(0.0, (m - 1) as f32);
    let v = ((y as f32 + 0.5) * m as f32 / s as f32 - 0.5).clamp(0.0, (m - 1) as f32);
    let (i, j) = (u.floor() as i32, v.floor() as i32);
    let (a, b) = (u - i as f32, v - j as f32);
    let (i1, j1) = ((i + 1).min(m - 1), (j + 1).min(m - 1));
    [0, 1, 2].map(|ch| {
        (1.0 - b) * ((1.0 - a) * small(i, j, ch) + a * small(i1, j, ch)) + b * ((1.0 - a) * small(i, j1, ch) + a * small(i1, j1, ch))
    })
}

// ------------------------------------------------------------------ the whole picture
/// Mask (s x s) + the look's parameter row and petals -> RGB bytes (s x s x 3).
pub fn run(mask: &[u8], p: &[f32], debris: &[f32], s: usize) -> Vec<u8> {
    let si = s as i32;
    let n = s * s;
    let nz = Noise::new(p, si);
    let mut a: Vec<f32> = (0..n).map(|k| warp_chalk(mask, p, &nz, (k % s) as i32, (k / s) as i32, si)).collect();
    max_filter(&mut a, s, p[P_THICK_R] as usize);
    blur(&mut a, 1, s, p[P_MASK_BLUR]);
    let mut rgb = vec![0.0f32; 3 * n];
    for k in 0..n {
        let c = colour(p, debris, &nz, a[k], (k % s) as i32, (k / s) as i32, si);
        for i in 0..3 {
            rgb[i * n + k] = c[i];
        }
    }
    if p[P_MOTION_N] > 0.0 {
        let src = rgb.clone();
        for k in 0..n {
            let c = motion_px(&src, p, (k % s) as i32, (k / s) as i32, si);
            for i in 0..3 {
                rgb[i * n + k] = c[i];
            }
        }
    }
    for k in 0..n {
        let mut c = [rgb[k], rgb[n + k], rgb[2 * n + k]];
        noise(p, &mut c, (k % s) as i32, (k / s) as i32);
        for i in 0..3 {
            rgb[i * n + k] = c[i];
        }
    }
    blur(&mut rgb, 3, s, p[P_DEFOCUS]);
    if p[P_LOWRES] > 0.0 {
        let m = ((s as f32 * p[P_LOWRES]) as i32).max(16);
        let src = rgb.clone();
        for k in 0..n {
            let c = lowres_px(&src, m, (k % s) as i32, (k / s) as i32, si);
            for i in 0..3 {
                rgb[i * n + k] = c[i];
            }
        }
    }
    let mut out = vec![0u8; 3 * n];
    for k in 0..n {
        for i in 0..3 {
            out[3 * k + i] = rgb[i * n + k].round().clamp(0.0, 255.0) as u8;
        }
    }
    out
}
