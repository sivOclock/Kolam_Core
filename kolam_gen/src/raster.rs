//! The kolam as a line mask (render.render_kolam, before any colour): only the dots some relation uses,
//! every stroke of geometry::drawn as a closed chained Bezier, anti-aliased by tiny-skia at the
//! picture's size. Colour, floor and every image change come after (fx.rs / fx.cu).

use crate::geometry::{self, per_segment, sample_stroke};
use crate::look::{normal, Style, M3};
use crate::model::{Kolam, P};
use crate::rng::Rng;
use tiny_skia::{FillRule, LineCap, LineJoin, Paint, PathBuilder, Pixmap, Stroke, Transform};

pub struct Raster {
    pub mask: Vec<u8>, // s x s, 255 = line
    pub m: M3,         // lattice point (lj, li, 1) -> pixel (x, y, 1)
    pub coverage: f64, // mean of the mask, 0..1
}

/// Hand wobble (render._jittered): a smooth noise field, random offsets at lattice points and bilinear in
/// between, so shared points move together.
fn jittered(strokes: &[Vec<P>], jitter: f64, r: &mut Rng) -> Vec<Vec<P>> {
    if jitter <= 0.0 || strokes.is_empty() {
        return strokes.to_vec();
    }
    let all = strokes.iter().flatten();
    let lo0 = all.clone().map(|p| p.0).fold(f64::INFINITY, f64::min).floor() as i64 - 1;
    let lo1 = all.clone().map(|p| p.1).fold(f64::INFINITY, f64::min).floor() as i64 - 1;
    let hi0 = all.clone().map(|p| p.0).fold(f64::NEG_INFINITY, f64::max).ceil() as i64 + 2;
    let hi1 = all.map(|p| p.1).fold(f64::NEG_INFINITY, f64::max).ceil() as i64 + 2;
    let (h, w) = ((hi0 - lo0) as usize, (hi1 - lo1) as usize);
    let n: Vec<(f64, f64)> = (0..h * w).map(|_| (normal(r) * jitter, normal(r) * jitter)).collect();
    let at = |i: usize, j: usize| n[i.min(h - 1) * w + j.min(w - 1)];
    let mv = |p: P| {
        let (x, y) = (p.0 - lo0 as f64, p.1 - lo1 as f64);
        let (i, j) = (x.floor() as usize, y.floor() as usize);
        let (a, b) = (x - i as f64, y - j as f64);
        let (n00, n10, n01, n11) = (at(i, j), at(i + 1, j), at(i, j + 1), at(i + 1, j + 1));
        let o0 = (1.0 - a) * (1.0 - b) * n00.0 + a * (1.0 - b) * n10.0 + (1.0 - a) * b * n01.0 + a * b * n11.0;
        let o1 = (1.0 - a) * (1.0 - b) * n00.1 + a * (1.0 - b) * n10.1 + (1.0 - a) * b * n01.1 + a * b * n11.1;
        (p.0 + o0, p.1 + o1)
    };
    strokes.iter().map(|s| s.iter().map(|&p| mv(p)).collect()).collect()
}

/// Draw kolam k (already turned) at s x s. None if it does not trace (never for a checked label).
pub fn draw(k: &Kolam, style: &Style, s: u32, r: &mut Rng) -> Option<Raster> {
    let c = geometry::compile(k)?;
    let shown = geometry::drawn(&c);
    // the frame: the dot grid, grown to fit a crescent that points off it
    let (mut t, mut b, mut lf, mut rt) = (0.0f64, 2.0 * k.rows as f64, 0.0f64, 2.0 * k.cols as f64);
    for p in shown.iter().flatten() {
        (t, b, lf, rt) = (t.min(p.0), b.max(p.0), lf.min(p.1), rt.max(p.1));
    }
    let sf = s as f64;
    let scale = sf / ((b - t).max(rt - lf) + 2.0 * style.margin);
    let ox = (sf - (rt - lf) * scale) / 2.0 - lf * scale;
    let oy = (sf - (b - t) * scale) / 2.0 - t * scale;
    let px = |p: P| ((ox + p.1 * scale) as f32, (oy + p.0 * scale) as f32);

    let mut pm = Pixmap::new(s, s)?;
    let mut paint = Paint::default();
    paint.set_color_rgba8(255, 255, 255, 255);
    paint.anti_alias = true;
    if style.show_dots {
        let rad = (style.dot_radius * scale) as f32;
        for i in 0..k.rows {
            for j in 0..k.cols {
                if k.is_used((i, j)) {
                    let (x, y) = px(((2 * i + 1) as f64, (2 * j + 1) as f64));
                    if let Some(path) = PathBuilder::from_circle(x, y, rad.max(0.5)) {
                        pm.fill_path(&path, &paint, FillRule::Winding, Transform::identity(), None);
                    }
                }
            }
        }
    }
    let stroke = Stroke {
        width: ((style.line_width * scale) as f32).max(0.6),
        line_cap: LineCap::Round,
        line_join: LineJoin::Round,
        ..Stroke::default()
    };
    for st in jittered(&shown, style.jitter, r) {
        let pts = sample_stroke(&st, per_segment(&st, 16));
        let mut pb = PathBuilder::new();
        for (i, &p) in pts.iter().enumerate() {
            let (x, y) = px(p);
            if i == 0 { pb.move_to(x, y) } else { pb.line_to(x, y) }
        }
        pb.close();
        if let Some(path) = pb.finish() {
            pm.stroke_path(&path, &paint, &stroke, Transform::identity(), None);
        }
    }
    let mask: Vec<u8> = pm.data().chunks_exact(4).map(|px| px[3]).collect();
    let coverage = mask.iter().map(|&v| v as f64).sum::<f64>() / (255.0 * mask.len() as f64);
    Some(Raster { mask, m: [[scale, 0.0, ox], [0.0, scale, oy], [0.0, 0.0, 1.0]], coverage })
}
