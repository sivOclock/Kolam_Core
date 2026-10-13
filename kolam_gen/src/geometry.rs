//! Gate compiler, drawing and geometric check: port of kolam_press/compiler.py (trace, horn tips) and
//! geometry.py (drawn: capsules, box holes, horn growth; check_geometry). Same rules and thresholds;
//! close pairs are found with a spatial grid instead of all pairs (same pairs, much faster).

use crate::model::*;
use std::collections::{HashMap, HashSet};

const D2: [V; 4] = [(1, 1), (1, -1), (-1, 1), (-1, -1)]; // compiler.DIRS: diagonal moves between gates

fn d2_index(d: V) -> usize {
    match d {
        (1, 1) => 0,
        (1, -1) => 1,
        (-1, 1) => 2,
        _ => 3,
    }
}

/// Follow the gate rules from every unvisited diagonal link until each stroke closes (compiler.trace).
/// Strokes that only circle an unused dot are dropped. None if a trace does not close.
pub fn trace(rows: i32, cols: i32, gates: &[u8], used: &[bool]) -> Option<Vec<Vec<V>>> {
    let (h, w) = (2 * rows, 2 * cols);
    let lw = (w + 1) as usize;
    let idx = |p: V| p.0 as usize * lw + p.1 as usize;
    let mut seen = vec![0u8; ((h + 1) * (w + 1)) as usize];
    let flip = &spec().flip;
    let limit = (4 * (h + 1) * (w + 1)) as usize;
    let mut strokes = Vec::new();
    for a in 0..=h {
        for b in 0..=w {
            if (a + b) % 2 == 0 {
                continue;
            }
            let g = (a, b);
            for (di, &d) in D2.iter().enumerate() {
                let q = (a + d.0, b + d.1);
                if q.0 < 0 || q.0 > h || q.1 < 0 || q.1 > w || seen[idx(g)] & (1 << di) != 0 {
                    continue;
                }
                let mut seq = vec![g];
                let (mut cur, mut dd) = (g, d);
                loop {
                    let nxt = (cur.0 + dd.0, cur.1 + dd.1);
                    let i = d2_index(dd);
                    seen[idx(cur)] |= 1 << i;
                    seen[idx(nxt)] |= 1 << (3 - i);
                    cur = nxt;
                    let f = flip[gates[idx(cur)] as usize][if is_h_gate(cur) { 0 } else { 1 }];
                    dd = (dd.0 * f.0, dd.1 * f.1);
                    if cur == g && dd == d {
                        break;
                    }
                    seq.push(cur);
                    if seq.len() > limit {
                        return None;
                    }
                }
                strokes.push(seq);
            }
        }
    }
    strokes.retain(|s| !bare_ring(s, rows, cols, used));
    Some(strokes)
}

/// The 4-gate stroke round a dot no relation uses (not drawn).
fn bare_ring(s: &[V], rows: i32, cols: i32, used: &[bool]) -> bool {
    if s.len() != 4 {
        return false;
    }
    let (sr, sc) = (s.iter().map(|p| p.0).sum::<i32>(), s.iter().map(|p| p.1).sum::<i32>());
    if sr % 4 != 0 || sc % 4 != 0 {
        return false;
    }
    let (li, lj) = (sr / 4, sc / 4);
    if li % 2 == 0 || lj % 2 == 0 {
        return false;
    }
    let v = ((li - 1) / 2, (lj - 1) / 2);
    if v.0 < 0 || v.0 >= rows || v.1 < 0 || v.1 >= cols || used[(v.0 * cols + v.1) as usize] {
        return false;
    }
    let ring = [(li - 1, lj), (li + 1, lj), (li, lj - 1), (li, lj + 1)];
    s.iter().all(|p| ring.contains(p)) && (0..4).all(|i| !s[..i].contains(&s[i]))
}

pub struct Tip {
    pub dot: V,
    pub dir: u8,
    pub var: u8,
    pub long: bool,
    pub point: P,
}

pub struct Compiled<'a> {
    pub k: &'a Kolam,
    pub strokes: Vec<Vec<V>>,
    pub tips: Vec<Tip>,
}

pub fn compile(k: &Kolam) -> Option<Compiled<'_>> {
    let strokes = trace(k.rows, k.cols, &k.gates, &k.used)?;
    let tips = k
        .horns
        .iter()
        .map(|&(v, d, var)| {
            let long = !k.is_used(horn_target(v, d));
            Tip { dot: v, dir: d, var, long, point: horn_tip(v, d, var, long) }
        })
        .collect();
    Some(Compiled { k, strokes, tips })
}

// ------------------------------------------------------------------ drawing (geometry.drawn)
fn round_even(x: f64) -> f64 {
    x.round_ties_even()
}

pub fn per_segment(pts: &[P], n: usize) -> usize {
    let k = pts.len();
    let mut s = 0.0;
    for i in 0..k {
        let (a, b) = (pts[i], pts[(i + 1) % k]);
        s += ((b.0 - a.0).powi(2) + (b.1 - a.1).powi(2)).sqrt();
    }
    let step = s / k as f64;
    (round_even(n as f64 * step / 1.41) as i64).max(2) as usize
}

/// Closed chained quadratic Bezier: segment i runs mid(P[i-1],P[i]) -> mid(P[i],P[i+1]) with control P[i].
pub fn sample_stroke(pts: &[P], n: usize) -> Vec<P> {
    let k = pts.len();
    let mut out = Vec::with_capacity(k * n);
    for i in 0..k {
        let (p0, p, p1) = (pts[(i + k - 1) % k], pts[i], pts[(i + 1) % k]);
        let a = ((p0.0 + p.0) / 2.0, (p0.1 + p.1) / 2.0);
        let b = ((p.0 + p1.0) / 2.0, (p.1 + p1.1) / 2.0);
        for j in 0..n {
            let t = j as f64 / n as f64;
            let (u, m, v) = ((1.0 - t) * (1.0 - t), 2.0 * (1.0 - t) * t, t * t);
            out.push((u * a.0 + m * p.0 + v * b.0, u * a.1 + m * p.1 + v * b.1));
        }
    }
    out
}

const CAPSULE_W: f64 = 0.75;
const CAPSULE_LEAN: f64 = 0.5;
const HORN_SAMPLES: usize = 10;

fn is_box_hole(k: &Kolam, s: &[V]) -> bool {
    s.len() == 4
        && s.iter().all(|&g| k.gate(g) == JOIN)
        && s.iter().map(|g| g.0).sum::<i32>() % 8 == 0
        && s.iter().map(|g| g.1).sum::<i32>() % 8 == 0
}

fn capsule_points(k: &Kolam, join_dot: &[bool], s: &[V]) -> Vec<P> {
    let n = s.len();
    let lw = k.lw();
    let w = CAPSULE_W;
    let mut pts: Vec<P> = Vec::with_capacity(n * 2);
    let mut around: Vec<Option<(P, P)>> = Vec::with_capacity(n);
    for i in 0..n {
        let g = s[i];
        let (p, q) = (s[(i + n - 1) % n], s[(i + 1) % n]);
        let st = k.gate(g);
        if st == JOIN {
            let feeds_x = k.gate(p) == CROSS || k.gate(q) == CROSS;
            let off = if feeds_x { w.min(CAPSULE_LEAN) } else { w };
            if is_h_gate(g) {
                let side = if p.0 > g.0 { 1.0 } else { -1.0 };
                pts.push((g.0 as f64 + side * off, g.1 as f64));
            } else {
                let side = if p.1 > g.1 { 1.0 } else { -1.0 };
                pts.push((g.0 as f64, g.1 as f64 + side * off));
            }
            around.push(None);
            continue;
        }
        let (sr, sc) = (p.0 + q.0, p.1 + q.1);
        if st == SPLIT && sr % 2 == 0 && sc % 2 == 0 && (sr / 2) % 2 != 0 && (sc / 2) % 2 != 0 {
            let d = (sr / 2, sc / 2);
            if d.0 >= 0 && d.1 >= 0 && join_dot[d.0 as usize * lw + d.1 as usize] {
                let u = ((g.0 - d.0) as f64, (g.1 - d.1) as f64);
                pts.push((d.0 as f64 + u.0 * w, d.1 as f64 + u.1 * w));
                around.push(Some(((d.0 as f64, d.1 as f64), u)));
                continue;
            }
        }
        pts.push((g.0 as f64, g.1 as f64));
        around.push(None);
    }
    let mut out = Vec::with_capacity(n * 2);
    for i in 0..n {
        out.push(pts[i]);
        if let (Some((d, u)), Some((d2, v))) = (around[i], around[(i + 1) % n]) {
            if d == d2 {
                let m = ((u.0 + v.0).powi(2) + (u.1 + v.1).powi(2)).sqrt();
                if m > 0.0 {
                    out.push((d.0 + w * (u.0 + v.0) / m, d.1 + w * (u.1 + v.1) / m));
                }
            }
        }
    }
    out
}

/// What is drawn (geometry.drawn): box holes dropped, parallel runs as capsules, horns grown.
pub fn drawn(c: &Compiled) -> Vec<Vec<P>> {
    let k = c.k;
    let lw = k.lw();
    let mut join_dot = vec![false; k.gates.len()];
    for i in 0..=2 * k.rows {
        for j in 0..=2 * k.cols {
            if k.gate((i, j)) == JOIN {
                let ds = if is_h_gate((i, j)) { [(i, j - 1), (i, j + 1)] } else { [(i - 1, j), (i + 1, j)] };
                for d in ds {
                    join_dot[d.0 as usize * lw + d.1 as usize] = true;
                }
            }
        }
    }
    let mut strokes: Vec<Vec<P>> =
        c.strokes.iter().filter(|s| !is_box_hole(k, s)).map(|s| capsule_points(k, &join_dot, s)).collect();
    if !c.tips.is_empty() && !strokes.is_empty() {
        grow_horns(&c.tips, &mut strokes);
    }
    strokes
}

fn grow_horns(tips: &[Tip], strokes: &mut [Vec<P>]) {
    let mut dense: Vec<Option<Vec<P>>> = vec![None; strokes.len()];
    let dirs: HashSet<(V, V)> = tips.iter().map(|t| (t.dot, DIRS[t.dir as usize])).collect();
    for t in tips {
        let o = ((2 * t.dot.0 + 1) as f64, (2 * t.dot.1 + 1) as f64);
        let fi = DIRS[t.dir as usize];
        let li = left_of(fi);
        let f = (fi.0 as f64, fi.1 as f64);
        let l = (li.0 as f64, li.1 as f64);
        let front = (o.0 + 0.75 * f.0, o.1 + 0.75 * f.1);
        let gap = |q: &[P]| q.iter().map(|p| ((p.0 - front.0).powi(2) + (p.1 - front.1).powi(2)).sqrt()).fold(f64::INFINITY, f64::min);
        let mut best = (f64::INFINITY, 0);
        for (kk, s) in strokes.iter().enumerate() {
            let g = match &dense[kk] {
                Some(d) => gap(d),
                None => gap(&sample_stroke(s, 4)),
            };
            if g < best.0 {
                best = (g, kk);
            }
        }
        let kk = best.1;
        let d = dense[kk].take().unwrap_or_else(|| sample_stroke(&strokes[kk], HORN_SAMPLES));
        let clip = [dirs.contains(&(t.dot, li)), dirs.contains(&(t.dot, (-li.0, -li.1)))];
        dense[kk] = Some(grow(d, t, o, f, l, clip));
    }
    for (kk, d) in dense.into_iter().enumerate() {
        if let Some(d) = d {
            strokes[kk] = d;
        }
    }
}

fn grow(d: Vec<P>, t: &Tip, o: P, f: P, l: P, clip: [bool; 2]) -> Vec<P> {
    let sh = shape(t.long);
    let n = d.len();
    let u: Vec<f64> = d.iter().map(|p| (p.0 - o.0) * f.0 + (p.1 - o.1) * f.1).collect();
    let v: Vec<f64> = d.iter().map(|p| (p.0 - o.0) * l.0 + (p.1 - o.1) * l.1).collect();
    let mut c = 0;
    let mut cb = f64::INFINITY;
    for i in 0..n {
        let x = if u[i] > 0.0 { (u[i] - 0.75).hypot(v[i]) } else { f64::INFINITY };
        if x < cb {
            cb = x;
            c = i;
        }
    }
    let mut pos = Vec::with_capacity(n);
    let mut total = 0.0;
    for i in 0..n {
        pos.push(total);
        let (a, b) = (d[i], d[(i + 1) % n]);
        total += ((b.0 - a.0).powi(2) + (b.1 - a.1).powi(2)).sqrt();
    }
    let flipd = v[(c + 1) % n] < v[c];
    let a: Vec<f64> = pos
        .iter()
        .map(|&p| {
            let ahead = (p - pos[c]).rem_euclid(total);
            let x = if ahead <= total - ahead { ahead } else { ahead - total };
            if flipd { -x } else { x }
        })
        .collect();
    let mut half = [sh.half, sh.half]; // [left side (+1), right side (-1)]
    for (si, sd) in [1.0f64, -1.0].iter().enumerate() {
        if clip[si] {
            for i in 0..n {
                let sg = if a[i] > 0.0 { 1.0 } else if a[i] < 0.0 { -1.0 } else { 0.0 };
                if sg == *sd && sd * v[i] > u[i] {
                    half[si] = half[si].min(a[i].abs());
                }
            }
        }
    }
    let curl = spec().curl[t.var as usize];
    let tp = t.point;
    let mut out = Vec::with_capacity(n + 1);
    for i in 0..n {
        let side = if a[i] >= 0.0 { 1.0 } else { -1.0 };
        let hs = if side > 0.0 { half[0] } else { half[1] };
        let w = if a[i].abs() < hs { (1.0 - (a[i].abs() / hs).min(1.0)).powf(sh.q) } else { 0.0 };
        let lean = sh.open * side - curl * sh.bend;
        let p = d[i];
        let k = w * (1.0 - w) * lean;
        let q = (p.0 + w * (tp.0 - p.0) + k * l.0, p.1 + w * (tp.1 - p.1) + k * l.1);
        if i == c {
            out.push(tp); // the tip, twice: a sharp corner
            out.push(tp);
        } else {
            out.push(q);
        }
    }
    out
}

// ------------------------------------------------------------------ geometric check
pub struct Geo {
    pub ok: bool,
    pub n_strokes: usize,
    pub n_crossings: usize,
    pub min_dot: f64,
    pub min_strand: f64,
    pub length: f64,
}

fn seg_intersection(a0: P, a1: P, b0: P, b1: P) -> Option<P> {
    let r = (a1.0 - a0.0, a1.1 - a0.1);
    let s = (b1.0 - b0.0, b1.1 - b0.1);
    let den = r.0 * s.1 - r.1 * s.0;
    if den.abs() <= 1e-12 {
        return None;
    }
    let qp = (b0.0 - a0.0, b0.1 - a0.1);
    let t = (qp.0 * s.1 - qp.1 * s.0) / den;
    let u = (qp.0 * r.1 - qp.1 * r.0) / den;
    if (0.0..1.0).contains(&t) && (0.0..1.0).contains(&u) {
        Some((a0.0 + t * r.0, a0.1 + t * r.1))
    } else {
        None
    }
}

const MEET_R: f64 = 0.5;

/// Is this a proper kolam? (geometry.check_geometry, same thresholds.) With `full`, every stat is
/// computed; otherwise it stops at the first issue.
pub fn check(c: &Compiled, full: bool) -> Geo {
    let (n, dot_clear, strand_clear, cross_r, arc_skip) = (11usize, 0.4, 0.2, 0.3, 0.9);
    let k = c.k;
    let (rows, cols) = (k.rows, k.cols);
    let shown = drawn(c);
    let n_crossings = k.gates.iter().filter(|&&g| g == CROSS).count();
    let mut geo = Geo { ok: false, n_strokes: shown.len(), n_crossings, min_dot: f64::INFINITY, min_strand: f64::INFINITY, length: 0.0 };
    if shown.is_empty() {
        return geo;
    }
    let mut pts: Vec<P> = Vec::new();
    let mut sid: Vec<u32> = Vec::new();
    let mut arc: Vec<f64> = Vec::new();
    let mut nxt: Vec<u32> = Vec::new();
    let mut lengths: Vec<f64> = Vec::new();
    let mut ranges: Vec<(usize, usize)> = Vec::new();
    for (si, s) in shown.iter().enumerate() {
        let q = sample_stroke(s, per_segment(s, n));
        let off = pts.len();
        let m = q.len();
        let mut acc = 0.0;
        for i in 0..m {
            arc.push(acc);
            let (a, b) = (q[i], q[(i + 1) % m]);
            acc += ((b.0 - a.0).powi(2) + (b.1 - a.1).powi(2)).sqrt();
            sid.push(si as u32);
            nxt.push((off + (i + 1) % m) as u32);
        }
        lengths.push(acc);
        pts.extend_from_slice(&q);
        ranges.push((off, off + m));
    }
    geo.length = lengths.iter().sum();
    let mut ok = true;

    // 1. inside the lattice box (a crescent pointing off the grid may reach one unit past it)
    let m = 1.0 + 1e-6;
    if pts.iter().any(|p| p.0 < -m || p.1 < -m || p.0 > 2.0 * rows as f64 + m || p.1 > 2.0 * cols as f64 + m) {
        ok = false;
        if !full {
            return geo;
        }
    }

    // 2. never touch a drawn dot
    for p in &pts {
        let ni = (2.0 * round_even((p.0 - 1.0) / 2.0) + 1.0).clamp(1.0, (2 * rows - 1) as f64);
        let nj = (2.0 * round_even((p.1 - 1.0) / 2.0) + 1.0).clamp(1.0, (2 * cols - 1) as f64);
        let v = (((ni as i32) - 1) / 2, ((nj as i32) - 1) / 2);
        if k.used[(v.0 * cols + v.1) as usize] {
            let dd = ((p.0 - ni).powi(2) + (p.1 - nj).powi(2)).sqrt();
            geo.min_dot = geo.min_dot.min(dd);
        }
    }
    if geo.min_dot < dot_clear {
        ok = false;
        if !full {
            return geo;
        }
    }

    // 3. close pairs (spatial grid, cell = the pair radius)
    let rc = 0.45f64;
    let cell = |p: P| (((p.0 + 2.0) / rc).floor() as i32, ((p.1 + 2.0) / rc).floor() as i32);
    let mut grid: HashMap<(i32, i32), Vec<u32>> = HashMap::with_capacity(pts.len() / 2);
    for (i, &p) in pts.iter().enumerate() {
        grid.entry(cell(p)).or_default().push(i as u32);
    }
    let near_cross = |p: P| {
        let g = (p.0.round() as i32, p.1.round() as i32);
        k.gate(g) == CROSS && ((p.0 - g.0 as f64).powi(2) + (p.1 - g.1 as f64).powi(2)).sqrt() < cross_r
    };
    let mut meet_n: HashMap<V, u32> = HashMap::new();
    for t in c.tips.iter().filter(|t| t.long) {
        let s = horn_target(t.dot, t.dir);
        *meet_n.entry((2 * s.0 + 1, 2 * s.1 + 1)).or_default() += 1;
    }
    let meets: Vec<P> = meet_n.into_iter().filter(|&(_, n)| n > 1).map(|(p, _)| (p.0 as f64, p.1 as f64)).collect();
    let near_meet = |p: P| meets.iter().any(|m| ((p.0 - m.0).powi(2) + (p.1 - m.1).powi(2)).sqrt() < MEET_R);
    let mut xs: HashSet<(i64, i64)> = HashSet::new();
    let rc2 = rc * rc;
    for (i, &pi) in pts.iter().enumerate() {
        let (ci, cj) = cell(pi);
        for di in -1..=1 {
            for dj in -1..=1 {
                let Some(b) = grid.get(&(ci + di, cj + dj)) else { continue };
                for &j in b {
                    let j = j as usize;
                    if j <= i {
                        continue;
                    }
                    let pj = pts[j];
                    let d2 = (pi.0 - pj.0).powi(2) + (pi.1 - pj.1).powi(2);
                    if d2 >= rc2 {
                        continue;
                    }
                    let dist = d2.sqrt();
                    let local = sid[i] == sid[j] && {
                        let sep = (arc[i] - arc[j]).abs();
                        sep.min(lengths[sid[i] as usize] - sep) < arc_skip
                    };
                    let at_cross = (near_cross(pi) && near_cross(pj)) || (near_meet(pi) && near_meet(pj));
                    if !local && !at_cross {
                        geo.min_strand = geo.min_strand.min(dist);
                        if dist < strand_clear {
                            ok = false;
                            if !full {
                                return geo;
                            }
                        }
                    }
                    // 4. crossings: candidate segment pairs (not neighbours on one stroke)
                    let (ni, nj) = (nxt[i] as usize, nxt[j] as usize);
                    if ni == j || nj == i {
                        continue;
                    }
                    if let Some(x) = seg_intersection(pi, pts[ni], pj, pts[nj]) {
                        if !near_meet(x) {
                            xs.insert(((x.0 * 1e4).round_ties_even() as i64, (x.1 * 1e4).round_ties_even() as i64));
                        }
                    }
                }
            }
        }
    }
    // every intersection sits on a CROSS gap, exactly one per gap
    let mut counts: HashMap<V, u32> = HashMap::new();
    for &(a, b) in &xs {
        let x = (a as f64 / 1e4, b as f64 / 1e4);
        if near_cross(x) {
            *counts.entry((x.0.round() as i32, x.1.round() as i32)).or_default() += 1;
        } else {
            ok = false; // unplanned strand intersection
        }
    }
    if counts.len() != n_crossings || counts.values().any(|&n| n != 1) {
        ok = false;
    }
    if !ok && !full {
        return geo;
    }

    // 5. no corners except horn tips
    let cos50 = 50f64.to_radians().cos();
    'strokes: for &(a, b) in &ranges {
        let q = &pts[a..b];
        let m = q.len();
        for i in 0..m {
            let (p0, p, p1) = (q[(i + m - 1) % m], q[i], q[(i + 1) % m]);
            let v0 = (p.0 - p0.0, p.1 - p0.1);
            let v1 = (p1.0 - p.0, p1.1 - p.1);
            let cs = (v0.0 * v1.0 + v0.1 * v1.1) / ((v0.0.hypot(v0.1)) * (v1.0.hypot(v1.1)) + 1e-12);
            if cs < cos50 && !c.tips.iter().any(|t| ((p.0 - t.point.0).powi(2) + (p.1 - t.point.1).powi(2)).sqrt() < 0.05) {
                ok = false;
                break 'strokes;
            }
        }
    }
    geo.ok = ok;
    geo
}
