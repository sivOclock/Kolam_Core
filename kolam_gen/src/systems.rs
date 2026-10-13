//! The gate-grid generators (docs §11.2-11.4). Each sets gap states
//! directly, then adds crescent horns whole symmetric orbits at a time; the label is read off the gaps.

use crate::geometry::trace;
use crate::model::*;
use crate::rng::Rng;
use std::collections::HashSet;

const STATES: [u8; 3] = [SPLIT, CROSS, JOIN];

/// REQUIRED: the photographed sikku kolam (refs/photo_remake.png) as its kolam encoding.
/// wfc() learns all its dot pieces from it. Do not remove.
pub const WFC_SOURCE_ENCODING: [&str; 23] = [
    "00000000000", "000000000000", "00000100000", "000071150000", "00001110000", "000001100000",
    "00811011700", "000111111000", "00111011100", "000100001000", "01110101110", "011001100110",
    "01110101110", "000100001000", "00111011100", "000111111000", "00411011500", "000001100000",
    "00001110000", "000081140000", "00000100000", "000000000000", "00000000000",
];

fn lat_len(n: i32) -> usize {
    ((2 * n + 1) * (2 * n + 1)) as usize
}

fn li(n: i32, p: V) -> usize {
    (p.0 * (2 * n + 1) + p.1) as usize
}

fn get(n: i32, g: &[u8], p: V) -> u8 {
    if p.0 < 0 || p.1 < 0 || p.0 > 2 * n || p.1 > 2 * n { SPLIT } else { g[li(n, p)] }
}

/// The 4 gaps of dot v, in N E S W order.
fn dot_gates(v: V) -> [V; 4] {
    let (a, b) = (2 * v.0 + 1, 2 * v.1 + 1);
    [(a - 1, b), (a, b + 1), (a + 1, b), (a, b - 1)]
}

fn inner_gates(mask: &[bool], n: i32) -> Vec<V> {
    let m = |r: i32, c: i32| mask[(r * n + c) as usize];
    let mut out = Vec::new();
    for r in 0..n {
        for c in 0..n {
            if m(r, c) && c + 1 < n && m(r, c + 1) {
                out.push(gate_between((r, c), (r, c + 1)));
            }
            if m(r, c) && r + 1 < n && m(r + 1, c) {
                out.push(gate_between((r, c), (r + 1, c)));
            }
        }
    }
    out
}

fn orbits(points: &[V], ops: &[Op], n: i32) -> Vec<Vec<V>> {
    let mut seen: HashSet<V> = HashSet::new();
    let mut out = Vec::new();
    for &p in points {
        if seen.contains(&p) {
            continue;
        }
        let mut o: Vec<V> = ops.iter().map(|&op| lat_op(p, op, n, n)).collect();
        o.sort();
        o.dedup();
        seen.extend(o.iter().copied());
        out.push(o);
    }
    out
}

/// A D4-symmetric dot layout on an n x n grid: square, diamond, octagon or plus.
fn shape_mask(rng: &mut Rng, n: i32) -> (Vec<bool>, &'static str) {
    let kind = ["square", "diamond", "octagon", "plus"][rng.choice_p(&[0.3, 0.35, 0.2, 0.15])];
    let ctr = (n - 1) as f64 / 2.0;
    let mut m = vec![false; (n * n) as usize];
    for r in 0..n {
        for c in 0..n {
            let (dr, dc) = ((r as f64 - ctr).abs(), (c as f64 - ctr).abs());
            let d = dr + dc;
            m[(r * n + c) as usize] = match kind {
                "diamond" => d <= (n / 2) as f64,
                "octagon" => d <= (n / 2 + n / 4) as f64,
                "plus" => {
                    let w = (n / 6).max(1) as f64;
                    dr <= w + 0.5 || dc <= w + 0.5
                }
                _ => true,
            };
        }
    }
    (m, kind)
}

fn used_dots(n: i32, g: &[u8]) -> Vec<bool> {
    let mut u = vec![false; (n * n) as usize];
    for r in 0..n {
        for c in 0..n {
            u[(r * n + c) as usize] = dot_gates((r, c)).iter().any(|&p| get(n, g, p) != SPLIT);
        }
    }
    u
}

fn sym_pick(rng: &mut Rng) -> &'static str {
    ["d4", "d2", "c4"][rng.choice_p(&[0.5, 0.3, 0.2])]
}

/// Crescent horns on dots of the kolam pointing at empty spots or off the grid, whole symmetric
/// orbits at a time, keeping only those that pass every label rule.
fn add_horns(rng: &mut Rng, n: i32, g: &[u8], ops: &[Op], rate: f64) -> Vec<(V, u8, u8)> {
    let used = used_dots(n, g);
    let is_used = |v: V| v.0 >= 0 && v.0 < n && v.1 >= 0 && v.1 < n && used[(v.0 * n + v.1) as usize];
    let mut cands = Vec::new();
    for r in 0..n {
        for c in 0..n {
            if !used[(r * n + c) as usize] {
                continue;
            }
            for d in 0..4u8 {
                let t = horn_target((r, c), d);
                if !is_used(t) && get(n, g, gate_between((r, c), t)) == SPLIT {
                    cands.push(((r, c), d));
                }
            }
        }
    }
    let n_used = used.iter().filter(|&&u| u).count();
    let cap = ((0.12 * n_used as f64) as usize).max(1);
    let mut horns: Vec<Rel> = Vec::new();
    for i in rng.permutation(cands.len()) {
        if horns.len() >= cap || rng.random() > rate {
            continue;
        }
        let (v, d) = cands[i];
        let rel = Rel::horn(v, rng.below(3) as u8, d);
        let mut orb: Vec<Rel> = orbit(&rel, ops, n, n).into_iter().filter(|r| !horns.contains(r)).collect();
        orb.sort();
        let mut trial = horns.clone();
        trial.extend(orb);
        let hs: Vec<(V, u8, u8)> = trial.iter().map(|r| (r.vs[0], r.dir, r.var)).collect();
        if !horns_bad(&hs, |p| get(n, g, p), is_used) {
            horns = trial;
        }
    }
    horns.iter().map(|r| (r.vs[0], r.dir, r.var)).collect()
}

// ------------------------------------------------------------------ 1. substitution (2D L-system)
fn palindrome(rng: &mut Rng, k: usize, p: &[f64; 3]) -> Vec<u8> {
    let half: Vec<u8> = (0..(k + 1) / 2).map(|_| STATES[rng.choice_p(p)]).collect();
    let mut out = half.clone();
    out.extend(half[..k / 2].iter().rev());
    out
}

/// `max`: rewrite while the grid stays within max x max (12 for the dataset).
pub fn substitution(rng: &mut Rng, max: i32) -> Option<(Kolam, &'static str)> {
    let (s, k) = [(2, 2), (3, 2), (4, 2), (2, 3), (3, 3)][rng.below(5)];
    let ops = group("d4");
    let mut gates = vec![SPLIT; lat_len(s)];
    for o in orbits(&inner_gates(&vec![true; (s * s) as usize], s), ops, s) {
        let st = STATES[rng.choice_p(&[0.2, 0.5, 0.3])];
        for p in o {
            gates[li(s, p)] = st;
        }
    }
    let open_p = [0.15, 0.55, 0.3];
    let motif: Vec<Vec<u8>> = (0..5)
        .map(|_| {
            if k == 2 {
                vec![STATES[rng.choice_p(&open_p)]]
            } else {
                vec![STATES[rng.choice_p(&open_p)], STATES[rng.choice_p(&open_p)]]
            }
        })
        .collect();
    let seam = [
        palindrome(rng, k as usize, &[0.7, 0.15, 0.15]),
        palindrome(rng, k as usize, &[0.2, 0.55, 0.25]),
        palindrome(rng, k as usize, &[0.2, 0.25, 0.55]),
    ]; // by parent gap state SPLIT, CROSS, JOIN
    let mut n = s;
    while n * k <= max {
        gates = rewrite(&gates, n, k, &motif, &seam);
        n *= k;
    }
    let rate = rng.uniform(0.1, 0.5);
    let horns = add_horns(rng, n, &gates, ops, rate);
    Some((Kolam::from_gates(n, n, gates, horns), "d4"))
}

fn rewrite(gates: &[u8], n: i32, k: i32, motif: &[Vec<u8>], seam: &[Vec<u8>; 3]) -> Vec<u8> {
    let nn = n * k;
    let mut out = vec![SPLIT; lat_len(nn)];
    for r in 0..n {
        for c in 0..n {
            let deg = dot_gates((r, c)).iter().filter(|&&p| get(n, gates, p) != SPLIT).count();
            let m = &motif[deg];
            let (r0, c0) = (r * k, c * k);
            for i in 0..k {
                for j in 0..k {
                    let a = (r0 + i, c0 + j);
                    for b in [(r0 + i, c0 + j + 1), (r0 + i + 1, c0 + j)] {
                        if b.0 >= r0 + k || b.1 >= c0 + k {
                            continue;
                        }
                        let st = if k == 2 {
                            m[0]
                        } else if (a.0 - r0, a.1 - c0) == (1, 1) || (b.0 - r0, b.1 - c0) == (1, 1) {
                            m[0] // the 4 gaps touching the block's centre dot
                        } else {
                            m[1] // the 8 round its edge
                        };
                        out[li(nn, gate_between(a, b))] = st;
                    }
                }
            }
            if c + 1 < n {
                let p = &seam[get(n, gates, gate_between((r, c), (r, c + 1))) as usize];
                for i in 0..k {
                    out[li(nn, gate_between((r0 + i, c0 + k - 1), (r0 + i, c0 + k)))] = p[i as usize];
                }
            }
            if r + 1 < n {
                let p = &seam[get(n, gates, gate_between((r, c), (r + 1, c))) as usize];
                for j in 0..k {
                    out[li(nn, gate_between((r0 + k - 1, c0 + j), (r0 + k, c0 + j)))] = p[j as usize];
                }
            }
        }
    }
    out
}

// ------------------------------------------------------------------ 2. gate switching (one stroke)
/// `size`: the grid side (None: 5-11 at random, as for the dataset).
pub fn switching(rng: &mut Rng, size: Option<i32>) -> Option<(Kolam, &'static str)> {
    let n = size.unwrap_or_else(|| rng.integers(5, 12));
    let (mask, _shape) = shape_mask(rng, n);
    let sym = sym_pick(rng);
    let ops = group(sym);
    let inner = inner_gates(&mask, n);
    let mut gates = vec![SPLIT; lat_len(n)];
    for &p in &inner {
        gates[li(n, p)] = CROSS;
    }
    let orbs = orbits(&inner, ops, n);
    let join_p = [0.0, 0.15, 0.35][rng.below(3)];
    let target = if rng.random() < 0.75 { 1 } else { rng.integers(2, 5) as usize };
    let mut strokes = trace(n, n, &gates, &mask)?;
    let mut stale = 0;
    let mut owner: Vec<[i32; 2]> = vec![[-1, -1]; lat_len(n)];
    while strokes.len() > target && stale < 150 {
        owner.iter_mut().for_each(|o| *o = [-1, -1]);
        for (k, s) in strokes.iter().enumerate() {
            for &g in s {
                let o = &mut owner[li(n, g)];
                let k = k as i32;
                if o[0] == -1 {
                    o[0] = k;
                } else if o[0] != k && o[1] == -1 {
                    o[1] = k;
                }
            }
        }
        let cands: Vec<&Vec<V>> = orbs.iter().filter(|o| o.iter().any(|&g| owner[li(n, g)][1] != -1)).collect();
        if cands.is_empty() {
            break;
        }
        let o = cands[rng.below(cands.len())].clone();
        let cur = gates[li(n, o[0])];
        let new: Vec<u8> = STATES.iter().copied().filter(|&x| x != cur).collect();
        let st = if new.contains(&JOIN) && rng.random() < join_p {
            JOIN
        } else if new.contains(&SPLIT) {
            SPLIT
        } else {
            CROSS
        };
        let old: Vec<u8> = o.iter().map(|&g| gates[li(n, g)]).collect();
        for &g in &o {
            gates[li(n, g)] = st;
        }
        let alive = (0..n * n).all(|i| !mask[i as usize] || dot_gates((i / n, i % n)).iter().any(|&p| get(n, &gates, p) != SPLIT));
        let trial = if alive { trace(n, n, &gates, &mask) } else { None };
        match trial {
            Some(t) if t.len() < strokes.len() => {
                strokes = t;
                stale = 0;
            }
            _ => {
                for (&g, &x) in o.iter().zip(&old) {
                    gates[li(n, g)] = x;
                }
                stale += 1;
            }
        }
    }
    let rate = rng.uniform(0.0, 0.6);
    let horns = add_horns(rng, n, &gates, ops, rate);
    Some((Kolam::from_gates(n, n, gates, horns), sym))
}

// ------------------------------------------------------------------ 3. wave function collapse
pub type Tiles = (Vec<[u8; 4]>, Vec<f64>);

/// Dot tiles (N, E, S, W gap states) of the photographed sikku kolam with all 8 symmetries, counted.
pub fn photo_tiles() -> Tiles {
    let enc: Vec<String> = WFC_SOURCE_ENCODING.iter().map(|s| s.to_string()).collect();
    let k = Kolam::from_encoding(&enc).expect("WFC source encoding");
    let mut keys: Vec<[u8; 4]> = Vec::new();
    let mut counts: Vec<f64> = Vec::new();
    for r in 0..k.rows {
        for c in 0..k.cols {
            if !k.is_used((r, c)) {
                continue;
            }
            let g = dot_gates((r, c));
            let t = [k.gate(g[0]), k.gate(g[1]), k.gate(g[2]), k.gate(g[3])];
            for q in 0..4 {
                for tt in [t, [t[0], t[3], t[2], t[1]]] {
                    let mut key = tt;
                    key.rotate_right(q); // tt[-q:] + tt[:-q]
                    match keys.iter().position(|x| *x == key) {
                        Some(i) => counts[i] += 1.0,
                        None => {
                            keys.push(key);
                            counts.push(1.0);
                        }
                    }
                }
            }
        }
    }
    (keys, counts)
}

/// `size`: the grid side (None: 5-11 at random, as for the dataset).
pub fn wfc(rng: &mut Rng, tiles: &Tiles, size: Option<i32>) -> Option<(Kolam, &'static str)> {
    let (keys, w) = tiles;
    for _ in 0..80 {
        let n = size.unwrap_or_else(|| rng.integers(5, 12));
        let (mask, _shape) = shape_mask(rng, n);
        let sym = sym_pick(rng);
        let ops = group(sym);
        let dots: Vec<V> = (0..n * n).filter(|&i| mask[i as usize]).map(|i| (i / n, i % n)).collect();
        let mut free = vec![false; lat_len(n)];
        for p in inner_gates(&mask, n) {
            free[li(n, p)] = true;
        }
        let mut fixed = vec![u8::MAX; lat_len(n)];
        let mut tile_of = vec![false; (n * n) as usize];
        let mut n_tiled = 0;
        let options = |v: V, fixed: &[u8]| -> Vec<usize> {
            let gs = dot_gates(v);
            (0..keys.len())
                .filter(|&i| {
                    (0..4).all(|s| {
                        let p = gs[s];
                        let inb = p.0 >= 0 && p.1 >= 0 && p.0 <= 2 * n && p.1 <= 2 * n;
                        if !inb || !free[li(n, p)] {
                            keys[i][s] == SPLIT
                        } else {
                            let f = fixed[li(n, p)];
                            f == u8::MAX || f == keys[i][s]
                        }
                    })
                })
                .collect()
        };
        let mut ok = true;
        while ok && n_tiled < dots.len() {
            let mut best: Option<(usize, f64, V, Vec<usize>)> = None;
            for &v in &dots {
                if tile_of[(v.0 * n + v.1) as usize] {
                    continue;
                }
                let o = options(v, &fixed);
                if o.is_empty() {
                    ok = false;
                    break;
                }
                let key = (o.len(), rng.random()); // least freedom first, random tie-break
                if best.as_ref().map_or(true, |b| (key.0, key.1) < (b.0, b.1)) {
                    best = Some((key.0, key.1, v, o));
                }
            }
            if !ok {
                break;
            }
            let (_, _, v, o) = best.unwrap();
            let ww: Vec<f64> = o.iter().map(|&i| w[i]).collect();
            let t = keys[o[rng.choice_p(&ww)]];
            for (s, &g) in dot_gates(v).iter().enumerate() {
                for &op in ops {
                    let gg = lat_op(g, op, n, n);
                    if gg.0 >= 0 && gg.1 >= 0 && gg.0 <= 2 * n && gg.1 <= 2 * n && free[li(n, gg)] {
                        let f = &mut fixed[li(n, gg)];
                        if *f == u8::MAX {
                            *f = t[s];
                        } else if *f != t[s] {
                            ok = false;
                        }
                    }
                }
            }
            for &op in ops {
                let (a, b) = lat_op((2 * v.0 + 1, 2 * v.1 + 1), op, n, n);
                let i = (((a - 1) / 2) * n + (b - 1) / 2) as usize;
                if !tile_of[i] {
                    tile_of[i] = true;
                    n_tiled += 1;
                }
            }
        }
        if !ok {
            continue;
        }
        let gates: Vec<u8> = fixed.iter().map(|&f| if f == u8::MAX { SPLIT } else { f }).collect();
        if used_dots(n, &gates) != mask {
            continue;
        }
        let rate = rng.uniform(0.0, 0.6);
        let horns = add_horns(rng, n, &gates, ops, rate);
        return Some((Kolam::from_gates(n, n, gates, horns), sym));
    }
    None
}
