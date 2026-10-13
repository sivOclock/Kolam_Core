//! Kolam model: brush spec, relations, symmetry, label rules, the gate grid and the kolam encoding.
//! Port of kolam_press/spec.py, hypergraph.py and encoding.py (docs/hypergraph_kolams.md §1-3, §8, §12).
//! Python is the source of truth; `kolam_gen check` and tests/test_kolam_gen.py keep the two in step.

use std::collections::HashMap;
use std::sync::OnceLock;

pub type V = (i32, i32); // anchor dot (row, col), or lattice point (li, lj)
pub type P = (f64, f64); // lattice-space point

pub const SPLIT: u8 = 0;
pub const CROSS: u8 = 1;
pub const JOIN: u8 = 2;

#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Debug)]
pub enum Ty {
    Cross = 0, // CROSSING_LOOP
    Par = 1,   // PARALLEL_LOOP
    Horn = 2,  // HORN
}

impl Ty {
    pub fn state(self) -> u8 {
        match self {
            Ty::Cross => CROSS,
            Ty::Par => JOIN,
            Ty::Horn => SPLIT,
        }
    }
}

/// Horn directions N E S W (where the tip points), as in spec.HORN_DIRS.
pub const DIRS: [V; 4] = [(-1, 0), (0, 1), (1, 0), (0, -1)];
/// Horn variants by encoding order: 0 PLAIN, 1 LEFT, 2 RIGHT (spec.HORN_VARIANTS).
/// Python sorts labels by name: LEFT < PLAIN < RIGHT, and E < N < S < W.
const VAR_RANK: [u8; 3] = [1, 0, 2];
const DIR_RANK: [u8; 4] = [1, 0, 2, 3];

pub fn left_of(d: V) -> V {
    (-d.1, d.0)
}

pub fn gate_between(a: V, b: V) -> V {
    (a.0 + b.0 + 1, a.1 + b.1 + 1)
}

pub fn horn_target(v: V, dir: u8) -> V {
    let f = DIRS[dir as usize];
    (v.0 + f.0, v.1 + f.1)
}

pub fn is_h_gate(p: V) -> bool {
    p.0 % 2 != 0 && p.1 % 2 == 0
}

pub fn adjacent(a: V, b: V) -> bool {
    (a.0 - b.0).abs() + (a.1 - b.1).abs() == 1
}

// ------------------------------------------------------------------ brush spec (brushes.json)
pub struct Shape {
    pub reach: f64,
    pub half: f64,
    pub q: f64,
    pub open: f64,
    pub curl: f64,
    pub bend: f64,
}

pub struct Spec {
    pub long: Shape,
    pub short: Shape,
    pub curl: [f64; 3],        // by variant: PLAIN 0, LEFT +1, RIGHT -1
    pub flip: [[V; 2]; 3],     // by gate state, [h_gate, v_gate]
}

pub fn spec() -> &'static Spec {
    static S: OnceLock<Spec> = OnceLock::new();
    S.get_or_init(|| {
        let j: serde_json::Value =
            serde_json::from_str(include_str!("../../kolam_press/brushes.json")).expect("brushes.json");
        let horn = &j["relations"]["HORN"];
        let shape = |k: &str| {
            let g = &horn["grow"][k];
            let f = |n: &str| g[n].as_f64().unwrap();
            Shape { reach: f("reach"), half: f("half"), q: f("q"), open: f("open"), curl: f("curl"), bend: f("bend") }
        };
        let var = |n: &str| horn["variants"][n].as_f64().unwrap();
        let fl = |s: &str, k: &str| {
            let a = &j["gate_states"][s][k];
            (a[0].as_i64().unwrap() as i32, a[1].as_i64().unwrap() as i32)
        };
        let fs = |s: &str| [fl(s, "h_gate"), fl(s, "v_gate")];
        for (i, d) in ["N", "E", "S", "W"].iter().enumerate() {
            let a = &horn["dirs"][d];
            assert_eq!((a[0].as_i64().unwrap() as i32, a[1].as_i64().unwrap() as i32), DIRS[i], "brushes.json dirs");
        }
        Spec {
            long: shape("long"),
            short: shape("short"),
            curl: [var("PLAIN_HORN"), var("LEFT_HORN"), var("RIGHT_HORN")],
            flip: [fs("SPLIT"), fs("CROSS"), fs("JOIN")],
        }
    })
}

pub fn shape(long: bool) -> &'static Shape {
    if long { &spec().long } else { &spec().short }
}

/// Lattice point of a horn's tip: reach along fwd, turned by the curl (spec.horn_tip).
pub fn horn_tip(v: V, dir: u8, var: u8, long: bool) -> P {
    let o = (2 * v.0 + 1, 2 * v.1 + 1);
    let f = DIRS[dir as usize];
    let l = left_of(f);
    let sh = shape(long);
    let a = spec().curl[var as usize] * sh.curl;
    let (cu, cv) = (sh.reach * a.cos(), sh.reach * a.sin());
    (o.0 as f64 + cu * f.0 as f64 + cv * l.0 as f64, o.1 as f64 + cu * f.1 as f64 + cv * l.1 as f64)
}

// ------------------------------------------------------------------ relations
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct Rel {
    pub ty: Ty,
    pub vs: Vec<V>,
    pub var: u8,     // horns only (0 for loops)
    pub closed: bool,
    pub dir: u8,     // horns only (0 for loops)
}

impl Rel {
    pub fn lp(ty: Ty, vs: Vec<V>, closed: bool) -> Rel {
        Rel { ty, vs, var: 0, closed, dir: 0 }
    }

    pub fn horn(v: V, var: u8, dir: u8) -> Rel {
        Rel { ty: Ty::Horn, vs: vec![v], var, closed: false, dir }
    }

    /// Lines from their smaller end; closed rings rotated/reflected to their smallest form.
    pub fn canonical(&self) -> Rel {
        if self.ty == Ty::Horn {
            return self.clone();
        }
        let vs = &self.vs;
        let mut rev = vs.clone();
        rev.reverse();
        let best = if !self.closed {
            if *vs <= rev { vs.clone() } else { rev }
        } else {
            let k = vs.len();
            let mut best = vs.clone();
            for seq in [vs, &rev] {
                for i in 0..k {
                    let rot: Vec<V> = seq[i..].iter().chain(seq[..i].iter()).copied().collect();
                    if rot < best {
                        best = rot;
                    }
                }
            }
            best
        };
        Rel::lp(self.ty, best, self.closed)
    }

    pub fn edges(&self) -> Vec<(V, V)> {
        let vs = &self.vs;
        let mut e: Vec<(V, V)> = vs.windows(2).map(|w| (w[0], w[1])).collect();
        if self.closed && vs.len() > 1 {
            e.push((vs[vs.len() - 1], vs[0]));
        }
        e
    }

    fn key(&self) -> (Ty, u8, bool, &Vec<V>, u8) {
        if self.ty == Ty::Horn {
            (self.ty, VAR_RANK[self.var as usize] + 1, self.closed, &self.vs, DIR_RANK[self.dir as usize] + 1)
        } else {
            (self.ty, 0, self.closed, &self.vs, 0)
        }
    }
}

impl PartialOrd for Rel {
    fn partial_cmp(&self, o: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(o))
    }
}

impl Ord for Rel {
    fn cmp(&self, o: &Self) -> std::cmp::Ordering {
        self.key().cmp(&o.key())
    }
}

// ------------------------------------------------------------------ symmetry
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Op {
    Id,
    MirLR,
    MirUD,
    Rot180,
    Rot90,
    Rot270,
    Tr,
    Anti,
}

pub const SYM_NAMES: [&str; 7] = ["none", "mirror_lr", "mirror_ud", "rot180", "d2", "c4", "d4"];

pub fn group(name: &str) -> &'static [Op] {
    use Op::*;
    match name {
        "none" => &[Id],
        "mirror_lr" => &[Id, MirLR],
        "mirror_ud" => &[Id, MirUD],
        "rot180" => &[Id, Rot180],
        "d2" => &[Id, MirLR, MirUD, Rot180],
        "c4" => &[Id, Rot90, Rot180, Rot270],
        "d4" => &[Id, Rot90, Rot180, Rot270, MirLR, MirUD, Tr, Anti],
        _ => panic!("unknown symmetry {name}"),
    }
}

pub fn transform_vertex(v: V, op: Op, rows: i32, cols: i32) -> V {
    let (r, c) = v;
    match op {
        Op::Id => (r, c),
        Op::MirLR => (r, cols - 1 - c),
        Op::MirUD => (rows - 1 - r, c),
        Op::Rot180 => (rows - 1 - r, cols - 1 - c),
        Op::Rot90 => (c, rows - 1 - r),
        Op::Rot270 => (cols - 1 - c, r),
        Op::Tr => (c, r),
        Op::Anti => (cols - 1 - c, rows - 1 - r),
    }
}

/// A lattice point under a symmetry of the rows x cols dot grid (systems.lat_op).
pub fn lat_op(p: V, op: Op, rows: i32, cols: i32) -> V {
    let (rr, cc) = (2 * rows, 2 * cols);
    let (i, j) = p;
    match op {
        Op::Id => (i, j),
        Op::MirLR => (i, cc - j),
        Op::MirUD => (rr - i, j),
        Op::Rot180 => (rr - i, cc - j),
        Op::Rot90 => (j, rr - i),
        Op::Rot270 => (cc - j, i),
        Op::Tr => (j, i),
        Op::Anti => (cc - j, rr - i),
    }
}

fn is_reflection(op: Op) -> bool {
    matches!(op, Op::MirLR | Op::MirUD | Op::Tr | Op::Anti)
}

pub fn transform_relation(r: &Rel, op: Op, rows: i32, cols: i32) -> Rel {
    let vs: Vec<V> = r.vs.iter().map(|&v| transform_vertex(v, op, rows, cols)).collect();
    if r.ty == Ty::Horn {
        let var = if is_reflection(op) && r.var != 0 { 3 - r.var } else { r.var }; // LEFT <-> RIGHT
        let b = transform_vertex(horn_target(r.vs[0], r.dir), op, rows, cols);
        let d = (b.0 - vs[0].0, b.1 - vs[0].1);
        let dir = DIRS.iter().position(|&x| x == d).unwrap() as u8;
        return Rel::horn(vs[0], var, dir);
    }
    Rel::lp(r.ty, vs, r.closed).canonical()
}

/// The distinct images of a relation under a symmetry group.
pub fn orbit(r: &Rel, ops: &[Op], rows: i32, cols: i32) -> Vec<Rel> {
    let mut out: Vec<Rel> = Vec::with_capacity(ops.len());
    for &op in ops {
        let t = transform_relation(r, op, rows, cols);
        if !out.contains(&t) {
            out.push(t);
        }
    }
    out
}

// ------------------------------------------------------------------ label rules (hypergraph.problems)
/// Horn rules: one horn per (dot, dir); it points through a wall at an empty spot or off the grid (a
/// crescent); tips stay apart unless they meet in one empty spot, curling the same way, and not from
/// one inner corner of a parallel tube. `gate` gives the state of a lattice gap set by the loops.
pub fn horns_bad(horns: &[(V, u8, u8)], gate: impl Fn(V) -> u8, used: impl Fn(V) -> bool) -> bool {
    let mut seen: Vec<(V, u8)> = Vec::with_capacity(horns.len());
    let mut tips: Vec<(P, Option<V>)> = Vec::with_capacity(horns.len());
    let mut meets: HashMap<V, Vec<(V, u8)>> = HashMap::new();
    for &(v, d, var) in horns {
        if seen.contains(&(v, d)) {
            return true;
        }
        seen.push((v, d));
        let t = horn_target(v, d);
        if gate(gate_between(v, t)) != SPLIT {
            return true; // points along a brush line
        }
        let long = !used(t);
        if !long {
            return true; // a short stub: not a dataset label
        }
        let p = horn_tip(v, d, var, long);
        for &(q, tj) in &tips {
            if long && tj == Some(t) {
                continue; // meeting in one empty spot
            }
            if (p.0 - q.0).powi(2) + (p.1 - q.1).powi(2) <= 1.0 {
                return true;
            }
        }
        tips.push((p, if long { Some(t) } else { None }));
        let m = meets.entry(t).or_default();
        for &(u, var2) in m.iter() {
            if var2 != var {
                return true;
            }
            let dd = (v.0 + u.0 - t.0, v.1 + u.1 - t.1); // the dot across the corner
            if dd != t && gate(gate_between(u, dd)) == JOIN && gate(gate_between(v, dd)) == JOIN {
                return true;
            }
        }
        m.push((v, var));
    }
    false
}

/// Does this list of relations break any label rule (problems() non-empty, any level)?
pub fn problems(rows: i32, cols: i32, rels: &[Rel]) -> bool {
    if !(1..=32).contains(&rows) || !(1..=32).contains(&cols) {
        return true;
    }
    let lw = (2 * cols + 1) as usize;
    let mut gs = vec![SPLIT; ((2 * rows + 1) * (2 * cols + 1)) as usize];
    let mut used = vec![false; (rows * cols) as usize];
    let inside = |v: V| v.0 >= 0 && v.0 < rows && v.1 >= 0 && v.1 < cols;
    let mut horns = Vec::new();
    for r in rels {
        if r.vs.iter().any(|&v| !inside(v)) {
            return true;
        }
        for (i, a) in r.vs.iter().enumerate() {
            if r.vs[..i].contains(a) {
                return true; // repeated vertex
            }
        }
        for &v in &r.vs {
            used[(v.0 * cols + v.1) as usize] = true;
        }
        if r.ty == Ty::Horn {
            if r.vs.len() != 1 || r.closed {
                return true;
            }
            horns.push((r.vs[0], r.dir, r.var));
            continue;
        }
        if r.vs.len() < 2 {
            return true;
        }
        if r.closed && (r.vs.len() < 4 || !adjacent(r.vs[r.vs.len() - 1], r.vs[0])) {
            return true;
        }
        let st = r.ty.state();
        for (a, b) in r.edges() {
            if !adjacent(a, b) {
                return true;
            }
            let g = gate_between(a, b);
            let k = g.0 as usize * lw + g.1 as usize;
            if gs[k] != SPLIT && gs[k] != st {
                return true; // one gap given two states
            }
            gs[k] = st;
        }
    }
    let gate = |p: V| {
        if p.0 < 0 || p.1 < 0 || p.0 > 2 * rows || p.1 > 2 * cols { SPLIT } else { gs[p.0 as usize * lw + p.1 as usize] }
    };
    let usedf = |v: V| inside(v) && used[(v.0 * cols + v.1) as usize];
    horns_bad(&horns, gate, usedf)
}

/// (is the brush-line graph a forest?) For a forest every piece compiles to one closed stroke.
pub fn is_forest(rows: i32, cols: i32, rels: &[Rel]) -> bool {
    let mut parent: Vec<usize> = (0..(rows * cols) as usize).collect();
    fn find(p: &mut [usize], mut x: usize) -> usize {
        while p[x] != x {
            p[x] = p[p[x]];
            x = p[x];
        }
        x
    }
    for r in rels {
        for (a, b) in r.edges() {
            let (ra, rb) = (find(&mut parent, (a.0 * cols + a.1) as usize), find(&mut parent, (b.0 * cols + b.1) as usize));
            if ra == rb {
                return false;
            }
            parent[ra] = rb;
        }
    }
    true
}

// ------------------------------------------------------------------ a kolam: its gap states + horns
/// What a picture shows and the label is read from: the state of every gap and the horns.
#[derive(Clone, Debug)]
pub struct Kolam {
    pub rows: i32,
    pub cols: i32,
    pub gates: Vec<u8>,          // (2*rows+1) x (2*cols+1) lattice, SPLIT/CROSS/JOIN on gaps
    pub horns: Vec<(V, u8, u8)>, // (dot, dir, variant), in label order
    pub used: Vec<bool>,         // rows x cols: dots some relation uses (the only dots drawn)
}

impl Kolam {
    pub fn lw(&self) -> usize {
        (2 * self.cols + 1) as usize
    }

    pub fn gate(&self, p: V) -> u8 {
        if p.0 < 0 || p.1 < 0 || p.0 > 2 * self.rows || p.1 > 2 * self.cols {
            return SPLIT;
        }
        self.gates[p.0 as usize * self.lw() + p.1 as usize]
    }

    pub fn is_used(&self, v: V) -> bool {
        v.0 >= 0 && v.0 < self.rows && v.1 >= 0 && v.1 < self.cols && self.used[(v.0 * self.cols + v.1) as usize]
    }

    fn sort_horns(horns: &mut Vec<(V, u8, u8)>) {
        horns.sort_by_key(|&(v, d, var)| (VAR_RANK[var as usize], v, DIR_RANK[d as usize]));
        horns.dedup();
    }

    pub fn from_rels(rows: i32, cols: i32, rels: &[Rel]) -> Kolam {
        let lw = (2 * cols + 1) as usize;
        let mut gates = vec![SPLIT; ((2 * rows + 1) * (2 * cols + 1)) as usize];
        let mut used = vec![false; (rows * cols) as usize];
        let mut horns = Vec::new();
        for r in rels {
            for &v in &r.vs {
                used[(v.0 * cols + v.1) as usize] = true;
            }
            if r.ty == Ty::Horn {
                horns.push((r.vs[0], r.dir, r.var));
            } else {
                for (a, b) in r.edges() {
                    let g = gate_between(a, b);
                    let k = g.0 as usize * lw + g.1 as usize;
                    if gates[k] == SPLIT {
                        gates[k] = r.ty.state();
                    }
                }
            }
        }
        Self::sort_horns(&mut horns);
        Kolam { rows, cols, gates, horns, used }
    }

    /// From gap states alone (+ horns): a dot is used when one of its gaps is not a wall.
    pub fn from_gates(rows: i32, cols: i32, gates: Vec<u8>, mut horns: Vec<(V, u8, u8)>) -> Kolam {
        let mut k = Kolam { rows, cols, gates, horns: Vec::new(), used: vec![false; (rows * cols) as usize] };
        for r in 0..rows {
            for c in 0..cols {
                let (li, lj) = (2 * r + 1, 2 * c + 1);
                k.used[(r * cols + c) as usize] =
                    [(li - 1, lj), (li + 1, lj), (li, lj - 1), (li, lj + 1)].iter().any(|&g| k.gate(g) != SPLIT);
            }
        }
        for &(v, _, _) in &horns {
            k.used[(v.0 * cols + v.1) as usize] = true;
        }
        Self::sort_horns(&mut horns);
        k.horns = horns;
        k
    }

    /// Drop border rows/cols with no kolam on them (unused dots are not drawn). None if empty.
    pub fn crop(&self) -> Option<Kolam> {
        let (mut r0, mut r1, mut c0, mut c1) = (i32::MAX, -1, i32::MAX, -1);
        for r in 0..self.rows {
            for c in 0..self.cols {
                if self.used[(r * self.cols + c) as usize] {
                    r0 = r0.min(r);
                    r1 = r1.max(r);
                    c0 = c0.min(c);
                    c1 = c1.max(c);
                }
            }
        }
        if r1 < 0 {
            return None;
        }
        let (rows, cols) = (r1 - r0 + 1, c1 - c0 + 1);
        let lw = (2 * cols + 1) as usize;
        let mut gates = vec![SPLIT; ((2 * rows + 1) * (2 * cols + 1)) as usize];
        for i in 0..=2 * rows {
            for j in 0..=2 * cols {
                gates[i as usize * lw + j as usize] = self.gate((i + 2 * r0, j + 2 * c0));
            }
        }
        let mut used = vec![false; (rows * cols) as usize];
        for r in 0..rows {
            for c in 0..cols {
                used[(r * cols + c) as usize] = self.used[((r + r0) * self.cols + c + c0) as usize];
            }
        }
        let mut horns: Vec<_> = self.horns.iter().map(|&(v, d, var)| ((v.0 - r0, v.1 - c0), d, var)).collect();
        Self::sort_horns(&mut horns);
        Some(Kolam { rows, cols, gates, horns, used })
    }

    /// The kolam encoding (spec §12): gap rows of the lattice, cols and cols+1 digits alternating.
    pub fn encoding(&self) -> Vec<String> {
        let mut g = self.gates.clone();
        let lw = self.lw();
        for &(v, d, var) in &self.horns {
            let p = gate_between(v, horn_target(v, d));
            g[p.0 as usize * lw + p.1 as usize] = 3 + var + if d == 1 || d == 2 { 0 } else { 3 };
        }
        (0..=2 * self.rows as usize)
            .map(|i| {
                let start = (i + 1) % 2;
                (start..lw).step_by(2).map(|j| (b'0' + g[i * lw + j]) as char).collect()
            })
            .collect()
    }

    pub fn from_encoding(enc: &[String]) -> Result<Kolam, String> {
        if enc.len() < 3 || enc.len() % 2 == 0 {
            return Err(format!("an encoding has 2*rows+1 rows, not {}", enc.len()));
        }
        let rows = ((enc.len() - 1) / 2) as i32;
        let cols = enc[0].len() as i32;
        let lw = (2 * cols + 1) as usize;
        let mut gates = vec![SPLIT; ((2 * rows + 1) * (2 * cols + 1)) as usize];
        let mut horns = Vec::new();
        for (i, row) in enc.iter().enumerate() {
            if row.len() as i32 != cols + (i % 2) as i32 {
                return Err(format!("encoding row {i} has {} digits", row.len()));
            }
            for (k, ch) in row.bytes().enumerate() {
                let x = ch.wrapping_sub(b'0');
                if x > 8 {
                    return Err(format!("bad digit in row {i}"));
                }
                let i = i as i32;
                let j = if i % 2 == 1 { 2 * k as i32 } else { 2 * k as i32 + 1 };
                if x == 0 {
                    continue;
                }
                if x <= 2 {
                    gates[i as usize * lw + j as usize] = x;
                    continue;
                }
                let (a, b) = if i % 2 == 1 {
                    (((i - 1) / 2, j / 2 - 1), ((i - 1) / 2, j / 2))
                } else {
                    ((i / 2 - 1, (j - 1) / 2), (i / 2, (j - 1) / 2))
                };
                let fwd = x <= 5;
                let dir = if i % 2 == 1 { if fwd { 1 } else { 3 } } else if fwd { 2 } else { 0 };
                horns.push((if fwd { a } else { b }, dir, (x - 3) % 3));
            }
        }
        Ok(Kolam::from_gates(rows, cols, gates, horns))
    }
}
