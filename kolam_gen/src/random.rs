//! The random painter (docs §11.1); first written in Python (sample_hypergraph, now retired).
//! grid -> symmetric brush strokes (kept only when every label rule holds) -> fill bare dots -> crop.

use crate::model::*;
use crate::rng::Rng;

const STEPS: [V; 4] = [(0, 1), (1, 0), (0, -1), (-1, 0)];
const TYPES: [Ty; 3] = [Ty::Cross, Ty::Par, Ty::Horn];

#[derive(Clone)]
pub struct GenCfg {
    pub min_size: i32,
    pub max_size: i32,
    pub square_prob: f64,
    pub symmetry: [f64; 7], // SYM_NAMES order
    pub brush: [f64; 3],    // CROSSING_LOOP, PARALLEL_LOOP, HORN
    pub tree_prob: f64,
    pub pure_prob: f64,
    pub ring_prob: f64,
    pub mix_prob: f64,
    pub share_prob: f64,
    pub straight_bias: f64,
    pub max_horn_frac: f64,
    pub max_chain: i32,
    pub coverage: (f64, f64),
    pub fill_prob: f64,
    pub max_proposals: usize,
}

impl Default for GenCfg {
    fn default() -> Self {
        GenCfg {
            min_size: 3,
            max_size: 8,
            square_prob: 0.5,
            symmetry: [0.25, 0.15, 0.1, 0.1, 0.2, 0.1, 0.1],
            brush: [0.4, 0.3, 0.3],
            tree_prob: 0.4,
            pure_prob: 0.15,
            ring_prob: 0.15,
            mix_prob: 0.3,
            share_prob: 0.35,
            straight_bias: 0.65,
            max_horn_frac: 0.2,
            max_chain: 8,
            coverage: (0.7, 1.0),
            fill_prob: 0.85,
            max_proposals: 300,
        }
    }
}

struct St {
    rows: i32,
    cols: i32,
    tree: bool,
    rels: Vec<Rel>,
    gates: Vec<bool>,        // lattice: gaps some brush line uses
    members: [Vec<bool>; 2], // dots of crossing / parallel loops
    horns: Vec<(V, u8)>,
    used: Vec<bool>,
    n_used: usize,
}

impl St {
    fn new(rows: i32, cols: i32, tree: bool) -> St {
        let n = (rows * cols) as usize;
        St {
            rows,
            cols,
            tree,
            rels: Vec::new(),
            gates: vec![false; ((2 * rows + 1) * (2 * cols + 1)) as usize],
            members: [vec![false; n], vec![false; n]],
            horns: Vec::new(),
            used: vec![false; n],
            n_used: 0,
        }
    }

    fn vi(&self, v: V) -> usize {
        (v.0 * self.cols + v.1) as usize
    }

    fn inside(&self, v: V) -> bool {
        v.0 >= 0 && v.0 < self.rows && v.1 >= 0 && v.1 < self.cols
    }

    fn is_used(&self, v: V) -> bool {
        self.inside(v) && self.used[self.vi(v)]
    }

    fn gate(&self, p: V) -> bool {
        p.0 >= 0 && p.1 >= 0 && p.0 <= 2 * self.rows && p.1 <= 2 * self.cols
            && self.gates[(p.0 * (2 * self.cols + 1) + p.1) as usize]
    }

    fn add(&mut self, rels: Vec<Rel>) {
        let lw = 2 * self.cols + 1;
        for r in rels {
            for &v in &r.vs {
                let i = self.vi(v);
                if !self.used[i] {
                    self.used[i] = true;
                    self.n_used += 1;
                }
            }
            for (a, b) in r.edges() {
                let g = gate_between(a, b);
                self.gates[(g.0 * lw + g.1) as usize] = true;
            }
            match r.ty {
                Ty::Horn => self.horns.push((r.vs[0], r.dir)),
                t => {
                    for &v in &r.vs {
                        let i = self.vi(v);
                        self.members[t as usize][i] = true;
                    }
                }
            }
            self.rels.push(r);
        }
    }

    fn rebuild(&mut self, rels: Vec<Rel>) {
        *self = St::new(self.rows, self.cols, self.tree);
        self.add(rels);
    }

    fn coverage(&self) -> f64 {
        self.n_used as f64 / (self.rows * self.cols) as f64
    }

    fn loop_ok(&self, t: Ty, mix: bool, share: bool, v: V) -> bool {
        let other = if t == Ty::Cross { Ty::Par } else { Ty::Cross };
        self.inside(v)
            && (share || !self.members[t as usize][self.vi(v)])
            && (mix || !self.members[other as usize][self.vi(v)])
    }

    /// Connected piece of every dot of the kolam (root id), -1 for bare dots.
    fn pieces(&self) -> Vec<i64> {
        let n = (self.rows * self.cols) as usize;
        let mut parent: Vec<usize> = (0..n).collect();
        fn find(p: &mut [usize], mut x: usize) -> usize {
            while p[x] != x {
                p[x] = p[p[x]];
                x = p[x];
            }
            x
        }
        for r in &self.rels {
            for (a, b) in r.edges() {
                let (ra, rb) = (find(&mut parent, self.vi(a)), find(&mut parent, self.vi(b)));
                parent[ra] = rb;
            }
        }
        (0..n).map(|i| if self.used[i] { find(&mut parent, i) as i64 } else { -1 }).collect()
    }
}

fn try_set(st: &mut St, ops: &[Op], add: &Rel, remove: Option<&Rel>) -> bool {
    let mut keep = st.rels.clone();
    if let Some(rm) = remove {
        let gone = orbit(rm, ops, st.rows, st.cols);
        keep.retain(|r| !gone.contains(r));
    }
    let mut new: Vec<Rel> = orbit(add, ops, st.rows, st.cols).into_iter().filter(|r| !keep.contains(r)).collect();
    new.sort();
    let mut trial = keep;
    trial.extend(new.iter().cloned());
    if problems(st.rows, st.cols, &trial) {
        return false;
    }
    if st.tree && !is_forest(st.rows, st.cols, &trial) {
        return false;
    }
    if remove.is_none() {
        st.add(new);
    } else {
        st.rebuild(trial);
    }
    true
}

fn walk(rng: &mut Rng, st: &St, start: V, want: usize, ok: &dyn Fn(V) -> bool, cfg: &GenCfg, piece: &[i64]) -> Vec<V> {
    let mut path = vec![start];
    let mut d: Option<usize> = None;
    while path.len() < want {
        let cur = *path.last().unwrap();
        let mut cands: Vec<usize> = Vec::with_capacity(4);
        for (si, s) in STEPS.iter().enumerate() {
            let nx = (cur.0 + s.0, cur.1 + s.1);
            if !(ok(nx) && !path.contains(&nx) && !st.gate(gate_between(cur, nx))) {
                continue;
            }
            // tree mode: only bare dots, except one final step that joins a different piece
            if st.tree && st.is_used(nx) && piece[st.vi(nx)] == piece[st.vi(start)] {
                continue;
            }
            cands.push(si);
        }
        if cands.is_empty() {
            break;
        }
        let s = match d {
            Some(dd) if cands.contains(&dd) && rng.random() < cfg.straight_bias => dd,
            _ => cands[rng.below(cands.len())],
        };
        let nx = (cur.0 + STEPS[s].0, cur.1 + STEPS[s].1);
        path.push(nx);
        d = Some(s);
        if st.tree && st.is_used(nx) {
            break;
        }
    }
    path
}

fn propose_ring(rng: &mut Rng, st: &St, ok: &dyn Fn(V) -> bool, cfg: &GenCfg) -> Option<Vec<V>> {
    for _ in 0..10 {
        let h = rng.integers(2, st.rows.min(cfg.max_chain / 2 + 1) + 1);
        let w = rng.integers(2, st.cols.min(cfg.max_chain / 2 + 1) + 1);
        if h > st.rows || w > st.cols {
            continue;
        }
        let r0 = rng.integers(0, st.rows - h + 1);
        let c0 = rng.integers(0, st.cols - w + 1);
        let mut ring: Vec<V> = (c0..c0 + w).map(|c| (r0, c)).collect();
        ring.extend((r0 + 1..r0 + h).map(|r| (r, c0 + w - 1)));
        ring.extend((c0..=c0 + w - 2).rev().map(|c| (r0 + h - 1, c)));
        ring.extend((r0 + 1..=r0 + h - 2).rev().map(|r| (r, c0)));
        let k = ring.len();
        if ring.iter().all(|&v| ok(v)) && (0..k).all(|i| !st.gate(gate_between(ring[i], ring[(i + 1) % k]))) {
            return Some(ring);
        }
    }
    None
}

fn propose_chain(rng: &mut Rng, st: &St, t: Ty, cfg: &GenCfg) -> Option<Rel> {
    let attach = st.tree && st.n_used > 0;
    let mix = attach || rng.random() < cfg.mix_prob;
    let share = rng.random() < cfg.share_prob;
    let ok = |v: V| st.loop_ok(t, mix, share, v);
    let mut pool: Vec<V> = Vec::new();
    for r in 0..st.rows {
        for c in 0..st.cols {
            if ok((r, c)) && (!attach || st.is_used((r, c))) {
                pool.push((r, c));
            }
        }
    }
    if pool.is_empty() {
        return None;
    }
    if !st.tree && rng.random() < cfg.ring_prob {
        if let Some(ring) = propose_ring(rng, st, &ok, cfg) {
            return Some(Rel::lp(t, ring, true));
        }
    }
    let want = rng.integers(2, cfg.max_chain + 1) as usize;
    let piece = if st.tree { st.pieces() } else { Vec::new() };
    let mut best: Vec<V> = Vec::new();
    for _ in 0..3 {
        let start = pool[rng.below(pool.len())];
        let path = walk(rng, st, start, want, &ok, cfg, &piece);
        if path.len() > best.len() {
            best = path;
        }
        if best.len() >= want {
            break;
        }
    }
    if best.len() < 2 {
        return None;
    }
    Some(Rel::lp(t, best, false))
}

/// A horn on a dot of the kolam (or on the bare dot `target`), pointing through a wall at an empty
/// spot or off the grid, so it is drawn as a crescent.
fn propose_horn(rng: &mut Rng, st: &St, target: Option<V>) -> Option<Rel> {
    let pool: Vec<V> = match target {
        Some(v) => vec![v],
        None => {
            let from_used = st.n_used > 0 && (st.tree || rng.random() < 0.85);
            let mut p = Vec::new();
            for r in 0..st.rows {
                for c in 0..st.cols {
                    if !from_used || st.is_used((r, c)) {
                        p.push((r, c));
                    }
                }
            }
            p
        }
    };
    if pool.is_empty() {
        return None;
    }
    for _ in 0..6 {
        let v = pool[rng.below(pool.len())];
        let dirs: Vec<u8> = (0..4u8)
            .filter(|&d| {
                let t = horn_target(v, d);
                !st.horns.contains(&(v, d)) && !st.is_used(t) && !st.gate(gate_between(v, t))
            })
            .collect();
        if !dirs.is_empty() {
            let var = rng.below(3) as u8;
            let d = dirs[rng.below(dirs.len())];
            return Some(Rel::horn(v, var, d));
        }
    }
    None
}

/// A straight chain that the symmetry maps onto itself, so a symmetric tree starts as one piece.
fn propose_seed(rng: &mut Rng, st: &St, t: Ty, ops: &[Op], cfg: &GenCfg) -> Option<Rel> {
    let mut cands = Vec::new();
    for l in 2..=cfg.max_chain {
        for r in 0..st.rows {
            for c in 0..=st.cols - l {
                cands.push(Rel::lp(t, (0..l).map(|k| (r, c + k)).collect(), false));
            }
        }
        for c in 0..st.cols {
            for r in 0..=st.rows - l {
                cands.push(Rel::lp(t, (0..l).map(|k| (r + k, c)).collect(), false));
            }
        }
    }
    cands.retain(|x| orbit(x, ops, st.rows, st.cols).len() == 1);
    if cands.is_empty() {
        None
    } else {
        let i = rng.below(cands.len());
        Some(cands.swap_remove(i))
    }
}

/// Serpentine through every dot: one brush line, one stroke, full coverage.
fn snake(rng: &mut Rng, rows: i32, cols: i32, t: Ty) -> Rel {
    let mut order = Vec::with_capacity((rows * cols) as usize);
    if rng.random() < 0.5 {
        for r in 0..rows {
            for k in 0..cols {
                order.push((r, if r % 2 == 0 { k } else { cols - 1 - k }));
            }
        }
    } else {
        for c in 0..cols {
            for k in 0..rows {
                order.push((if c % 2 == 0 { k } else { rows - 1 - k }, c));
            }
        }
    }
    Rel::lp(t, order, false)
}

/// Grow an existing open chain by one dot into the bare dot v.
fn propose_extension(rng: &mut Rng, st: &St, v: V) -> Option<(Rel, Rel)> {
    let nb = |a: V, b: V| adjacent(a, b);
    let mut opts: Vec<(Rel, Rel)> = Vec::new();
    for r in &st.rels {
        if r.ty == Ty::Horn || r.closed {
            continue;
        }
        let (first, last) = (r.vs[0], r.vs[r.vs.len() - 1]);
        if !nb(v, first) && !nb(v, last) {
            continue;
        }
        if st.members[r.ty as usize][st.vi(v)] {
            continue;
        }
        if nb(last, v) && !st.gate(gate_between(last, v)) {
            let mut vs = r.vs.clone();
            vs.push(v);
            opts.push((r.clone(), Rel::lp(r.ty, vs, false)));
        }
        if nb(first, v) && !st.gate(gate_between(first, v)) {
            let mut vs = vec![v];
            vs.extend_from_slice(&r.vs);
            opts.push((r.clone(), Rel::lp(r.ty, vs, false)));
        }
    }
    if opts.is_empty() {
        None
    } else {
        let i = rng.below(opts.len());
        Some(opts.swap_remove(i))
    }
}

/// One random kolam (uncropped) and its symmetry name.
pub fn sample(rng: &mut Rng, cfg: &GenCfg) -> (Kolam, &'static str) {
    let m = rng.integers(cfg.min_size, cfg.max_size + 1);
    let n = if rng.random() < cfg.square_prob { m } else { rng.integers(cfg.min_size, cfg.max_size + 1) };
    let mut sym = SYM_NAMES[rng.choice_p(&cfg.symmetry)];
    if m != n {
        sym = match sym {
            "c4" => "rot180",
            "d4" => "d2",
            s => s,
        };
    }
    let mut ops = group(sym);
    let bs: f64 = cfg.brush.iter().sum();
    let base: Vec<f64> = cfg.brush.iter().map(|x| x / bs).collect();
    let w: Vec<f64> = if rng.random() < cfg.pure_prob {
        let k = rng.choice_p(&base);
        (0..3).map(|i| if i == k { 1.0 } else { 0.0 }).collect()
    } else {
        let d = rng.dirichlet_ones(3);
        (0..3).map(|i| 0.5 * base[i] + 0.5 * d[i]).collect()
    };
    let mut tree = rng.random() < cfg.tree_prob;
    if tree && w[2] == 1.0 {
        tree = false; // horns alone cannot grow a tree
    }
    let target = rng.uniform(cfg.coverage.0, cfg.coverage.1);
    let mut st = St::new(m, n, tree);
    let pure_loop: Vec<Ty> = [Ty::Cross, Ty::Par].into_iter().filter(|&t| w[t as usize] == 1.0).collect();
    if tree && !pure_loop.is_empty() {
        let s = snake(rng, m, n, pure_loop[0]);
        if orbit(&s, ops, m, n).len() > 1 {
            sym = "none";
            ops = group("none");
        }
        try_set(&mut st, ops, &s, None);
    } else if tree && ops.len() > 1 {
        let loops: Vec<Ty> = [Ty::Cross, Ty::Par].into_iter().filter(|&t| w[t as usize] > 0.0).collect();
        let t = loops[rng.below(loops.len())];
        if let Some(seed) = propose_seed(rng, &st, t, ops, cfg) {
            try_set(&mut st, ops, &seed, None);
        }
    }
    let horn_cap = ((cfg.max_horn_frac * (m * n) as f64).round_ties_even() as usize).max(1);
    for _ in 0..cfg.max_proposals {
        if st.coverage() >= target {
            break;
        }
        let t = TYPES[rng.choice_p(&w)];
        if t == Ty::Horn && ((tree && st.n_used == 0) || st.horns.len() >= horn_cap) {
            continue;
        }
        let rel = if t == Ty::Horn { propose_horn(rng, &st, None) } else { propose_chain(rng, &st, t, cfg) };
        if let Some(r) = rel {
            try_set(&mut st, ops, &r, None);
        }
    }
    if rng.random() < cfg.fill_prob {
        // cover leftover bare dots: grow a neighbouring chain, put a horn on it, or a 2-dot loop
        let allowed: Vec<Ty> = TYPES.iter().copied().filter(|&t| w[t as usize] > 0.0).collect();
        let mut bare = Vec::new();
        for r in 0..m {
            for c in 0..n {
                if !st.is_used((r, c)) {
                    bare.push((r, c));
                }
            }
        }
        for i in rng.permutation(bare.len()) {
            let v = bare[i];
            if st.is_used(v) {
                continue;
            }
            if let Some((old, new)) = propose_extension(rng, &st, v) {
                if try_set(&mut st, ops, &new, Some(&old)) {
                    continue;
                }
            }
            for j in rng.permutation(allowed.len()) {
                if st.is_used(v) {
                    break;
                }
                let t = allowed[j];
                let rel = if t == Ty::Horn {
                    if tree || st.horns.len() >= horn_cap { None } else { propose_horn(rng, &st, Some(v)) }
                } else {
                    let nb: Vec<V> = STEPS
                        .iter()
                        .map(|s| (v.0 + s.0, v.1 + s.1))
                        .filter(|&u| st.loop_ok(t, true, false, u) && !st.gate(gate_between(u, v)) && (!tree || st.is_used(u)))
                        .collect();
                    if !nb.is_empty() && st.loop_ok(t, true, false, v) {
                        Some(Rel::lp(t, vec![v, nb[rng.below(nb.len())]], false))
                    } else {
                        None
                    }
                };
                if let Some(r) = rel {
                    try_set(&mut st, ops, &r, None);
                }
            }
        }
    }
    (Kolam::from_rels(m, n, &st.rels), sym)
}
