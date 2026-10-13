//! kolam_gen: kolam encodings at scale (docs/hypergraph_kolams.md §11-12).
//!
//!   kolam_gen generate -n 1000000 -o data/enc_1m          # all four systems, shards of encodings
//!   kolam_gen check labels.jsonl                          # geometry stats per encoding (parity tests)
//!
//! Every sample: pick a system -> build gap states + horns -> crop -> compile -> geometry check ->
//! its kolam encoding. Reproducible from (seed, index, attempt); duplicates are drawn again.
//! Images are rendered later, while training (kolam_feed/).

mod geometry;
mod model;
mod random;
mod rng;
mod systems;

use model::Kolam;
use rayon::prelude::*;
use std::collections::HashSet;
use std::hash::{DefaultHasher, Hash, Hasher};
use std::io::{BufRead, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

const SYSTEMS: [&str; 4] = ["random", "substitution", "switching", "wfc"];

struct Opts {
    n: usize,
    out: PathBuf,
    seed: u64,
    shard: usize,
    threads: usize,
    mix: [f64; 4],
    cfg: random::GenCfg,
}

struct Sample {
    index: usize,
    attempt: u64,
    system: usize,
    symmetry: &'static str,
    k: Kolam,
    strokes: usize,
    crossings: usize,
}

/// The system of kolam `index`: drawn once, from its own stream, so redraws stay in that system
/// and the mix holds.
fn system_of(index: usize, o: &Opts) -> usize {
    rng::Rng::new(o.seed ^ 0x5157_E115, index as u64, 0).choice_p(&o.mix)
}

/// One sample: up to 200 tries until a kolam passes every check (label rules, then geometry).
fn make(index: usize, attempt: u64, o: &Opts, tiles: &systems::Tiles) -> Option<Sample> {
    let mut rng = rng::Rng::new(o.seed, index as u64, attempt);
    let system = system_of(index, o);
    for _ in 0..200 {
        let made = match system {
            0 => Some(random::sample(&mut rng, &o.cfg)),
            1 => systems::substitution(&mut rng, 12),
            2 => systems::switching(&mut rng, None),
            _ => systems::wfc(&mut rng, tiles, None),
        };
        let Some((raw, symmetry)) = made else { continue };
        let Some(k) = raw.crop() else { continue };
        let Some(c) = geometry::compile(&k) else { continue };
        let g = geometry::check(&c, false);
        if g.ok {
            let (strokes, crossings) = (g.n_strokes, g.n_crossings);
            return Some(Sample { index, attempt, system, symmetry, k, strokes, crossings });
        }
    }
    None
}

fn key(enc: &[String]) -> u128 {
    let (mut a, mut b) = (DefaultHasher::new(), DefaultHasher::new());
    enc.hash(&mut a);
    (0x5eedu32, enc).hash(&mut b);
    ((a.finish() as u128) << 64) | b.finish() as u128
}

fn line(s: &Sample, seed: u64, enc: &[String]) -> String {
    let rows: Vec<String> = enc.iter().map(|r| format!("\"{r}\"")).collect();
    format!(
        "{{\"id\":{},\"system\":\"{}\",\"symmetry\":\"{}\",\"seed\":[{},{},{}],\"rows\":{},\"cols\":{},\"encoding\":[{}],\"strokes\":{},\"crossings\":{},\"horns\":{}}}",
        s.index, SYSTEMS[s.system], s.symmetry, seed, s.index, s.attempt, s.k.rows, s.k.cols,
        rows.join(","), s.strokes, s.crossings, s.k.horns.len()
    )
}

fn shard_path(out: &Path, i: usize) -> PathBuf {
    out.join(format!("shard_{i:05}.jsonl"))
}

fn generate(o: &Opts) -> std::io::Result<()> {
    std::fs::create_dir_all(&o.out)?;
    let tiles = systems::photo_tiles();
    let t0 = Instant::now();
    let mut seen: HashSet<u128> = HashSet::with_capacity(o.n);
    let mut counts = [0usize; 4];
    let mut failed = 0usize;
    let (mut redrawn, mut dropped) = ([0usize; 4], [0usize; 4]); // per system
    let n_shards = o.n.div_ceil(o.shard);
    for si in 0..n_shards {
        let (lo, hi) = (si * o.shard, ((si + 1) * o.shard).min(o.n));
        let path = shard_path(&o.out, si);
        if path.exists() {
            // resume: a finished shard (written to .tmp, then renamed) is kept as it is
            for l in std::io::BufReader::new(std::fs::File::open(&path)?).lines() {
                let v: serde_json::Value = serde_json::from_str(&l?).expect("shard line");
                let enc: Vec<String> = v["encoding"].as_array().unwrap().iter().map(|x| x.as_str().unwrap().to_string()).collect();
                seen.insert(key(&enc));
                let s = SYSTEMS.iter().position(|x| Some(*x) == v["system"].as_str()).unwrap();
                counts[s] += 1;
            }
            eprintln!("shard {si}: kept ({} kolams)", hi - lo);
            continue;
        }
        let tmp = path.with_extension("jsonl.tmp");
        let mut f = BufWriter::new(std::fs::File::create(&tmp)?);
        for b in (lo..hi).step_by(8192) {
            let idx: Vec<usize> = (b..(b + 8192).min(hi)).collect();
            let mut batch: Vec<Option<Sample>> = idx.par_iter().map(|&i| make(i, 0, o, &tiles)).collect();
            // duplicates: draw those indices again (attempt 1, 2, ...), in parallel, until unique
            let mut pending: Vec<usize> = Vec::new();
            let mut encs: Vec<Option<Vec<String>>> = vec![None; batch.len()];
            for (j, s) in batch.iter().enumerate() {
                match s {
                    Some(s) => {
                        let e = s.k.encoding();
                        if seen.insert(key(&e)) {
                            encs[j] = Some(e);
                        } else {
                            pending.push(j);
                        }
                    }
                    None => failed += 1,
                }
            }
            let mut attempt = 0;
            while !pending.is_empty() && attempt < 50 {
                attempt += 1;
                for &j in &pending {
                    redrawn[system_of(idx[j], o)] += 1;
                }
                let again: Vec<Option<Sample>> = pending.par_iter().map(|&j| make(idx[j], attempt, o, &tiles)).collect();
                let mut still = Vec::new();
                for (&j, s) in pending.iter().zip(again) {
                    match s {
                        Some(s) => {
                            let e = s.k.encoding();
                            if seen.insert(key(&e)) {
                                encs[j] = Some(e);
                                batch[j] = Some(s);
                            } else {
                                still.push(j);
                            }
                        }
                        None => still.push(j),
                    }
                }
                pending = still;
            }
            for &j in &pending {
                dropped[system_of(idx[j], o)] += 1; // its system found nothing new in 50 redraws: no duplicate is kept
            }
            for (s, e) in batch.iter().zip(&encs) {
                if let (Some(s), Some(e)) = (s, e) {
                    counts[s.system] += 1;
                    writeln!(f, "{}", line(s, o.seed, e))?;
                }
            }
            let done = b + idx.len();
            let rate = done as f64 / t0.elapsed().as_secs_f64().max(1e-9);
            eprintln!("  {done}/{}  {:.0}/s  ({:.1}s)", o.n, rate, t0.elapsed().as_secs_f64());
        }
        f.flush()?;
        drop(f);
        std::fs::rename(&tmp, &path)?;
    }
    let secs = t0.elapsed().as_secs_f64();
    let mix: Vec<String> = SYSTEMS.iter().zip(o.mix).map(|(s, w)| format!("\"{s}\":{w}")).collect();
    let per = |v: [usize; 4]| SYSTEMS.iter().zip(v).map(|(s, c)| format!("\"{s}\":{c}")).collect::<Vec<_>>().join(", ");
    let manifest = format!(
        "{{\n  \"n\": {},\n  \"seed\": {},\n  \"shard\": {},\n  \"shards\": {},\n  \"mix\": {{{}}},\n  \"systems\": {{{}}},\n  \"random_grid\": [{}, {}],\n  \"redrawn_duplicates\": {{{}}},\n  \"dropped_duplicates\": {{{}}},\n  \"failed\": {},\n  \"seconds\": {:.1},\n  \"generator\": \"kolam_gen {}\",\n  \"brushes\": \"kolam_press/brushes.json\"\n}}\n",
        o.n, o.seed, o.shard, n_shards, mix.join(", "), per(counts), o.cfg.min_size, o.cfg.max_size,
        per(redrawn), per(dropped), failed, secs, env!("CARGO_PKG_VERSION")
    );
    std::fs::write(o.out.join("manifest.json"), manifest)?;
    eprintln!("wrote {} kolams to {} in {:.1}s ({:.0}/s)", counts.iter().sum::<usize>(), o.out.display(), secs, o.n as f64 / secs);
    Ok(())
}

/// Geometry stats for every encoding in a jsonl file (one object with "encoding" per line).
fn check_file(path: &Path) -> std::io::Result<()> {
    let out = std::io::stdout();
    let mut out = BufWriter::new(out.lock());
    for l in std::io::BufReader::new(std::fs::File::open(path)?).lines() {
        let v: serde_json::Value = serde_json::from_str(&l?).expect("json line");
        let enc: Vec<String> = v["encoding"].as_array().unwrap().iter().map(|x| x.as_str().unwrap().to_string()).collect();
        let k = Kolam::from_encoding(&enc).expect("encoding");
        let num = |x: f64| if x.is_finite() { format!("{x}") } else { "null".into() };
        match geometry::compile(&k) {
            Some(c) => {
                let g = geometry::check(&c, true);
                writeln!(out, "{{\"ok\":{},\"n_strokes\":{},\"n_crossings\":{},\"min_dot\":{},\"min_strand\":{},\"length\":{}}}",
                    g.ok, g.n_strokes, g.n_crossings, num(g.min_dot), num(g.min_strand), num(g.length))?;
            }
            None => writeln!(out, "{{\"ok\":false,\"error\":\"trace did not close\"}}")?,
        }
    }
    Ok(())
}

/// One big kolam (any size, beyond the dataset's 32 x 32 label limit), checked, as one jsonl line.
fn big(system: &str, size: i32, seed: u64, out: &Path) -> std::io::Result<()> {
    let tiles = systems::photo_tiles();
    let t0 = Instant::now();
    for attempt in 0..20 {
        let mut rng = rng::Rng::new(seed, 0, attempt);
        let made = match system {
            "substitution" => systems::substitution(&mut rng, size),
            "switching" => systems::switching(&mut rng, Some(size)),
            "wfc" => systems::wfc(&mut rng, &tiles, Some(size)),
            _ => usage(),
        };
        let Some(k) = made.and_then(|(k, _)| k.crop()) else { continue };
        let Some(c) = geometry::compile(&k) else { continue };
        let g = geometry::check(&c, true);
        eprintln!("attempt {attempt}: {}x{}, {} strokes, {} crossings, {} horns, geometry ok: {} ({:.1}s)",
            k.rows, k.cols, g.n_strokes, g.n_crossings, k.horns.len(), g.ok, t0.elapsed().as_secs_f64());
        if g.ok {
            let rows: Vec<String> = k.encoding().iter().map(|r| format!("\"{r}\"")).collect();
            std::fs::write(out, format!("{{\"system\":\"{system}\",\"rows\":{},\"cols\":{},\"encoding\":[{}]}}\n", k.rows, k.cols, rows.join(",")))?;
            eprintln!("wrote {}", out.display());
            return Ok(());
        }
    }
    eprintln!("no proper kolam in 20 attempts");
    std::process::exit(1)
}

fn usage() -> ! {
    eprintln!(
        "usage:\n  kolam_gen generate -n N -o DIR [--seed 2026] [--shard 50000] [--threads 0]\n                     [--mix random=0.55,substitution=0.15,switching=0.15,wfc=0.15] [--min-size 3] [--max-size 8]\n  kolam_gen check FILE.jsonl\n  kolam_gen big --system switching|wfc|substitution --size 100 [--seed 1] -o big.jsonl"
    );
    std::process::exit(2)
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("big") => {
            let (mut system, mut size, mut seed, mut out) = ("switching".to_string(), 100, 1u64, PathBuf::from("big.jsonl"));
            let mut it = args[1..].iter();
            while let Some(a) = it.next() {
                let v = it.next().unwrap_or_else(|| usage());
                match a.as_str() {
                    "--system" => system = v.clone(),
                    "--size" => size = v.parse().unwrap_or_else(|_| usage()),
                    "--seed" => seed = v.parse().unwrap_or_else(|_| usage()),
                    "-o" | "--out" => out = PathBuf::from(v),
                    _ => usage(),
                }
            }
            big(&system, size, seed, &out).unwrap();
        }
        Some("check") => check_file(Path::new(args.get(1).unwrap_or_else(|| usage()))).unwrap(),
        Some("generate") => {
            let mut o = Opts {
                n: 0,
                out: PathBuf::new(),
                seed: 2026,
                shard: 50_000,
                threads: 0,
                mix: [0.55, 0.15, 0.15, 0.15],
                cfg: random::GenCfg::default(),
            };
            let mut it = args[1..].iter();
            while let Some(a) = it.next() {
                let mut val = || it.next().unwrap_or_else(|| usage()).clone();
                match a.as_str() {
                    "-n" => o.n = val().parse().unwrap_or_else(|_| usage()),
                    "-o" | "--out" => o.out = PathBuf::from(val()),
                    "--seed" => o.seed = val().parse().unwrap_or_else(|_| usage()),
                    "--shard" => o.shard = val().parse().unwrap_or_else(|_| usage()),
                    "--threads" => o.threads = val().parse().unwrap_or_else(|_| usage()),
                    "--min-size" => o.cfg.min_size = val().parse().unwrap_or_else(|_| usage()),
                    "--max-size" => o.cfg.max_size = val().parse().unwrap_or_else(|_| usage()),
                    "--mix" => {
                        o.mix = [0.0; 4];
                        for kv in val().split(',') {
                            let (k, w) = kv.split_once('=').unwrap_or_else(|| usage());
                            let i = SYSTEMS.iter().position(|s| *s == k).unwrap_or_else(|| usage());
                            o.mix[i] = w.parse().unwrap_or_else(|_| usage());
                        }
                    }
                    _ => usage(),
                }
            }
            if o.n == 0 || o.out.as_os_str().is_empty() || o.shard == 0 || o.mix.iter().sum::<f64>() <= 0.0 {
                usage();
            }
            if o.cfg.min_size < 2 || o.cfg.max_size < o.cfg.min_size || o.cfg.max_size > 32 {
                usage();
            }
            if o.threads > 0 {
                rayon::ThreadPoolBuilder::new().num_threads(o.threads).build_global().unwrap();
            }
            generate(&o).unwrap();
        }
        _ => usage(),
    }
}
