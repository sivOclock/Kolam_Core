//! kolam_gen render: training pictures drawn ahead of time, every effect included, onto disk.
//!
//!   kolam_gen render data/enc_2m_20 -o data/img_12 -n 1440000 --max-dots 12 [--gpu]
//!
//! Kolams come from kolam_gen shards (load order = kolam_feed.data.shard_files); every 100th is left out,
//! like kolam_feed.train.split's held-out set, so training on these pictures never sees a val kolam.
//! Picture j shows kolam order[j % kolams] with look pass j / kolams: its own Rng (seed, kolam, pass) ->
//! Look (look.rs) -> turned kolam -> line mask (raster.rs) -> pixels (fx.rs, or fx.cu with --gpu) -> JPEG.
//! Out: img_00000.bin (the JPEGs one after another) + img_00000.jsonl (per picture: i, kolam, pass, turn,
//! encoding, M = lattice (lj, li, 1) -> pixel, off, len), written as .tmp then renamed, so a stopped run
//! resumes at the first missing shard; render.json describes the set.

use crate::look::{mul, Look, DEBRIS_W, MAX_DEBRIS, NP, OP_NAMES};
use crate::model::Kolam;
use crate::raster;
use crate::rng::Rng;
use rayon::prelude::*;
use std::io::{BufRead, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

pub struct Opts {
    pub src: PathBuf,
    pub out: PathBuf,
    pub n: usize,
    pub size: usize,
    pub max_dots: i32,
    pub seed: u64,
    pub shard: usize,
    pub batch: usize,
    pub gpu: bool,
}

struct Pic {
    params: [f32; NP],
    debris: [[f32; DEBRIS_W]; MAX_DEBRIS],
    mask: Vec<u8>,
    quality: u8,
    label: String, // the jsonl line, without off/len
}

/// Every kolam a picture may show: (index in load order, encoding rows joined by "|").
fn kolams(src: &Path, max_dots: i32) -> std::io::Result<Vec<(usize, String)>> {
    let mut files: Vec<PathBuf> = if src.is_dir() {
        std::fs::read_dir(src)?
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.file_name().and_then(|n| n.to_str()).is_some_and(|n| n.starts_with("shard_") && n.ends_with(".jsonl")))
            .collect()
    } else {
        vec![src.to_path_buf()]
    };
    files.sort();
    let mut out = Vec::new();
    let mut i = 0usize;
    for f in files {
        for l in std::io::BufReader::new(std::fs::File::open(f)?).lines() {
            let v: serde_json::Value = serde_json::from_str(&l?).expect("shard line");
            let enc: Vec<&str> = v["encoding"].as_array().unwrap().iter().map(|x| x.as_str().unwrap()).collect();
            let (rows, cols) = (((enc.len() - 1) / 2) as i32, enc[0].len() as i32);
            if i % 100 != 0 && rows.max(cols) <= max_dots {
                out.push((i, enc.join("|")));
            }
            i += 1;
        }
    }
    Ok(out)
}

/// Picture j, up to its pixels: look, turned kolam, line mask, label.
fn prepare(j: usize, kolam: usize, pass: usize, enc: &str, o: &Opts) -> Option<Pic> {
    let mut r = Rng::new(o.seed, kolam as u64, pass as u64);
    let mut look = Look::new(&mut r, o.size as f64);
    let rows: Vec<String> = enc.split('|').map(String::from).collect();
    let k = crate::look::turn(&Kolam::from_encoding(&rows).ok()?, look.op());
    let ras = raster::draw(&k, &look.style, o.size as u32, &mut r)?;
    look.finish(ras.coverage, o.size as f64);
    let m = mul(&look.h, &ras.m); // lattice point (lj, li, 1) -> picture pixel
    let rows: Vec<String> = k.encoding().iter().map(|r| format!("\"{r}\"")).collect();
    let mj: Vec<String> = m.iter().map(|r| format!("[{:.6},{:.6},{:.6}]", r[0], r[1], r[2])).collect();
    let label = format!(
        "{{\"i\":{j},\"kolam\":{kolam},\"pass\":{pass},\"turn\":\"{}\",\"rows\":{},\"cols\":{},\"encoding\":[{}],\"M\":[{}]",
        OP_NAMES[look.op], k.rows, k.cols, rows.join(","), mj.join(",")
    );
    Some(Pic { params: look.p, debris: look.debris, mask: ras.mask, quality: look.jpeg_quality, label })
}

fn jpeg(rgb: &[u8], s: usize, q: u8) -> Vec<u8> {
    let mut buf = Vec::with_capacity(s * s / 4);
    jpeg_encoder::Encoder::new(&mut buf, q).encode(rgb, s as u16, s as u16, jpeg_encoder::ColorType::Rgb).expect("jpeg");
    buf
}

#[cfg(feature = "gpu")]
type Gpu = crate::gpu::Gpu;
#[cfg(not(feature = "gpu"))]
struct Gpu;

/// The pictures' pixels: on the GPU (one batch) or, without one, fx.rs on every CPU thread.
fn pixels(pics: &[Pic], s: usize, gpu: Option<&Gpu>) -> Vec<Vec<u8>> {
    #[cfg(feature = "gpu")]
    if let Some(g) = gpu {
        let masks: Vec<u8> = pics.iter().flat_map(|p| p.mask.iter().copied()).collect();
        let params: Vec<f32> = pics.iter().flat_map(|p| p.params).collect();
        let debris: Vec<f32> = pics.iter().flat_map(|p| p.debris.iter().flatten().copied()).collect();
        let rgb = g.run(&masks, &params, &debris, s, pics.len()).unwrap_or_else(|e| panic!("gpu: {e}"));
        return rgb.chunks_exact(3 * s * s).map(|c| c.to_vec()).collect();
    }
    let _ = gpu;
    pics.par_iter().map(|p| crate::fx::run(&p.mask, &p.params, p.debris.as_flattened(), s)).collect()
}

pub fn render(o: &Opts) -> std::io::Result<()> {
    std::fs::create_dir_all(&o.out)?;
    let t0 = Instant::now();
    #[cfg(feature = "gpu")]
    let gpu = if o.gpu { Some(crate::gpu::Gpu::new().unwrap_or_else(|e| panic!("no GPU: {e}"))) } else { None };
    #[cfg(not(feature = "gpu"))]
    let gpu: Option<Gpu> = {
        if o.gpu {
            eprintln!("this kolam_gen was built without the GPU: cargo build --release --features gpu");
            std::process::exit(2);
        }
        None
    };
    #[cfg(feature = "gpu")]
    if let Some(g) = &gpu {
        eprintln!("pixels on the GPU: {}", g.name);
    }
    let ks = kolams(&o.src, o.max_dots)?;
    if ks.is_empty() {
        eprintln!("no kolams up to {0} x {0} in {1}", o.max_dots, o.src.display());
        std::process::exit(1);
    }
    let order = Rng::new(o.seed ^ 0x0DE5_0DE5, 0, 0).permutation(ks.len());
    eprintln!("{} kolams up to {} x {} ({:.1}s); {} pictures, {} looks each at most",
        ks.len(), o.max_dots, o.max_dots, t0.elapsed().as_secs_f64(), o.n, o.n.div_ceil(ks.len()));
    let n_shards = o.n.div_ceil(o.shard);
    let mut done = 0usize;
    for si in 0..n_shards {
        let (lo, hi) = (si * o.shard, ((si + 1) * o.shard).min(o.n));
        let (bin, labels) = (o.out.join(format!("img_{si:05}.bin")), o.out.join(format!("img_{si:05}.jsonl")));
        if bin.exists() && labels.exists() {
            eprintln!("shard {si}: kept");
            continue;
        }
        let (bin_tmp, labels_tmp) = (bin.with_extension("bin.tmp"), labels.with_extension("jsonl.tmp"));
        let (mut fb, mut fl) = (BufWriter::new(std::fs::File::create(&bin_tmp)?), BufWriter::new(std::fs::File::create(&labels_tmp)?));
        let mut off = 0usize;
        for b in (lo..hi).step_by(o.batch) {
            let js: Vec<usize> = (b..(b + o.batch).min(hi)).collect();
            let t = Instant::now();
            let pics: Vec<Pic> = js
                .par_iter()
                .filter_map(|&j| {
                    let (kolam, enc) = &ks[order[j % ks.len()]];
                    prepare(j, *kolam, j / ks.len(), enc, o)
                })
                .collect();
            let t_draw = t.elapsed().as_secs_f64();
            let rgb = pixels(&pics, o.size, gpu.as_ref());
            let t_px = t.elapsed().as_secs_f64() - t_draw;
            let jpegs: Vec<Vec<u8>> = pics.par_iter().zip(&rgb).map(|(p, c)| jpeg(c, o.size, p.quality)).collect();
            let t_jpeg = t.elapsed().as_secs_f64() - t_draw - t_px;
            for (p, jp) in pics.iter().zip(&jpegs) {
                fb.write_all(jp)?;
                writeln!(fl, "{},\"off\":{off},\"len\":{}}}", p.label, jp.len())?;
                off += jp.len();
            }
            done += pics.len();
            let secs = t0.elapsed().as_secs_f64();
            eprintln!("  {}/{}  {:.0} pictures/s  ({:.0}s, {:.1} GB; this batch: draw {:.2}s, pixels {:.2}s, jpeg {:.2}s)",
                b + js.len(), o.n, done as f64 / secs, secs, off as f64 / 1e9, t_draw, t_px, t_jpeg);
        }
        fb.flush()?;
        fl.flush()?;
        drop((fb, fl));
        std::fs::rename(&bin_tmp, &bin)?;
        std::fs::rename(&labels_tmp, &labels)?;
    }
    let secs = t0.elapsed().as_secs_f64();
    std::fs::write(
        o.out.join("render.json"),
        format!(
            "{{\n  \"src\": \"{}\",\n  \"n\": {},\n  \"size\": {},\n  \"max_dots\": {},\n  \"seed\": {},\n  \"shard\": {},\n  \"shards\": {},\n  \"kolams\": {},\n  \"look\": \"real\",\n  \"gpu\": {},\n  \"seconds\": {:.1},\n  \"held_out\": \"kolams whose index % 100 == 0 (kolam_feed.train.split's val)\",\n  \"generator\": \"kolam_gen {}\"\n}}\n",
            o.src.display(), o.n, o.size, o.max_dots, o.seed, o.shard, n_shards, ks.len(), o.gpu, secs, env!("CARGO_PKG_VERSION")
        ),
    )?;
    eprintln!("wrote {} pictures to {} in {:.0}s ({:.0}/s)", done, o.out.display(), secs, done as f64 / secs.max(1e-9));
    Ok(())
}
