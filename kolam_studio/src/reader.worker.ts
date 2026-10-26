// The model, in a Web Worker so the page never freezes while it reads (reader.ts talks to it).
// Messages in:  {id, type: 'load', base}            -> {id, progress: 0..1}*, then {id, meta}
//               {id, type: 'read', base, x: Float32Array (3*S*S, CHW in [0,1]), check} -> {id, encoding, kolam, ms}
//               (check: stop after the encoder when it says "not a kolam"; encoding is then [])
// Errors come back as {id, error}. Decoding is SeqModel.predict: greedy, one token at a time, constrained
// by allowedNext (encoding.allowed_next) so every answer is a well-formed encoding.

import * as ort from 'onnxruntime-web/wasm' // vite copies its .wasm into the build: no CDN, no server

const VOCAB = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '|', '<end>']
const BAR = 9
const END = 10
const FACING_OUT: Record<string, string> = { N: '0678', W: '0678', S: '0345', E: '0345' } // digits a border gap may hold

// kind "grid" (model B): encoder.onnx + gaps.onnx, read by readGrid; otherwise the seq model (A)
type Meta = { kind?: 'seq' | 'grid'; size: number; max_len: number; bos: number; dynamic?: boolean }
type Model = { enc: ort.InferenceSession; dec: ort.InferenceSession; meta: Meta }
const models = new Map<string, Promise<Model>>() // by folder, each loaded once

/** Fetch a file, reporting bytes as they arrive (a phone on mobile data wants to see progress). */
async function download(url: string, onBytes: (n: number, total: number) => void): Promise<Uint8Array> {
  const r = await fetch(url)
  if (!r.ok || !r.body) throw new Error(`${url}: ${r.status}`)
  const total = +(r.headers.get('content-length') ?? 0)
  const parts: Uint8Array[] = []
  let n = 0
  for (const rd = r.body.getReader(); ; ) {
    const { done, value } = await rd.read()
    if (done) break
    parts.push(value)
    onBytes((n += value.length), total)
  }
  const out = new Uint8Array(n)
  parts.reduce((o, p) => (out.set(p, o), o + p.length), 0)
  return out
}

function load(base: string, onProgress: (f: number) => void): Promise<Model> {
  let m = models.get(base)
  if (!m) {
    m = (async () => {
      const meta: Meta = await (await fetch(base + 'model.json')).json()
      const got = [0, 0], size = [0, 0]
      const files = await Promise.all(
        ['encoder.onnx', meta.kind === 'grid' ? 'gaps.onnx' : 'decoder.onnx'].map((f, i) =>
          download(base + f, (n, t) => {
            ;[got[i], size[i]] = [n, t]
            if (size[0] && size[1]) onProgress((got[0] + got[1]) / (size[0] + size[1]))
          }),
        ),
      )
      const [enc, dec] = await Promise.all(files.map((b) => ort.InferenceSession.create(b)))
      return { enc, dec, meta }
    })()
    m.catch(() => models.delete(base))
    models.set(base, m)
  }
  return m
}

/** encoding.allowed_next: the tokens that keep a partial encoding well formed. */
function allowedNext(prefix: number[]): number[] {
  const rows: number[] = []
  let row = 0
  for (const t of prefix) {
    if (t === END) return []
    if (t === BAR) (rows.push(row), (row = 0))
    else row++
  }
  const i = rows.length
  const digits = (ok: string) => [...ok].map(Number)
  if (i === 0) return [...digits(FACING_OUT.N), ...(row ? [BAR] : [])]
  const want = rows[0] + (i % 2)
  if (row === want) return [BAR]
  const end = row === 0 && i >= 3 && i % 2 === 1 ? [END] : []
  let ok = '012345678'
  if (i % 2 && row === 0) ok = FACING_OUT.W
  else if (i % 2 && row === want - 1) ok = FACING_OUT.E
  return [...digits(ok), ...end]
}

/** model.homography: the 3x3 H (row-major) with dst ~ H src, from 4 point pairs (an 8x8 solve). */
function homography(src: number[][], dst: number[][]): number[] {
  const A: number[][] = []
  src.forEach(([x, y], i) => A.push([x, y, 1, 0, 0, 0, -dst[i][0] * x, -dst[i][0] * y, dst[i][0]]))
  src.forEach(([x, y], i) => A.push([0, 0, 0, x, y, 1, -dst[i][1] * x, -dst[i][1] * y, dst[i][1]]))
  for (let c = 0; c < 8; c++) {
    const p = A.slice(c).reduce((b, r, k) => (Math.abs(r[c]) > Math.abs(A[b][c]) ? c + k : b), c)
    ;[A[c], A[p]] = [A[p], A[c]]
    for (let r = 0; r < 8; r++) {
      if (r === c) continue
      const f = A[r][c] / A[c][c]
      for (let k = c; k < 9; k++) A[r][k] -= f * A[c][k]
    }
  }
  return [...A.map((r, i) => r[8] / r[i]), 1]
}

const argmax = (a: ArrayLike<number>, from = 0, to = a.length) => {
  let b = from
  for (let i = from; i < to; i++) if (a[i] > a[b]) b = i
  return b - from
}

/** GridModel.predict: size and lattice corners from the picture, then every gap read at its spot. */
async function readGrid({ enc, dec, meta }: Model, x: Float32Array, check: boolean) {
  const t0 = performance.now()
  const S = meta.size
  const e = await enc.run({ image: new ort.Tensor('float32', x, [1, 3, S, S]) })
  const kolam = (e.kolam.data as Float32Array)[0]
  if (check && kolam < 0.5) return { encoding: [], kolam, ms: Math.round(performance.now() - t0) } // no gaps read
  const R = argmax(e.rows.data as Float32Array) + 1
  const C = argmax(e.cols.data as Float32Array) + 1
  const c = e.corners.data as Float32Array
  const H = homography([[0, 0], [2 * C, 0], [2 * C, 2 * R], [0, 2 * R]], [0, 1, 2, 3].map((k) => [c[2 * k], c[2 * k + 1]]))
  const xy: number[] = [], orient: bigint[] = []
  for (let li = 0; li <= 2 * R; li++)            // model.gap_points, in encoding order
    for (let lj = 1 - (li % 2); lj <= 2 * C; lj += 2) {
      const w = H[6] * lj + H[7] * li + H[8]
      xy.push((H[0] * lj + H[1] * li + H[2]) / w, (H[3] * lj + H[4] * li + H[5]) / w)
      orient.push(BigInt(li % 2))
    }
  const n = orient.length
  const out = await dec.run({
    f8: e.f8, f16: e.f16,
    xy: new ort.Tensor('float32', Float32Array.from(xy), [n, 2]),
    orient: new ort.Tensor('int64', BigInt64Array.from(orient), [n]),
  })
  const lg = out.logits.data as Float32Array
  const D = lg.length / n
  const encoding: string[] = []
  let k = 0
  for (let li = 0; li <= 2 * R; li++) {
    let row = ''
    const n = li % 2 ? C + 1 : C
    for (let j = 0; j < n; j++, k++) {
      // a border gap only holds a wall or a horn facing out (encoding.FACING_OUT), else it can't be drawn
      const side = li === 0 ? 'N' : li === 2 * R ? 'S' : li % 2 && j === 0 ? 'W' : li % 2 && j === n - 1 ? 'E' : ''
      const ok = side ? [...FACING_OUT[side]].map(Number) : [...Array(D).keys()]
      row += ok.reduce((a, b) => (lg[k * D + b] > lg[k * D + a] ? b : a))
    }
    encoding.push(row)
  }
  return { encoding, kolam, ms: Math.round(performance.now() - t0) }
}

/** check: the model's kolam head is trained, so a picture it calls "not a kolam" (< 0.5) stops after
 *  the encoder: no digits, no decoder time. */
async function read(m: Model, x: Float32Array, check: boolean, onRows?: (rows: string[]) => void) {
  if (m.meta.kind === 'grid') return readGrid(m, x, check)
  const { enc, dec, meta } = m
  const t0 = performance.now()
  const S = meta.size
  const e = await enc.run({ image: new ort.Tensor('float32', x, [1, 3, S, S]) })
  const mem = e.mem
  const kolam = e.kolam ? (e.kolam.data as Float32Array)[0] : 1
  if (check && kolam < 0.5) return { encoding: [], kolam, ms: Math.round(performance.now() - t0) } // decoder never runs
  const L = meta.max_len
  const ids = new BigInt64Array(L).fill(BigInt(meta.bos))
  const row = new BigInt64Array(L)
  const col = new BigInt64Array(L)
  const seq: number[] = []
  let bars = 0
  let last = 0
  for (let p = 0; ; p++) {
    // model.token_slots for position p: rows end at "|", BOS starts row 0
    const t = p === 0 ? meta.bos : seq[p - 1]
    ids[p] = BigInt(t)
    if (t === BAR) bars++
    if (t === BAR || p === 0) last = p
    row[p] = BigInt(bars)
    col[p] = BigInt(p - last)
    const n = meta.dynamic ? p + 1 : L // a dynamic decoder takes only the tokens so far, not the padding
    const out = await dec.run({
      mem,
      ids: new ort.Tensor('int64', ids.subarray(0, n), [1, n]),
      row: new ort.Tensor('int64', row.subarray(0, n), [1, n]),
      col: new ort.Tensor('int64', col.subarray(0, n), [1, n]),
      pos: new ort.Tensor('int64', BigInt64Array.of(BigInt(p)), [1]),
    })
    const lg = out.logits.data as Float32Array
    let ok = seq.length < L - 1 ? allowedNext(seq) : []
    if (!ok.length) ok = [END]
    const best = ok.reduce((a, b) => (lg[b] > lg[a] ? b : a))
    seq.push(best)
    if (best === BAR && bars % 2 === 0) onRows?.(rowsOf(seq)) // a dot row is done: its rows so far draw a kolam
    if (best === END || p + 1 >= L) break
  }
  return { encoding: rowsOf(seq), kolam, ms: Math.round(performance.now() - t0) }
}

/** Tokens -> rows of digits (the rows a "|" has closed, then the rest up to <end>). */
function rowsOf(seq: number[]): string[] {
  const rows: string[] = []
  let r = ''
  for (const t of seq) {
    if (t === END) break
    if (t === BAR) (rows.push(r), (r = ''))
    else r += VOCAB[t]
  }
  return rows
}

self.onmessage = async (ev: MessageEvent) => {
  const { id, type, base, x, check } = ev.data
  try {
    const m = await load(base, (progress) => self.postMessage({ id, progress }))
    if (type === 'load') return self.postMessage({ id, meta: m.meta })
    self.postMessage({ id, ...(await read(m, x, !!check, (rows) => self.postMessage({ id, rows }))) })
  } catch (e) {
    self.postMessage({ id, error: (e as Error).message })
  }
}
