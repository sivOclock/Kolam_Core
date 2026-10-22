// The model, in a Web Worker so the page never freezes while it reads (reader.ts talks to it).
// Messages in:  {id, type: 'load', base}            -> {id, progress: 0..1}*, then {id, meta}
//               {id, type: 'read', base, x: Float32Array (3*S*S, CHW in [0,1])} -> {id, encoding, kolam, ms}
// Errors come back as {id, error}. Decoding is SeqModel.predict: greedy, one token at a time, constrained
// by allowedNext (encoding.allowed_next) so every answer is a well-formed encoding.

import * as ort from 'onnxruntime-web/wasm' // vite copies its .wasm into the build: no CDN, no server

const VOCAB = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '|', '<end>']
const BAR = 9
const END = 10
const FACING_OUT = { N: '0678', W: '0678', E: '0345' } // digits a border gap may hold

type Meta = { size: number; max_len: number; bos: number }
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
        ['encoder.onnx', 'decoder.onnx'].map((f, i) =>
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

async function read({ enc, dec, meta }: Model, x: Float32Array) {
  const t0 = performance.now()
  const S = meta.size
  const e = await enc.run({ image: new ort.Tensor('float32', x, [1, 3, S, S]) })
  const mem = e.mem
  const kolam = e.kolam ? (e.kolam.data as Float32Array)[0] : 1
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
    const out = await dec.run({
      mem,
      ids: new ort.Tensor('int64', ids, [1, L]),
      row: new ort.Tensor('int64', row, [1, L]),
      col: new ort.Tensor('int64', col, [1, L]),
      pos: new ort.Tensor('int64', BigInt64Array.of(BigInt(p)), [1]),
    })
    const lg = out.logits.data as Float32Array
    let ok = seq.length < L - 1 ? allowedNext(seq) : []
    if (!ok.length) ok = [END]
    const best = ok.reduce((a, b) => (lg[b] > lg[a] ? b : a))
    seq.push(best)
    if (best === END || p + 1 >= L) break
  }
  const encoding: string[] = []
  let r = ''
  for (const t of seq) {
    if (t === END) break
    if (t === BAR) (encoding.push(r), (r = ''))
    else r += VOCAB[t]
  }
  return { encoding, kolam, ms: Math.round(performance.now() - t0) }
}

self.onmessage = async (ev: MessageEvent) => {
  const { id, type, base, x } = ev.data
  try {
    const m = await load(base, (progress) => self.postMessage({ id, progress }))
    if (type === 'load') return self.postMessage({ id, meta: m.meta })
    self.postMessage({ id, ...(await read(m, x)) })
  } catch (e) {
    self.postMessage({ id, error: (e as Error).message })
  }
}
