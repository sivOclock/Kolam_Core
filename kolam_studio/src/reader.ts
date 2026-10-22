// The trained seq model, run in the browser (no server): `python -m kolam_feed export <ckpt>` writes
// public/model/{encoder,decoder}.onnx + model.json. The image goes in like kolam_feed.train.prepare
// (padded to a square with its edge colour, resized); decoding is SeqModel.predict: greedy, one token
// at a time, constrained by allowedNext (encoding.allowed_next) so every answer is a well-formed encoding.

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

/** The model in folder `base` (encoder.onnx, decoder.onnx, model.json); onProgress gets 0..1. */
export function loadReader(base = './model/', onProgress?: (f: number) => void): Promise<Model> {
  let m = models.get(base)
  if (!m) {
    m = (async () => {
      const meta: Meta = await (await fetch(base + 'model.json')).json()
      const got = [0, 0], size = [0, 0]
      const files = await Promise.all(
        ['encoder.onnx', 'decoder.onnx'].map((f, i) =>
          download(base + f, (n, t) => {
            ;[got[i], size[i]] = [n, t]
            if (size[0] && size[1]) onProgress?.((got[0] + got[1]) / (size[0] + size[1]))
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
export function allowedNext(prefix: number[]): number[] {
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

/** kolam_feed.train.prepare: upright, padded to a square with its edge colour, resized to size. */
export async function prepare(img: Blob, size: number): Promise<HTMLCanvasElement> {
  const bm = await createImageBitmap(img, { imageOrientation: 'from-image' })
  const { width: w, height: h } = bm
  const src = document.createElement('canvas')
  ;[src.width, src.height] = [w, h]
  const sc = src.getContext('2d', { willReadFrequently: true })!
  sc.drawImage(bm, 0, 0)
  const px = sc.getImageData(0, 0, w, h).data
  const edge = [0, 0, 0]
  let n = 0
  const add = (x: number, y: number) => {
    const k = (y * w + x) * 4
    for (let c = 0; c < 3; c++) edge[c] += px[k + c]
    n++
  }
  for (let x = 0; x < w; x++) (add(x, 0), add(x, h - 1))
  for (let y = 0; y < h; y++) (add(0, y), add(w - 1, y))
  const S = Math.max(w, h)
  const out = document.createElement('canvas')
  out.width = out.height = size
  const oc = out.getContext('2d', { willReadFrequently: true })!
  oc.fillStyle = `rgb(${edge.map((v) => Math.floor(v / n)).join(',')})`
  oc.fillRect(0, 0, size, size)
  oc.imageSmoothingQuality = 'high'
  const f = size / S
  // halve step by step first: one big downscale on a canvas aliases badly
  let cur: CanvasImageSource = bm
  let [cw, ch] = [w, h]
  while (cw * f < cw / 2 && cw > 2 * size) {
    const t = document.createElement('canvas')
    ;[t.width, t.height] = [Math.ceil(cw / 2), Math.ceil(ch / 2)]
    const tc = t.getContext('2d')!
    tc.imageSmoothingQuality = 'high'
    tc.drawImage(cur, 0, 0, t.width, t.height)
    ;[cur, cw, ch] = [t, t.width, t.height]
  }
  oc.drawImage(cur, ((S - w) / 2) * f, ((S - h) / 2) * f, w * f, h * f)
  return out
}

/** Photo -> kolam encoding (rows of digits), the chance the photo is a kolam at all (the model's kolam
 *  head; 1 for a model exported before it had one), the square the model saw and the time taken. */
export async function readKolam(
  img: Blob,
  base = './model/',
): Promise<{ encoding: string[]; kolam: number; seen: HTMLCanvasElement; ms: number }> {
  const { enc, dec, meta } = await loadReader(base)
  const seen = await prepare(img, meta.size)
  const t0 = performance.now()
  const S = meta.size
  const px = seen.getContext('2d')!.getImageData(0, 0, S, S).data
  const x = new Float32Array(3 * S * S)
  for (let i = 0; i < S * S; i++) for (let c = 0; c < 3; c++) x[c * S * S + i] = px[i * 4 + c] / 255
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
  return { encoding, kolam, seen, ms: Math.round(performance.now() - t0) }
}
