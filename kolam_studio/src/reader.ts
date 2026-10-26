// The trained model (seq or grid), run in the browser (no server): `python -m kolam_feed export <ckpt>` writes
// public/model/<id>/{encoder,decoder}.onnx + model.json. The model runs in a Web Worker
// (reader.worker.ts) so the page stays smooth while it reads; here the photo is prepared like
// kolam_feed.train.prepare (upright, padded to a square with its edge colour, resized) and sent over.

type Meta = { kind?: string; size: number; max_len: number; bos: number }
type Reply = { id: number; progress?: number; rows?: string[]; meta?: Meta; error?: string; encoding?: string[]; kolam?: number; ms?: number }

const worker = new Worker(new URL('./reader.worker.ts', import.meta.url), { type: 'module' })
type Waiting = { done: (r: Reply) => void; fail: (e: Error) => void; progress?: (f: number) => void; rows?: (rows: string[]) => void }
const waiting = new Map<number, Waiting>()
let next = 0
worker.onmessage = (ev: MessageEvent<Reply>) => {
  const w = waiting.get(ev.data.id)
  if (!w) return
  if (ev.data.progress !== undefined) return w.progress?.(ev.data.progress)
  if (ev.data.rows) return w.rows?.(ev.data.rows)
  waiting.delete(ev.data.id)
  if (ev.data.error) w.fail(new Error(ev.data.error))
  else w.done(ev.data)
}

function ask(msg: object, transfer: Transferable[] = [], progress?: (f: number) => void, rows?: (rows: string[]) => void): Promise<Reply> {
  const id = next++
  return new Promise((done, fail) => {
    waiting.set(id, { done, fail, progress, rows })
    worker.postMessage({ id, ...msg }, transfer)
  })
}

const metas = new Map<string, Promise<Meta>>()
// the worker resolves URLs against its own script (assets/), so it gets them absolute
const abs = (base: string) => new URL(base, location.href).href

/** Load the model in folder `base` (once); onProgress gets 0..1 while it downloads. */
export function loadReader(base = './model/', onProgress?: (f: number) => void): Promise<Meta> {
  let m = metas.get(base)
  if (!m) {
    m = ask({ type: 'load', base: abs(base) }, [], onProgress).then((r) => r.meta!)
    m.catch(() => metas.delete(base))
    metas.set(base, m)
  }
  return m
}

/** kolam_feed.train.prepare: upright, padded to a square with its edge colour, resized to size. The
 *  browser shrinks the photo first (createImageBitmap's resize), so a 12-megapixel phone photo
 *  costs the page almost nothing. */
export async function prepare(img: Blob, size: number): Promise<HTMLCanvasElement> {
  const full = await createImageBitmap(img, { imageOrientation: 'from-image' })
  const { width: w, height: h } = full
  const f = size / Math.max(w, h)
  const [sw, sh] = [Math.max(1, Math.round(w * f)), Math.max(1, Math.round(h * f))]
  const bm = await createImageBitmap(full, { resizeWidth: sw, resizeHeight: sh, resizeQuality: 'high' })
  full.close()
  const src = document.createElement('canvas')
  ;[src.width, src.height] = [sw, sh]
  const sc = src.getContext('2d', { willReadFrequently: true })!
  sc.drawImage(bm, 0, 0)
  const px = sc.getImageData(0, 0, sw, sh).data
  const edge = [0, 0, 0]
  let n = 0
  const add = (x: number, y: number) => {
    const k = (y * sw + x) * 4
    for (let c = 0; c < 3; c++) edge[c] += px[k + c]
    n++
  }
  for (let x = 0; x < sw; x++) (add(x, 0), add(x, sh - 1))
  for (let y = 0; y < sh; y++) (add(0, y), add(sw - 1, y))
  const out = document.createElement('canvas')
  out.width = out.height = size
  const oc = out.getContext('2d', { willReadFrequently: true })!
  oc.fillStyle = `rgb(${edge.map((v) => Math.floor(v / n)).join(',')})`
  oc.fillRect(0, 0, size, size)
  oc.drawImage(bm, Math.round((size - sw) / 2), Math.round((size - sh) / 2))
  bm.close()
  return out
}

/** Photo -> kolam encoding (rows of digits), the chance the photo is a kolam at all (the model's kolam
 *  head; 1 for a model exported before it had one), the square the model saw and the time taken.
 *  onRows (seq models): the rows read so far, each time a dot row is done, while it reads. */
export async function readKolam(
  img: Blob,
  base = './model/',
  check = false, // the model's kolam head is trained: on "not a kolam" skip the decoder (encoding [])
  onRows?: (rows: string[]) => void,
): Promise<{ encoding: string[]; kolam: number; seen: HTMLCanvasElement; ms: number }> {
  const meta = await loadReader(base)
  const seen = await prepare(img, meta.size)
  const S = meta.size
  const px = seen.getContext('2d')!.getImageData(0, 0, S, S).data
  const x = new Float32Array(3 * S * S)
  for (let i = 0; i < S * S; i++) for (let c = 0; c < 3; c++) x[c * S * S + i] = px[i * 4 + c] / 255
  const r = await ask({ type: 'read', base: abs(base), x, check }, [x.buffer], undefined, onRows)
  return { encoding: r.encoding!, kolam: r.kolam!, seen, ms: r.ms! }
}
