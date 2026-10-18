// Photo reader: drop a kolam picture, the trained model reads its encoding (spec §12), and the board
// traces that kolam stroke by stroke. The model runs in `python -m kolam_feed serve <ckpt>...`;
// the dev server proxies /api to it. Drawing is the encoding viewer's path: fromEncoding -> compile -> drawn.

import './style.css'
import { Board, CHALK, bezierSegments, type XY } from './board'
import { compile } from './kolam/compiler'
import { drawn } from './kolam/drawn'
import { fromEncoding } from './kolam/encoding'
import { validate } from './kolam/model'

const NAMES: Record<string, string> = { seq: 'A · writer (seq)', grid: 'B · reader (grid)' }

document.querySelector<HTMLDivElement>('#app')!.innerHTML = `
  <div class="enc-page">
    <div class="enc-input">
      <div class="panel-top">
        <span class="title">read a kolam photo</span>
        <button id="pick">choose image</button>
        <button id="again" disabled>trace again</button>
      </div>
      <input id="file" type="file" accept="image/*" hidden />
      <div id="drop" class="drop">drop an image here, paste one, or choose a file</div>
      <div id="status" class="status" role="status"></div>
    </div>
    <div id="grid" class="enc-grid"></div>
  </div>`

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T
const drop = $<HTMLDivElement>('drop')
const grid = $<HTMLDivElement>('grid')
const status = $<HTMLDivElement>('status')
const file = $<HTMLInputElement>('file')
const again = $<HTMLButtonElement>('again')

type Card = { board: Board; draw: (animate: boolean) => void }
let models: string[] = []
let cards: Card[] = []
let run = 0 // bumps on every new image so old animations stop

function say(text: string, error = false): void {
  status.textContent = text
  status.classList.toggle('error', error)
}

async function loadModels(): Promise<void> {
  try {
    const r = await fetch('/api/models')
    models = (await r.json()).map((m: { model: string }) => m.model)
    say(`models loaded: ${models.map((m) => NAMES[m] ?? m).join(', ')}`)
  } catch {
    say('model server not running: python -m kolam_feed serve runs/pilot_seq/last.pt runs/pilot_grid/last.pt', true)
  }
}

/** One board for one encoding; draw(true) traces it stroke by stroke in about 3 seconds. */
function kolamCard(canvas: HTMLCanvasElement, label: string, enc: string[], note: HTMLDivElement): Card | null {
  try {
    const h = fromEncoding(enc)
    const k = compile(h)
    const problems = validate(h)
    note.textContent += ` · ${h.rows}×${h.cols} · ${k.strokes.length} stroke${k.strokes.length === 1 ? '' : 's'}`
    if (problems.length) {
      note.textContent += ` · ${problems[0].message}`
      note.classList.add('error')
    }
    const board = new Board(canvas, label)
    const draw = (animate: boolean) => {
      const mine = run
      board.layout(h.rows, h.cols, 1) // room for crescents pointing off the grid
      board.clear()
      board.dots(CHALK.main, Math.max(2, board.s * 0.14), k.used) // unused dots are not drawn
      const w = Math.max(1.5, board.s * 0.14)
      const segs: [XY[], number][] = []
      drawn(k).strokes.forEach((P, si) =>
        bezierSegments(board, P).forEach((seg, i) => segs.push([seg, si * 1009 + i * 7 + 1])),
      )
      if (!animate) return segs.forEach(([seg, seed]) => board.chalk(seg, CHALK.main, w, seed))
      const per = Math.max(1, Math.ceil(segs.length / 180)) // ~180 frames whatever the size
      let n = 0
      const step = () => {
        if (mine !== run) return
        for (const [seg, seed] of segs.slice(n, n + per)) board.chalk(seg, CHALK.main, w, seed)
        n += per
        if (n < segs.length) requestAnimationFrame(step)
      }
      requestAnimationFrame(step)
    }
    return { board, draw }
  } catch (e) {
    note.textContent += ` · not a drawable kolam: ${(e as Error).message}`
    note.classList.add('error')
    return null
  }
}

function card(): { box: HTMLDivElement; canvas: HTMLCanvasElement; note: HTMLDivElement; pre: HTMLPreElement } {
  const box = document.createElement('div')
  box.className = 'enc-card'
  const canvas = document.createElement('canvas')
  canvas.className = 'board'
  const note = document.createElement('div')
  note.className = 'status'
  const pre = document.createElement('pre')
  box.append(canvas, note, pre)
  note.onclick = () => box.classList.toggle('open')
  return { box, canvas, note, pre }
}

async function read(img: Blob): Promise<void> {
  if (!models.length) await loadModels()
  if (!models.length) return
  run++
  cards = []
  grid.innerHTML = ''
  again.disabled = true
  const photo = document.createElement('img')
  photo.src = URL.createObjectURL(img)
  drop.replaceChildren(photo)
  say(`reading with ${models.length} model${models.length === 1 ? '' : 's'}…`)
  const seen = document.createElement('div')
  seen.className = 'enc-card seen'
  grid.append(seen)
  const results = await Promise.all(
    models.map(async (m) => {
      const r = await fetch(`/api/predict?model=${m}`, { method: 'POST', body: img })
      return { m, r: await r.json() }
    }),
  ).catch((e) => (say(`model server error: ${e}`, true), null))
  if (!results) return
  for (const { m, r } of results) {
    if (r.error) {
      say(r.error, true)
      continue
    }
    if (!seen.childElementCount) {
      const im = document.createElement('img')
      im.src = r.seen
      const cap = document.createElement('div')
      cap.className = 'status'
      im.onload = () => (cap.textContent = `what the model saw: ${im.naturalWidth}px square, padded + resized`)
      seen.append(im, cap)
    }
    const c = card()
    grid.append(c.box)
    c.note.textContent = `${NAMES[m] ?? m} · ${r.ms} ms`
    c.pre.textContent = r.encoding.length ? r.encoding.join('\n') : '(empty)'
    const k = kolamCard(c.canvas, m === 'seq' ? 'A' : 'B', r.encoding, c.note)
    if (k) cards.push(k)
    else c.box.classList.add('open')
  }
  if (results.some(({ r }) => !r.error)) say('click a caption for the digits it read')
  cards.forEach((c) => c.draw(true))
  again.disabled = !cards.length
}

function firstImage(items: DataTransferItemList | FileList | undefined): Blob | null {
  for (const it of Array.from<DataTransferItem | File>(items ?? [])) {
    const f = it instanceof File ? it : it.kind === 'file' ? it.getAsFile() : null
    if (f && f.type.startsWith('image/')) return f
  }
  return null
}

drop.addEventListener('dragover', (e) => (e.preventDefault(), drop.classList.add('over')))
drop.addEventListener('dragleave', () => drop.classList.remove('over'))
drop.addEventListener('drop', (e) => {
  e.preventDefault()
  drop.classList.remove('over')
  const f = firstImage(e.dataTransfer?.files)
  if (f) read(f)
})
document.addEventListener('dragover', (e) => e.preventDefault())
document.addEventListener('drop', (e) => e.preventDefault()) // a missed drop shouldn't open the image
document.addEventListener('paste', (e) => {
  const f = firstImage(e.clipboardData?.items)
  if (f) read(f)
})
drop.onclick = () => file.click()
$<HTMLButtonElement>('pick').onclick = () => file.click()
file.onchange = () => file.files?.[0] && read(file.files[0])
again.onclick = () => (run++, cards.forEach((c) => c.draw(true)))

let resize = 0
window.addEventListener('resize', () => {
  clearTimeout(resize)
  resize = window.setTimeout(() => cards.forEach(({ board, draw }) => (board.invalidateBase(), draw(false))), 150)
})
document.fonts.ready.then(() => cards.forEach(({ board }) => board.invalidateBase()))
loadModels()
