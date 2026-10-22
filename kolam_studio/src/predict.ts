// Kolam Reader: take or pick a photo of a kolam, a trained model reads its encoding (spec §12) and the
// board traces that kolam stroke by stroke. Everything runs on the device (reader.ts: ONNX files from
// `python -m kolam_feed export`, listed in public/model/models.json), so the built site is static files.
// One column on a phone, photo and drawing side by side on a wider screen.

import './style.css'
import { Board, CHALK, bezierSegments, type XY } from './board'
import { compile } from './kolam/compiler'
import { drawn } from './kolam/drawn'
import { fromEncoding } from './kolam/encoding'
import { validate } from './kolam/model'
import { loadReader, readKolam } from './reader'

type ModelInfo = { id: string; name: string; params: string; note: string; kolam_check: boolean }

document.querySelector<HTMLDivElement>('#app')!.innerHTML = `
  <main class="rd">
    <header class="rd-head">
      <h1>Kolam Reader</h1>
      <p>Show it a photo of a kolam: a small model reads its dots and lines and draws it back.
         It runs on this device, the photo never leaves it.</p>
    </header>

    <section class="rd-pick">
      <div class="rd-models" id="models" role="radiogroup" aria-label="model"></div>
      <div class="rd-actions">
        <label class="rd-btn primary"><input id="camera" type="file" accept="image/*" capture="environment" hidden />
          <span aria-hidden="true">◉</span> Take photo</label>
        <label class="rd-btn"><input id="file" type="file" accept="image/*" hidden />Choose image</label>
      </div>
      <div class="rd-drop" id="drop">or drop / paste an image here</div>
      <div class="rd-bar" id="bar" hidden><i></i></div>
      <div class="rd-status" id="status" role="status"></div>
    </section>

    <section class="rd-result" id="result" hidden>
      <figure class="rd-card">
        <div class="rd-frame"><canvas id="seen"></canvas></div>
        <figcaption>what the model saw</figcaption>
      </figure>
      <figure class="rd-card">
        <div class="rd-frame"><canvas id="board" class="board"></canvas></div>
        <figcaption id="caption">what it read</figcaption>
      </figure>
      <details class="rd-digits"><summary>the digits it read</summary><pre id="digits"></pre></details>
      <button class="rd-btn" id="again">trace again</button>
    </section>

    <footer class="rd-foot">
      Works best on dot kolams up to 5 × 5 dots, photographed from above. Early models, trained on drawn
      kolams; bigger kolams are next. <a href="./index.html">Draw your own kolam</a>
    </footer>
  </main>`

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T
const status = $<HTMLDivElement>('status')
const bar = $<HTMLDivElement>('bar')
let models: ModelInfo[] = []
let current = ''
let run = 0 // bumps on every new reading so an old trace stops
let redraw: ((animate: boolean) => void) | null = null

function say(text: string, error = false): void {
  status.textContent = text
  status.classList.toggle('error', error)
}

const base = (id: string) => `./model/${id}/`

async function useModel(id: string): Promise<boolean> {
  current = id
  document.querySelectorAll<HTMLButtonElement>('#models button').forEach((b) => b.setAttribute('aria-checked', String(b.dataset.id === id)))
  const m = models.find((x) => x.id === id)!
  say(`loading the ${m.name.toLowerCase()} model…`)
  bar.hidden = false
  const fill = bar.querySelector('i')!
  fill.style.width = '0%'
  try {
    await loadReader(base(id), (f) => (fill.style.width = `${Math.round(100 * f)}%`))
    if (current !== id) return false
    bar.hidden = true
    say(`${m.name} model ready (${m.params} parameters) · take or choose a photo`)
    return true
  } catch (e) {
    bar.hidden = true
    say(`could not load the model: ${(e as Error).message}`, true)
    return false
  }
}

async function init(): Promise<void> {
  try {
    models = await (await fetch('./model/models.json')).json()
  } catch {
    return say('no models found (model/models.json)', true)
  }
  $('models').innerHTML = models
    .map((m) => `<button role="radio" data-id="${m.id}" title="${m.note}"><b>${m.name}</b><small>${m.params}</small></button>`)
    .join('')
  $('models').querySelectorAll<HTMLButtonElement>('button').forEach((b) => (b.onclick = () => useModel(b.dataset.id!)))
  await useModel(models[0].id)
}

/** Draw an encoding on the board; animate traces it stroke by stroke in about 3 seconds. */
function drawKolam(enc: string[]): string {
  const h = fromEncoding(enc)
  const k = compile(h)
  const problems = validate(h)
  const board = new Board($<HTMLCanvasElement>('board'), '')
  redraw = (animate: boolean) => {
    const mine = run
    board.layout(h.rows, h.cols, 1) // room for crescents pointing off the grid
    board.clear()
    board.dots(CHALK.main, Math.max(2, board.s * 0.14), k.used) // unused dots are not drawn
    const w = Math.max(1.5, board.s * 0.14)
    const segs: [XY[], number][] = []
    drawn(k).strokes.forEach((P, si) => bezierSegments(board, P).forEach((seg, i) => segs.push([seg, si * 1009 + i * 7 + 1])))
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
  redraw(true)
  const lines = k.strokes.length
  const caption = `${h.rows} × ${h.cols} dots · ${lines} line${lines === 1 ? '' : 's'}`
  // a reading that breaks a kolam rule is still drawn, flagged plainly (the details go in the tooltip)
  $<HTMLElement>('caption').title = problems.map((p) => p.message).join('\n')
  return problems.length ? `${caption} · breaks a kolam rule, so likely misread` : caption
}

async function read(img: Blob): Promise<void> {
  if (!current || !(await useModel(current))) return
  run++
  const m = models.find((x) => x.id === current)!
  say('reading…')
  $('result').hidden = true
  let r
  try {
    r = await readKolam(img, base(current))
  } catch (e) {
    return say(`could not read it: ${(e as Error).message}`, true)
  }
  const seen = $<HTMLCanvasElement>('seen')
  ;[seen.width, seen.height] = [r.seen.width, r.seen.height]
  seen.getContext('2d')!.drawImage(r.seen, 0, 0)
  $('digits').textContent = r.encoding.length ? r.encoding.join('\n') : '(nothing)'
  $('result').hidden = false
  if (m.kolam_check && r.kolam < 0.5) {
    $<HTMLElement>('caption').textContent = 'not a kolam?'
    redraw = null
    return say(`this doesn't look like a kolam (${Math.round(100 * (1 - r.kolam))}% sure) · try a photo of a kolam`, true)
  }
  try {
    $<HTMLElement>('caption').textContent = drawKolam(r.encoding)
    say(`read in ${r.ms} ms with the ${m.name.toLowerCase()} model`)
  } catch (e) {
    redraw = null
    $<HTMLElement>('caption').textContent = 'not a drawable kolam'
    say(`it read something that isn't a drawable kolam (${(e as Error).message}) · try a straighter, closer photo`, true)
  }
  $('result').scrollIntoView({ behavior: 'smooth', block: 'start' })
}

function firstImage(items: DataTransferItemList | FileList | undefined | null): Blob | null {
  for (const it of Array.from<DataTransferItem | File>(items ?? [])) {
    const f = it instanceof File ? it : it.kind === 'file' ? it.getAsFile() : null
    if (f && f.type.startsWith('image/')) return f
  }
  return null
}

const drop = $<HTMLDivElement>('drop')
for (const id of ['camera', 'file']) {
  const input = $<HTMLInputElement>(id)
  input.onchange = () => {
    if (input.files?.[0]) read(input.files[0])
    input.value = '' // the same photo again still fires
  }
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
$('again').onclick = () => (run++, redraw?.(true))

let resize = 0
window.addEventListener('resize', () => {
  clearTimeout(resize)
  resize = window.setTimeout(() => redraw?.(false), 150)
})
init()
