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
type Example = { file: string; dots: string; encoding: string[] } // public/examples: the truth is known

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
      <div class="rd-examples">
        <span>or try one of these (tap, or drag it onto the box above):</span>
        <div class="rd-strip" id="examples"></div>
      </div>
      <div class="rd-bar" id="bar" hidden><i></i></div>
      <div class="rd-status" id="status" role="status"></div>
    </section>

    <section class="rd-result" id="result" hidden>
      <figure class="rd-main">
        <div class="rd-frame"><canvas id="board" class="board"></canvas></div>
        <figcaption id="caption">what it read</figcaption>
      </figure>
      <aside class="rd-side">
        <figure class="rd-thumb">
          <canvas id="seen"></canvas>
          <figcaption>the photo, as the model saw it</figcaption>
        </figure>
        <button class="rd-btn" id="again">↻ draw it again</button>
        <details class="rd-digits"><summary>the digits it read</summary><pre id="digits"></pre></details>
      </aside>
    </section>

    <footer class="rd-foot">
      Works best on dot kolams up to 5 × 5 dots, photographed from above. Early models, trained on drawn
      kolams; bigger kolams are next. <a href="./studio.html">Draw your own kolam</a>
    </footer>
  </main>`

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T
const status = $<HTMLDivElement>('status')
const bar = $<HTMLDivElement>('bar')
let models: ModelInfo[] = []
let current = ''
let run = 0 // bumps on every new reading so an old trace stops
let redraw: ((animate: boolean) => void) | null = null
let examples: Example[] = []

function say(text: string, error = false): void {
  status.textContent = text
  status.classList.toggle('error', error)
}

const base = (id: string) => `./model/${id}/`

async function useModel(id: string): Promise<boolean> {
  current = id
  document.querySelectorAll<HTMLButtonElement>('#models button').forEach((b) => b.setAttribute('aria-checked', String(b.dataset.id === id)))
  const m = models.find((x) => x.id === id)!
  say(`loading the model (${m.params} parameters, once)…`)
  bar.hidden = false
  const fill = bar.querySelector('i')!
  fill.style.width = '0%'
  try {
    await loadReader(base(id), (f) => (fill.style.width = `${Math.round(100 * f)}%`))
    if (current !== id) return false
    bar.hidden = true
    say(`model ready (${m.params} parameters, on this device) · take or choose a photo`)
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
  $('models').hidden = models.length < 2 // one model: no picker
  $('models').innerHTML = models
    .map((m) => `<button role="radio" data-id="${m.id}" title="${m.note}"><b>${m.name}</b><small>${m.params}</small></button>`)
    .join('')
  $('models').querySelectorAll<HTMLButtonElement>('button').forEach((b) => (b.onclick = () => useModel(b.dataset.id!)))
  loadExamples()
  await useModel(models[0].id)
}

async function loadExamples(): Promise<void> {
  try {
    examples = await (await fetch('./examples/examples.json')).json()
  } catch {
    return
  }
  const strip = $('examples')
  strip.innerHTML = examples
    .map((e, i) => `<button data-i="${i}" draggable="true" aria-label="example kolam ${i + 1}, ${e.dots} dots">
        <img src="./examples/${e.file}" alt="" draggable="false" loading="lazy"></button>`)
    .join('')
  strip.querySelectorAll<HTMLButtonElement>('button').forEach((b) => {
    b.onclick = () => readExample(+b.dataset.i!)
    b.ondragstart = (e) => e.dataTransfer?.setData('text/x-kolam-example', b.dataset.i!)
  })
}

async function readExample(i: number): Promise<void> {
  const e = examples[i]
  read(await (await fetch(`./examples/${e.file}`)).blob(), e.encoding)
}

/** Draw an encoding on the board. animate: the dots appear one by one, then each line is traced in turn
 *  (about 4 s whatever the size); otherwise everything at once (on resize). */
function drawKolam(enc: string[]): string {
  const h = fromEncoding(enc)
  const k = compile(h)
  const problems = validate(h)
  const board = new Board($<HTMLCanvasElement>('board'), '')
  const caption = $<HTMLElement>('caption')
  const lines = k.strokes.length
  const dots = [...k.used].map((key) => key.split(',').map(Number)).sort((a, b) => a[0] - b[0] || a[1] - b[1])
  const done = `${h.rows} × ${h.cols} dots · ${lines} line${lines === 1 ? '' : 's'}` +
    (problems.length ? ' · breaks a kolam rule, so likely misread' : '')
  caption.title = problems.map((p) => p.message).join('\n') // the details, for whoever hovers
  redraw = (animate: boolean) => {
    const mine = run
    board.layout(h.rows, h.cols, 1) // room for crescents pointing off the grid
    board.clear()
    const r = Math.max(2, board.s * 0.14)
    const w = Math.max(1.5, board.s * 0.14)
    const strokes = drawn(k).strokes.map((P, si) => bezierSegments(board, P).map((seg, i): [XY[], number] => [seg, si * 1009 + i * 7 + 1]))
    if (!animate) {
      board.dots(CHALK.main, r, k.used)
      strokes.flat().forEach(([seg, seed]) => board.chalk(seg, CHALK.main, w, seed))
      caption.textContent = done
      return
    }
    const dotMs = Math.min(1200, 45 * dots.length) // dots: up to 1.2 s
    const total = strokes.reduce((n, s) => n + s.length, 0)
    const perFrame = Math.max(1, Math.ceil(total / 160)) // lines: ~160 frames
    let shown = 0, line = 0, seg = 0
    const t0 = performance.now()
    const frame = (now: number) => {
      if (mine !== run) return
      const due = dotMs ? Math.min(dots.length, Math.floor(((now - t0) / dotMs) * dots.length) + 1) : dots.length
      for (; shown < due; shown++) {
        const [i, j] = dots[shown]
        board.disc([2 * i + 1, 2 * j + 1], r, CHALK.main) // each dot pressed into the floor
        caption.textContent = `placing the dots… ${shown + 1} of ${dots.length}`
      }
      if (shown < dots.length) return void requestAnimationFrame(frame)
      for (let n = 0; n < perFrame && line < strokes.length; n++) {
        const [pts, seed] = strokes[line][seg]
        board.chalk(pts, CHALK.main, w, seed)
        if (++seg === strokes[line].length) (line++, (seg = 0))
      }
      if (line < strokes.length) {
        caption.textContent = `drawing line ${line + 1} of ${lines}…`
        return void requestAnimationFrame(frame)
      }
      caption.textContent = done
    }
    requestAnimationFrame(frame)
  }
  redraw(true)
  return done
}

/** Read a picture; `truth` (an example's real encoding) adds whether the reading matches it. */
async function read(img: Blob, truth?: string[]): Promise<void> {
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
    drawKolam(r.encoding)
    const verdict = !truth ? '' : r.encoding.join('|') === truth.join('|') ? ' · ✓ matches the real kolam exactly'
      : ` · ✗ differs from the real kolam in ${diff(r.encoding, truth)}`
    say(`read in ${(r.ms / 1000).toFixed(1)} s${verdict}`)
  } catch (e) {
    redraw = null
    $<HTMLElement>('caption').textContent = 'not a drawable kolam'
    say(`it read something that isn't a drawable kolam (${(e as Error).message}) · try a straighter, closer photo`, true)
  }
  $('board').scrollIntoView({ behavior: 'smooth', block: 'center' })
}

/** How a reading differs from the truth: gaps that differ, or that the grid size is off. */
function diff(a: string[], b: string[]): string {
  if (a.length !== b.length || a.some((row, k) => row.length !== b[k].length)) return 'its grid size'
  const n = a.reduce((s, row, k) => s + [...row].filter((d, j) => d !== b[k][j]).length, 0)
  return `${n} gap${n === 1 ? '' : 's'}`
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
  const ex = e.dataTransfer?.getData('text/x-kolam-example')
  if (ex) return void readExample(+ex)
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
