// Kolam Core, the main page: take or pick a photo of a kolam, a trained model reads its encoding
// (spec §12) and the board draws that kolam back: the dots first, then each line traced at a calm,
// steady pen speed. The model runs in a Web Worker on the device (reader.ts; ONNX files from
// `python -m kolam_feed export`, listed in public/model/models.json), so the page never freezes and
// the built site is static files. One column on a phone, the board beside the photo on a wider screen.

import './style.css'
import { Board, CHALK, bezierSegments, type XY } from './board'
import { compile } from './kolam/compiler'
import { drawn } from './kolam/drawn'
import { fromEncoding } from './kolam/encoding'
import { validate } from './kolam/model'
import { GITHUB } from './links'
import { loadReader, readKolam } from './reader'

type ModelInfo = { id: string; name: string; params: string; note: string; kolam_check: boolean }
type Example = { file: string; dots: string; encoding: string[] } // public/examples: the truth is known

document.querySelector<HTMLDivElement>('#app')!.innerHTML = `
  <main class="rd">
    <header class="rd-head">
      <div class="rd-top">
        <span class="rd-badge"><span aria-hidden="true">✦</span> Fully trained AI model</span>
        <a class="rd-pill" href="./how.html">How it works <span aria-hidden="true">→</span></a>
      </div>
      <h1>Kolam Core</h1>
      <p>Show it a kolam. A fully trained AI model finds the dots, follows the lines, and draws the kolam
         back for you, right here on your device.</p>
    </header>

    <section class="rd-pick">
      <div class="rd-models" id="models" role="radiogroup" aria-label="model" hidden></div>
      <div class="rd-actions">
        <label class="rd-btn primary"><input id="camera" type="file" accept="image/*" capture="environment" hidden />
          <span aria-hidden="true">◉</span> Take a photo</label>
        <label class="rd-btn"><input id="file" type="file" accept="image/*" hidden />Choose an image</label>
      </div>
      <div class="rd-drop" id="drop">or drop an image here</div>
      <div class="rd-examples">
        <span>or try one of these</span>
        <div class="rd-strip" id="examples"></div>
      </div>
      <div class="rd-bar" id="bar" hidden><i></i></div>
      <div class="rd-status" id="status" role="status"></div>
    </section>

    <section class="rd-result" id="result" hidden>
      <figure class="rd-main">
        <div class="rd-frame" id="frame">
          <canvas id="board" class="board"></canvas>
          <img id="preview" alt="" />
        </div>
        <figcaption id="caption"></figcaption>
      </figure>
      <aside class="rd-side">
        <figure class="rd-thumb">
          <canvas id="seen"></canvas>
          <figcaption>your photo</figcaption>
        </figure>
        <button class="rd-btn" id="again">↻ Draw it again</button>
        <details class="rd-digits"><summary>the kolam's pattern code</summary><pre id="digits"></pre></details>
      </aside>
    </section>

    <footer class="rd-foot">
      <p>${GITHUB}</p>
      <p class="rd-small">Currently serving a smaller model for speed. A bigger, better model is on its way.</p>
    </footer>
  </main>`

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T
const status = $<HTMLDivElement>('status')
const bar = $<HTMLDivElement>('bar')
const caption = $<HTMLElement>('caption')
const preview = $<HTMLImageElement>('preview')
let models: ModelInfo[] = []
let current = ''
let run = 0 // bumps on every new reading so an old drawing stops
let redraw: ((animate: boolean) => void) | null = null
let examples: Example[] = []
let busy = false

function say(text: string, error = false): void {
  status.textContent = text
  status.classList.toggle('error', error)
}

const base = (id: string) => `./model/${id}/`
const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms))
const ease = (t: number) => (t < 0.5 ? 4 * t * t * t : 1 - (-2 * t + 2) ** 3 / 2) // ease in-out, cubic

async function useModel(id: string): Promise<boolean> {
  current = id
  document.querySelectorAll<HTMLButtonElement>('#models button').forEach((b) => b.setAttribute('aria-checked', String(b.dataset.id === id)))
  if (!busy) say('Getting ready…')
  bar.hidden = false
  const fill = bar.querySelector('i')!
  try {
    await loadReader(base(id), (f) => (fill.style.width = `${Math.round(100 * f)}%`))
    if (current !== id) return false
    bar.hidden = true
    if (!busy) say('')
    return true
  } catch (e) {
    console.error('loading the model:', e)
    bar.hidden = true
    say('Something went wrong while getting ready. Please reload the page.', true)
    return false
  }
}

async function init(): Promise<void> {
  try {
    models = await (await fetch('./model/models.json')).json()
  } catch {
    return say('Something went wrong while getting ready. Please reload the page.', true)
  }
  $('models').hidden = models.length < 2 // one model: no picker
  $('models').innerHTML = models
    .map((m) => `<button role="radio" data-id="${m.id}" title="${m.note}"><b>${m.name}</b></button>`)
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
    .map((e, i) => `<button data-i="${i}" aria-label="example kolam ${i + 1}">
        <img src="./examples/${e.file}" alt="" draggable="false" loading="lazy"></button>`)
    .join('')
  strip.querySelectorAll<HTMLButtonElement>('button').forEach((b) => (b.onclick = () => readExample(+b.dataset.i!, b)))
}

/** An example: its picture flies from the strip into the frame, then it is read like any photo. */
async function readExample(i: number, from?: HTMLElement): Promise<void> {
  if (busy) return
  const e = examples[i]
  const blob = await (await fetch(`./examples/${e.file}`)).blob()
  read(blob, e.encoding, from?.querySelector('img') ?? undefined)
}

/** A copy of `img` glides from where it is into the frame (following the page as it scrolls). */
function fly(img: HTMLImageElement): Promise<void> {
  const start = img.getBoundingClientRect()
  const ghost = img.cloneNode() as HTMLImageElement
  ghost.className = 'rd-ghost'
  document.body.append(ghost)
  const ms = matchMedia('(prefers-reduced-motion: reduce)').matches ? 0 : 750
  const t0 = performance.now()
  return new Promise((done) => {
    const step = (now: number) => {
      const f = ms ? Math.min(1, (now - t0) / ms) : 1
      const k = ease(f)
      const end = $('frame').getBoundingClientRect() // the target moves while the page scrolls
      const lerp = (a: number, b: number) => a + (b - a) * k
      Object.assign(ghost.style, {
        left: `${lerp(start.left, end.left)}px`, top: `${lerp(start.top, end.top)}px`,
        width: `${lerp(start.width, end.width)}px`, height: `${lerp(start.height, end.height)}px`,
        borderRadius: `${lerp(10, 14)}px`,
      })
      if (f < 1) return void requestAnimationFrame(step)
      ghost.remove()
      done()
    }
    requestAnimationFrame(step)
  })
}

/** What a reading is, in words: "A 5 × 5 kolam, drawn in a single line." */
function describe(rows: number, cols: number, lines: number): string {
  const size = `A ${rows} × ${cols} kolam`
  return lines === 1 ? `${size}, drawn in a single line.` : `${size}, woven from ${lines} lines.`
}

/** The rows read so far, drawn at once while the model reads on: the dot rows done make a smaller
 *  kolam, closed along its bottom edge, that grows downward in a square laid out for cols rows. */
function drawPartial(enc: string[]): void {
  const h = fromEncoding(enc)
  const k = compile(h)
  const board = new Board($<HTMLCanvasElement>('board'), '')
  board.layout(Math.max(h.rows, h.cols), h.cols, 1)
  board.clear()
  board.dots(CHALK.main, Math.max(2, board.s * 0.14), k.used)
  drawn(k).strokes.forEach((P, i) =>
    board.chalk(bezierSegments(board, P).flatMap((seg, j) => (j ? seg.slice(1) : seg)), CHALK.main, Math.max(1.5, board.s * 0.14), 1009 * i + 1),
  )
}

/** Draw an encoding on the board. animate: the dots appear one after another, then each line is traced
 *  at a slow, steady pen speed with a breath between lines (about 12-20 s in all); otherwise everything at once.
 *  Returns whether the reading breaks a kolam rule (then it is likely misread somewhere). */
function drawKolam(enc: string[], done: () => void): boolean {
  const h = fromEncoding(enc)
  const k = compile(h)
  const flawed = validate(h).length > 0
  const board = new Board($<HTMLCanvasElement>('board'), '')
  const dots = [...k.used].map((key) => key.split(',').map(Number)).sort((a, b) => a[0] - b[0] || a[1] - b[1])
  const lines = k.strokes.length
  const words = describe(h.rows, h.cols, lines)
  redraw = (animate: boolean) => {
    const mine = run
    board.layout(h.rows, h.cols, 1) // room for crescents pointing off the grid
    board.clear()
    const r = Math.max(2, board.s * 0.14)
    const w = Math.max(1.5, board.s * 0.14)
    // each line as one polyline in pixels, with its running length
    const paths = drawn(k).strokes.map((P) => {
      const pts: XY[] = bezierSegments(board, P).flatMap((seg, i) => (i ? seg.slice(1) : seg))
      const at = [0]
      for (let i = 1; i < pts.length; i++) at.push(at[i - 1] + Math.hypot(pts[i][0] - pts[i - 1][0], pts[i][1] - pts[i - 1][1]))
      return { pts, at, len: at[at.length - 1] }
    })
    if (!animate) {
      board.dots(CHALK.main, r, k.used)
      paths.forEach((p, i) => board.chalk(p.pts, CHALK.main, w, 1009 * i + 1))
      caption.textContent = words
      return
    }
    const gap = Math.min(200, 2600 / Math.max(1, dots.length)) // between dots
    const grow = 600 // each dot's own fade-in
    const dotsEnd = gap * (dots.length - 1) + grow
    const pause = 450 // a breath between lines
    const total = paths.reduce((n, p) => n + p.len, 0)
    const lineMs = Math.min(16000, Math.max(9000, 6000 + 1500 * lines)) // all lines, without the pauses
    const spans = paths.map((p) => Math.max(1200, (lineMs * p.len) / Math.max(total, 1)))
    let line = 0, drawnTo = 0, lineStart = dotsEnd + 200
    caption.textContent = 'Placing the dots…'
    const t0 = performance.now()
    const frame = (now: number) => {
      if (mine !== run) return
      const t = now - t0
      if (t < dotsEnd + 16) {
        // dots: each grows in with an ease, one after another
        dots.forEach(([i, j], n) => {
          const f = Math.min(1, Math.max(0, (t - n * gap) / grow))
          if (f > 0) board.disc([2 * i + 1, 2 * j + 1], r * (0.35 + 0.65 * ease(f)), CHALK.main, 0.25 + 0.75 * f)
        })
        return void requestAnimationFrame(frame)
      }
      if (line < paths.length) {
        if (t < lineStart) return void requestAnimationFrame(frame)
        if (drawnTo === 0) caption.textContent = lines === 1 ? 'Drawing the line…' : `Drawing the lines… ${line + 1} of ${lines}`
        const p = paths[line]
        const f = Math.min(1, (t - lineStart) / spans[line])
        const to = ease(f) * p.len
        let a = 0
        while (a + 1 < p.at.length && p.at[a + 1] <= drawnTo) a++
        let b = a
        while (b + 1 < p.at.length && p.at[b] < to) b++
        if (b > a) board.chalk(p.pts.slice(a, b + 1), CHALK.main, w, 1009 * line + b)
        drawnTo = to
        if (f >= 1) (line++, (drawnTo = 0), (lineStart = t + pause))
        return void requestAnimationFrame(frame)
      }
      caption.textContent = words
      done()
    }
    requestAnimationFrame(frame)
  }
  redraw(true)
  return flawed
}

/** Read a picture; `truth` (an example's real pattern) adds whether the reading matches it. */
async function read(img: Blob, truth?: string[], from?: HTMLImageElement): Promise<void> {
  if (busy || !current) return
  busy = true
  run++
  const mine = run
  // show the photo at once, softly scanned while the model reads it in the background
  const url = URL.createObjectURL(img)
  preview.src = url
  $('frame').classList.add('reading')
  $('frame').classList.remove('drawn')
  caption.textContent = 'Reading your kolam…'
  $('result').hidden = false
  say('')
  $('frame').scrollIntoView({ behavior: 'smooth', block: 'center' })
  if (from) {
    // the tapped example travels into the frame, then the frame shows it
    $('frame').classList.add('arriving')
    await fly(from)
    $('frame').classList.remove('arriving')
  }
  try {
    if (!(await useModel(current))) {
      $('frame').classList.remove('reading')
      caption.textContent = ''
      return
    }
    const m = models.find((x) => x.id === current)!
    // live: as the model reads, the kolam grows on the board row by row and its pattern code prints,
    // with the photo beside it
    const code = $<HTMLDetailsElement>('digits').parentElement as HTMLDetailsElement
    $('digits').textContent = ''
    const seen = $<HTMLCanvasElement>('seen')
    const bm = await createImageBitmap(img, { imageOrientation: 'from-image' })
    ;[seen.width, seen.height] = [bm.width, bm.height]
    seen.getContext('2d')!.drawImage(bm, 0, 0)
    bm.close()
    let live = false
    const r = await readKolam(img, base(current), m.kolam_check, (rows) => {
      if (mine !== run || rows.length < 3) return
      try {
        drawPartial(rows)
      } catch {
        return // an odd partial reading: wait for the next row
      }
      if (!live) ($('frame').classList.remove('reading'), $('frame').classList.add('drawn'))
      live = true
      code.open = true
      $('digits').textContent = rows.join('\n')
      $('digits').scrollTop = $('digits').scrollHeight // the newest row in view
      caption.textContent = `Reading row ${(rows.length - 1) / 2} of the dots…`
    })
    if (mine !== run) return
    ;[seen.width, seen.height] = [r.seen.width, r.seen.height]
    seen.getContext('2d')!.drawImage(r.seen, 0, 0)
    $('digits').textContent = r.encoding.length ? r.encoding.join('\n') : ''
    if (m.kolam_check && r.kolam < 0.5) {
      $('frame').classList.remove('reading')
      caption.textContent = 'This doesn’t look like a kolam.'
      return say('Try a photo of a kolam, taken from above.', true)
    }
    // the photo fades away and the board takes its place
    $('frame').classList.remove('reading')
    $('frame').classList.add('drawn')
    if (live) caption.textContent = 'Kolam done! Let me draw it for you…'
    await sleep(live ? 1800 : 450) // a live reading: a moment to see it whole, then it is traced again
    if (mine !== run) return
    let flawed = false
    try {
      flawed = drawKolam(r.encoding, () => {
        if (mine !== run) return
        if (truth) {
          const same = r.encoding.join('|') === truth.join('|')
          say(same ? 'Exactly the original kolam.' : 'Close to the original, with a few differences.')
        } else say(flawed ? 'Some parts were hard to read, so a line or two may be off.' : 'Here’s your kolam.')
      })
    } catch {
      redraw = null
      caption.textContent = 'This one was hard to read.'
      say('Try a clearer photo, taken straight from above.', true)
    }
  } catch (e) {
    console.error('reading the image:', e)
    $('frame').classList.remove('reading')
    caption.textContent = ''
    say('Couldn’t read that image. Please try another photo.', true)
  } finally {
    busy = false
    setTimeout(() => URL.revokeObjectURL(url), 2000)
  }
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
