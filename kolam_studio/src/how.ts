// How Kolam Core works: dots and gates, a kolam as a grid of digits (encoding <-> drawing, live and
// exact, spec §12), the generated data, and how the model learns to read it. Every chalkboard here is
// drawn from its encoding by the same code as the studio (fromEncoding -> compile -> drawn).

import './style.css'
import { Board, CHALK, bezierSegments } from './board'
import { compile } from './kolam/compiler'
import { drawn } from './kolam/drawn'
import { fromEncoding } from './kolam/encoding'

const GATES = [
  { enc: ['00', '010', '00', '010', '00'], d: '0', name: 'wall', says: 'the line turns back, so the two sides stay apart', mark: [2, 0] },
  { enc: ['00', '010', '00'], d: '1', name: 'cross', says: 'the lines cross over, from one dot to the next', mark: [1, 1] },
  { enc: ['00', '020', '00'], d: '2', name: 'tube', says: 'the lines run side by side, wrapping both dots', mark: [1, 1] },
  { enc: ['0', '03', '0'], d: '3–8', name: 'horn', says: 'a petal tip grows out, pointing one way' },
] as { enc: string[]; d: string; name: string; says: string; mark?: [number, number] }[]
// a 5 x 5 kolam from the generated data (enc_demo_100k #51371): crossings inside a tube band, with petal
// horns at the sides and corners; it uses every digit, 0 to 8
const START = ['70008', '012210', '11111', '001100', '11111', '611113', '11111', '001100', '11111', '012210', '50004']
const NAMES = ['wall', 'cross', 'tube', 'horn', 'horn curving left', 'horn curving right', 'horn', 'horn curving left', 'horn curving right']
const HINT_GATE: [number, number] = [5, 3] // the gate the "tap me" ring sits on: the centre
const SYSTEMS = [
  ['random', 'grows outward from a seed, gate by gate'],
  ['substitution', 'repeats a small motif inside itself'],
  ['switching', 'flips gates in a symmetric pattern'],
  ['wfc', 'pieces tiles together like a puzzle'],
]

document.querySelector<HTMLDivElement>('#app')!.innerHTML = `
  <main class="rd hw">
    <header class="rd-head">
      <nav class="rd-nav"><a href="./index.html">← Try it</a></nav>
      <h1>How it works</h1>
      <p>A kolam looks endlessly intricate, yet it follows a few simple rules. Kolam Core turns those rules
         into numbers, makes millions of kolams from them, and learns to read them back from a photo.</p>
    </header>

    <section class="hw-sec">
      <h2><span>1</span> Dots and gates</h2>
      <p>Every kolam starts as a grid of dots. Between any two neighbouring dots there is a gap, and the
         line does one of a few things there. We call each gap a <b>gate</b>, and give each kind a digit.</p>
      <div class="hw-gates" id="gates"></div>
    </section>

    <section class="hw-sec">
      <h2><span>2</span> A kolam is a grid of digits</h2>
      <p>Write down every gate's digit, row by row, and you have the whole kolam. Go the other way, follow
         the line through each gate, and the kolam draws itself back. Both directions are exact: one
         kolam, one grid of digits.</p>
      <div class="hw-hint" id="hint"><span aria-hidden="true">👆</span> Tap on the kolam and see it change</div>
      <div class="hw-play">
        <figure class="hw-board"><div class="rd-frame"><canvas id="play" class="board"></canvas>
          <span class="hw-tap" id="tap" aria-hidden="true"></span></div>
          <figcaption id="playcap"></figcaption></figure>
        <div class="hw-digits">
          <div id="grid" class="hw-grid" aria-label="the kolam as digits: tap one to change it"></div>
          <div class="hw-legend"><span class="d0">0 wall</span><span class="d1">1 cross</span><span class="d2">2 tube</span>
            <span class="d3">3–5 horn → or ↓</span><span class="d6">6–8 horn ← or ↑</span>
            <small>(each horn: plain, curving left, curving right)</small></div>
          <button class="rd-btn" id="reset">↺ start over</button>
        </div>
      </div>
      <div class="hw-math">
        <p>For a kolam with <b>R</b> rows and <b>C</b> columns of dots, the digits come in
           <b>2R + 1</b> rows, alternating <b>C</b> gates (between dots one above the other) and
           <b>C + 1</b> gates (between dots side by side):</p>
        <p class="hw-formula">gates = R(C + 1) + (R + 1)C</p>
        <p>A 3 × 3 kolam has 24 gates, a 5 × 5 has 60, a 20 × 20 has 840. Gates on the outer edge can only
           be a wall or a horn pointing outward, which is why so many digits there are 0.</p>
      </div>
    </section>

    <section class="hw-sec">
      <h2><span>3</span> Millions of kolams to learn from</h2>
      <p>To learn, a model needs kolams together with their exact digits. We make them: four small programs
         grow kolams from the rules above, and every kolam is checked against the kolam rules before it is
         kept. Around two million so far, from 3 × 3 up to 20 × 20 dots.</p>
      <div class="hw-four" id="systems"></div>
      <p>Then each kolam is drawn as if photographed: chalk or rice-flour lines, cement, red oxide or tiled
         floors, a tilted camera, sunlight and glare. The digits never change, only the look.</p>
      <div class="hw-looks" id="looks"></div>
    </section>

    <section class="hw-sec">
      <h2><span>4</span> The model that reads</h2>
      <p>Two networks, one after the other: a <b>ResNet</b> looks at the photo, and a <b>transformer
         decoder</b> writes the digits.</p>
      <div class="hw-arch">
        <figure class="hw-stage"><img src="./how/look_4.jpg" alt="a kolam photo"><figcaption><b>photo</b>
          <small>shrunk to a 192 × 192 square</small></figcaption></figure>
        <i class="hw-arrow">→</i>
        <figure class="hw-stage"><div class="hw-resnet"><span>edges</span><span>curves</span><span>dots &amp; gates</span></div>
          <figcaption><b>ResNet</b><small>the eyes: each layer sees a bigger patch, from strokes up to whole gates</small></figcaption></figure>
        <i class="hw-arrow">→</i>
        <figure class="hw-stage"><div class="hw-fmap" id="fmap"></div>
          <figcaption><b>a 12 × 12 map</b><small>what is where in the picture</small></figcaption></figure>
        <i class="hw-arrow">→</i>
        <figure class="hw-stage"><div class="hw-writer"><code id="typing"></code>
          <div class="hw-rule"><small>at the top edge, only these fit:</small><div id="rule"></div></div></div>
          <figcaption><b>transformer decoder</b><small>the writer: one digit at a time, looking back at the map
          and at what it has written; rules let through only digits that still make a kolam</small></figcaption></figure>
        <i class="hw-arrow">→</i>
        <figure class="hw-stage"><div class="rd-frame hw-mini"><canvas id="mini" class="board"></canvas></div>
          <figcaption><b>kolam</b><small>the digits, drawn back</small></figcaption></figure>
      </div>
    </section>

    <section class="hw-sec">
      <h2><span>5</span> How it learned</h2>
      <div class="hw-train">
        <div class="hw-loop" aria-label="the training loop">
          <div><b>1</b> a generated photo, with its true digits</div>
          <div><b>2</b> the model writes its guess</div>
          <div><b>3</b> each digit is compared with the truth</div>
          <div><b>4</b> the model is nudged to do a little better</div>
          <span class="hw-again">↻ and again, with the next photo</span>
        </div>
        <figure class="hw-chart"><svg id="chart" viewBox="0 0 360 220" role="img"
          aria-label="training: mistakes fall while exact readings rise"></svg>
          <figcaption id="chartcap"></figcaption></figure>
      </div>
      <p>This is the real record of the model on the first page. The yellow line is how wrong its guesses
         were; the green dots are how many kolams it had <b>never seen</b> it read exactly, checked along
         the way. A bigger model is now practising on millions of photos and kolams up to 20 × 20 dots.</p>
    </section>

    <section class="hw-sec">
      <h2><span>6</span> All inside Chrome</h2>
      <div class="hw-chrome">
        <div><span class="hw-icon">⬇</span><b>once</b><small>the model downloads to your browser the first time</small></div>
        <i class="hw-arrow">→</i>
        <div><span class="hw-icon">⚙</span><b>in Chrome</b><small>it runs on WebAssembly, in a background thread</small></div>
        <i class="hw-arrow">→</i>
        <div><span class="hw-icon">🔒</span><b>on your device</b><small>your photo never leaves it</small></div>
      </div>
      <p>The models are optimized to run in Chrome: converted to a compact format (ONNX) and run by
         WebAssembly in a background thread, so everything happens inside Chrome itself. There is no
         server doing the work, and the page stays smooth while the model reads.</p>
    </section>

    <footer class="rd-foot">
      <p><a href="./index.html">← Try it on a kolam</a> · <a href="https://github.com/sivOclock/Kolam_Core" target="_blank" rel="noopener">GitHub</a></p>
    </footer>
  </main>`

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T

/** Draw an encoding on a board; `mark` rings one gate (encoding row, digit index). */
function draw(board: Board, enc: string[], mark?: [number, number]): void {
  const h = fromEncoding(enc)
  const k = compile(h)
  board.layout(h.rows, h.cols, 1)
  board.clear()
  board.dots(CHALK.main, Math.max(2, board.s * 0.14), new Set(Array.from({ length: h.rows * h.cols }, (_, n) => `${Math.floor(n / h.cols)},${n % h.cols}`)))
  const w = Math.max(1.5, board.s * 0.14)
  drawn(k).strokes.forEach((P, si) => bezierSegments(board, P).forEach((seg, i) => board.chalk(seg, CHALK.main, w, si * 1009 + i * 7 + 1)))
  if (mark) {
    const [x, y] = board.px(gapAt(mark[0], mark[1]))
    const ctx = board.ctx
    ctx.strokeStyle = CHALK.highlight
    ctx.lineWidth = 2.5
    ctx.beginPath()
    ctx.arc(x, y, board.s * 0.55, 0, Math.PI * 2)
    ctx.stroke()
  }
}

/** The lattice point of digit k in encoding row i (odd rows: left/right gaps at even columns). */
const gapAt = (i: number, k: number): [number, number] => [i, i % 2 ? 2 * k : 2 * k + 1]

// 1 · the gates, one small board each
$('gates').innerHTML = GATES.map((g, n) => `<figure class="hw-gate"><div class="rd-frame"><canvas id="g${n}" class="board"></canvas></div>
  <figcaption><b>${g.d}</b> ${g.name}<small>${g.says}</small></figcaption></figure>`).join('')
const gateBoards = GATES.map((_, n) => new Board($<HTMLCanvasElement>(`g${n}`), ''))
const drawGates = () => GATES.forEach((g, n) => draw(gateBoards[n], g.enc, g.mark))

// 2 · the playground: a 5 x 5 kolam and its digits, linked both ways
let enc = [...START]
let mark: [number, number] | undefined
let touched = false // until the first tap, a ring and a hint say "tap me"
const play = new Board($<HTMLCanvasElement>('play'), '')
const last = () => enc.length - 1

/** The digits gate (i, k) can hold: inside, any of 0-8; on the outer edge, a wall or a horn pointing
 *  out of the grid (top and left edges: 6-8, pointing up / left; bottom and right: 3-5). */
function choices(i: number, k: number): number[] {
  const out = (up: boolean) => [0, ...(up ? [6, 7, 8] : [3, 4, 5])]
  if (i % 2 === 0) return i === 0 ? out(true) : i === last() ? out(false) : [0, 1, 2, 3, 4, 5, 6, 7, 8]
  return k === 0 ? out(true) : k === enc[i].length - 1 ? out(false) : [0, 1, 2, 3, 4, 5, 6, 7, 8]
}

/** What a digit means at gate (i, k), in words. */
function named(d: number, i: number): string {
  if (d < 3) return NAMES[d]
  const way = i % 2 ? (d < 6 ? 'right' : 'left') : d < 6 ? 'down' : 'up'
  return `${NAMES[d]}, pointing ${way}`
}

function render(): void {
  draw(play, enc, mark)
  $('grid').innerHTML = enc
    .map((row, i) => `<div class="hw-row ${i % 2 ? 'lr' : 'ud'}">${[...row].map((d, k) => {
      const hint = !touched && i === HINT_GATE[0] && k === HINT_GATE[1] ? ' pulse' : ''
      const on = mark && mark[0] === i && mark[1] === k ? ' on' : ''
      const edge = choices(i, k).length < 9 ? ' edge' : ''
      return `<button data-i="${i}" data-k="${k}" class="d${d}${on}${edge}${hint}" aria-label="gate ${i + 1}.${k + 1}: ${named(+d, i)}">${d}</button>`
    }).join('')}</div>`)
    .join('')
  $('grid').querySelectorAll<HTMLButtonElement>('button').forEach((b) => (b.onclick = () => flip(+b.dataset.i!, +b.dataset.k!)))
  $<HTMLButtonElement>('reset').disabled = enc.join() === START.join() // nothing to undo yet
  const h = fromEncoding(enc)
  const lines = compile(h).strokes.length
  $('playcap').textContent = mark
    ? `that gate is now a ${named(+enc[mark[0]][mark[1]], mark[0])}: ${lines} line${lines === 1 ? '' : 's'} in all`
    : `${h.rows} × ${h.cols} dots, ${lines} line${lines === 1 ? '' : 's'}`
  // the "tap me" ring sits on the hint gate until the first tap
  const tap = $('tap')
  tap.hidden = touched
  const [x, y] = play.px(gapAt(HINT_GATE[0], HINT_GATE[1]))
  Object.assign(tap.style, { left: `${x}px`, top: `${y}px`, width: `${play.s * 1.3}px`, height: `${play.s * 1.3}px` })
}

/** Step gate (i, k) to its next digit that can be drawn (one that can't is skipped). */
function flip(i: number, k: number): void {
  if (!touched) ((touched = true), ($('hint').innerHTML = 'Keep tapping, and watch the kolam and its numbers change together'))
  const before = enc[i]
  const opts = choices(i, k)
  const at = opts.indexOf(+before[k])
  for (let n = 1; n <= opts.length; n++) {
    const d = opts[(at + n) % opts.length]
    enc[i] = before.slice(0, k) + d + before.slice(k + 1)
    try {
      compile(fromEncoding(enc)) // drawable?
      break
    } catch {
      enc[i] = before
    }
  }
  mark = [i, k]
  render()
}

$<HTMLCanvasElement>('play').addEventListener('click', (e) => {
  const r = (e.target as HTMLCanvasElement).getBoundingClientRect()
  const [x, y] = [e.clientX - r.left, e.clientY - r.top]
  let best: [number, number] | null = null
  let dist = Infinity
  enc.forEach((row, i) => [...row].forEach((_, k) => {
    const [gx, gy] = play.px(gapAt(i, k))
    const d = Math.hypot(gx - x, gy - y)
    if (d < dist) (dist = d), (best = [i, k])
  }))
  if (best && dist < play.s * 1.1) flip(best[0], best[1])
})
/** Back to the original kolam: the board comes into view and the kolam fades back in. */
$('reset').onclick = () => {
  enc = [...START]
  mark = undefined
  const c = $('play')
  c.scrollIntoView({ behavior: 'smooth', block: 'center' })
  c.classList.remove('hw-fade')
  void c.offsetWidth // restart the fade
  c.classList.add('hw-fade')
  render()
}

// 3 · the data
$('systems').innerHTML = SYSTEMS.map(([s, says]) => `<figure><img src="./how/system_${s}.jpg" alt="a kolam made by the ${s} system" loading="lazy">
  <figcaption><b>${s === 'wfc' ? 'tiles' : s}</b><small>${says}</small></figcaption></figure>`).join('')
$('looks').innerHTML = [1, 2, 3, 4, 5, 6].map((n) => `<img src="./how/look_${n}.jpg" alt="the same kolam, look ${n}" loading="lazy">`).join('')

// 4 · the model: a lit-up feature map, the writer typing the digits, the rule at the top edge
$('fmap').innerHTML = Array.from({ length: 144 }, (_, n) => {
  const [r, c] = [Math.floor(n / 12), n % 12]
  const on = r > 1 && r < 10 && c > 1 && c < 10 && (r + c) % 2 === 0
  return `<i style="opacity:${on ? 0.55 + 0.45 * Math.abs(Math.sin(n)) : 0.08 + 0.1 * Math.abs(Math.cos(n))}"></i>`
}).join('')
$('rule').innerHTML = [...'012345678'].map((d) => `<b class="${'0678'.includes(d) ? 'ok' : ''}">${d}</b>`).join('')
const mini = new Board($<HTMLCanvasElement>('mini'), '')
const typing = $('typing')
const calm = matchMedia('(prefers-reduced-motion: reduce)').matches
let typed = 0
const text = START.join('\n')
function type(): void {
  if (calm) return void (typing.textContent = text)
  typed = typed >= text.length + 24 ? 0 : typed + 1 // a short rest at the end, then again
  typing.textContent = text.slice(0, Math.min(typed, text.length)) + (typed < text.length ? '▍' : '')
}
setInterval(type, 140)

// 5 · the real training record of the served model
async function chart(): Promise<void> {
  const t: { loss: [number, number][]; val: [number, number, number][]; images_per_step: number } =
    await (await fetch('./how/training.json')).json()
  const W = 360, H = 220, L = 34, R = 34, T = 14, B = 30
  const xmax = t.loss[t.loss.length - 1][0], lmax = Math.max(...t.loss.map((p) => p[1]))
  const X = (s: number) => L + (s / xmax) * (W - L - R)
  const YL = (v: number) => T + (1 - v / lmax) * (H - T - B)
  const YA = (v: number) => T + (1 - v) * (H - T - B)
  let g = ''
  for (const f of [0, 0.5, 1]) g += `<line x1="${L}" x2="${W - R}" y1="${YA(f)}" y2="${YA(f)}" class="grid"/>
    <text x="${W - R + 4}" y="${YA(f) + 4}" class="acc">${Math.round(100 * f)}%</text>`
  g += `<path d="${t.loss.map(([s, v], i) => `${i ? 'L' : 'M'}${X(s).toFixed(1)},${YL(v).toFixed(1)}`).join('')}" class="loss"/>`
  g += t.val.map(([s, e]) => `<circle cx="${X(s)}" cy="${YA(e)}" r="5" class="exact"/>
    <text x="${X(s)}" y="${YA(e) - 10}" class="acc" text-anchor="middle">${Math.round(100 * e)}%</text>`).join('')
  const imgs = (s: number) => (s * t.images_per_step).toLocaleString()
  g += `<text x="${L}" y="${H - 8}" class="ax">0</text><text x="${W - R}" y="${H - 8}" class="ax" text-anchor="end">${imgs(xmax)} photos</text>
        <text x="${L - 6}" y="${T + 4}" class="lossl" text-anchor="end">wrong</text>`
  $('chart').innerHTML = g
  const [first, last] = [t.val[0], t.val[t.val.length - 1]]
  $('chartcap').textContent = `unseen kolams read exactly: ${Math.round(100 * first[1])}% after ${imgs(first[0])} photos, ` +
    `${Math.round(100 * last[1])}% after ${imgs(last[0])}`
}
chart()

function drawAll(): void {
  drawGates()
  render()
  draw(mini, START)
}
let resize = 0
window.addEventListener('resize', () => {
  clearTimeout(resize)
  resize = window.setTimeout(() => {
    gateBoards.forEach((b) => b.invalidateBase())
    play.invalidateBase()
    mini.invalidateBase()
    drawAll()
  }, 150)
})
document.fonts.ready.then(drawAll)
drawAll()
