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
const START = ['000', '0110', '111', '0110', '111', '0110', '000'] // 3 x 3 dots, every inner gate a cross
const NAMES = ['wall', 'cross', 'tube']
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
         kolam, one grid of digits. <b>Tap a gate</b> on the board, or a digit, to change it.</p>
      <div class="hw-play">
        <figure class="hw-board"><div class="rd-frame"><canvas id="play" class="board"></canvas></div>
          <figcaption id="playcap"></figcaption></figure>
        <div class="hw-digits">
          <div id="grid" class="hw-grid" aria-label="the kolam as digits"></div>
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
      <h2><span>4</span> Learning to read</h2>
      <div class="hw-flow" aria-label="photo, eyes, writer, digits, kolam">
        <div><img src="./how/look_4.jpg" alt=""><b>photo</b></div><i>→</i>
        <div><span class="hw-icon">◉</span><b>eyes</b><small>looks over the whole picture</small></div><i>→</i>
        <div><span class="hw-icon">✎</span><b>writer</b><small>writes the digits, one at a time</small></div><i>→</i>
        <div><code>0110<br>111<br>0110</code><b>digits</b></div><i>→</i>
        <div><img src="./how/look_1.jpg" alt=""><b>kolam</b></div>
      </div>
      <p>The model has two parts. The <b>eyes</b> look over the photo and note where the dots and lines
         are. The <b>writer</b> then writes the digits one by one; at each step, the rules of section 2
         only let it pick digits that still make a real kolam, so its answer is always a kolam.</p>
      <p>Training is practice with the answers at the back of the book. It is shown a generated photo,
         writes its guess, and is told how far each digit was from the true one; it adjusts a little and
         tries the next photo. Millions of photos later, we test it on kolams it has never seen, which is
         how the examples on the first page were chosen: it reads each of them exactly.</p>
      <p>A second, larger model is training now on bigger kolams, up to 20 × 20 dots, and on photos that
         look more like real floors.</p>
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

// 2 · the playground: a 3 x 3 kolam and its digits, linked both ways
let enc = [...START]
let mark: [number, number] | undefined
const play = new Board($<HTMLCanvasElement>('play'), '')
const inner = (i: number, k: number) => (i % 2 ? k > 0 && k < enc[i].length - 1 : i > 0 && i < enc.length - 1)

function render(): void {
  draw(play, enc, mark)
  $('grid').innerHTML = enc
    .map((row, i) => `<div class="hw-row ${i % 2 ? 'lr' : 'ud'}">${[...row].map((d, k) =>
      `<button data-i="${i}" data-k="${k}" class="d${d}${mark && mark[0] === i && mark[1] === k ? ' on' : ''}${inner(i, k) ? '' : ' edge'}"
        aria-label="gate row ${i + 1}, ${k + 1}: ${NAMES[+d] ?? 'horn'}">${d}</button>`).join('')}</div>`)
    .join('')
  $('grid').querySelectorAll<HTMLButtonElement>('button').forEach((b) => (b.onclick = () => flip(+b.dataset.i!, +b.dataset.k!)))
  const h = fromEncoding(enc)
  const lines = compile(h).strokes.length
  $('playcap').textContent = mark
    ? `that gate is now a ${NAMES[+enc[mark[0]][mark[1]]]}: ${lines} line${lines === 1 ? '' : 's'} in all`
    : `${h.rows} × ${h.cols} dots, ${lines} line${lines === 1 ? '' : 's'}`
}

/** Change one inner gate: wall -> cross -> tube -> wall (edge gates stay walls here). */
function flip(i: number, k: number): void {
  if (!inner(i, k)) return
  const before = enc[i]
  enc[i] = before.slice(0, k) + String((+before[k] + 1) % 3) + before.slice(k + 1)
  mark = [i, k]
  try {
    render()
  } catch {
    enc[i] = before // a combination that cannot be drawn: keep the old one
    render()
  }
}

$<HTMLCanvasElement>('play').addEventListener('click', (e) => {
  const r = (e.target as HTMLCanvasElement).getBoundingClientRect()
  const [x, y] = [e.clientX - r.left, e.clientY - r.top]
  let best: [number, number] | null = null
  let dist = Infinity
  enc.forEach((row, i) => [...row].forEach((_, k) => {
    if (!inner(i, k)) return
    const [gx, gy] = play.px(gapAt(i, k))
    const d = Math.hypot(gx - x, gy - y)
    if (d < dist) (dist = d), (best = [i, k])
  }))
  if (best && dist < play.s * 1.1) flip(best[0], best[1])
})
$('reset').onclick = () => ((enc = [...START]), (mark = undefined), render())

// 3 · the data
$('systems').innerHTML = SYSTEMS.map(([s, says]) => `<figure><img src="./how/system_${s}.jpg" alt="a kolam made by the ${s} system" loading="lazy">
  <figcaption><b>${s === 'wfc' ? 'tiles' : s}</b><small>${says}</small></figcaption></figure>`).join('')
$('looks').innerHTML = [1, 2, 3, 4, 5, 6].map((n) => `<img src="./how/look_${n}.jpg" alt="the same kolam, look ${n}" loading="lazy">`).join('')

function drawAll(): void {
  drawGates()
  render()
}
let resize = 0
window.addEventListener('resize', () => {
  clearTimeout(resize)
  resize = window.setTimeout(() => {
    gateBoards.forEach((b) => b.invalidateBase())
    play.invalidateBase()
    drawAll()
  }, 150)
})
document.fonts.ready.then(drawAll)
drawAll()
