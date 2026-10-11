import './style.css'
import { Board, CHALK, bezierSegments, type XY } from './board'
import { type Compiled, compile } from './kolam/compiler'
import {
  type Hypergraph,
  type HornVariant,
  type Problem,
  type Relation,
  type Vertex,
  KolamError,
  adjacent,
  canonical,
  formatProgram,
  parse,
  toLattice,
  validate,
} from './kolam/model'

// ── brushes ────────────────────────────────────────────────────────────────
type BrushKey = 'c' | 'p' | 'h' | 'hl' | 'hr'
const BRUSHES: { key: BrushKey; hotkey: string; name: string; colour: string }[] = [
  { key: 'c', hotkey: 'c', name: 'crossing', colour: CHALK.pink },
  { key: 'p', hotkey: 'p', name: 'parallel', colour: CHALK.cyan },
  { key: 'h', hotkey: 'h', name: 'horn', colour: CHALK.green },
  { key: 'hl', hotkey: 'l', name: 'left', colour: CHALK.green },
  { key: 'hr', hotkey: 'r', name: 'right', colour: CHALK.green },
]
const VARIANT: Record<string, HornVariant> = { h: 'PLAIN_HORN', hl: 'LEFT_HORN', hr: 'RIGHT_HORN' }
const isHorn = (b: BrushKey) => b in VARIANT
const relColour = (r: Relation) =>
  r.type === 'CROSSING_LOOP' ? CHALK.pink : r.type === 'PARALLEL_LOOP' ? CHALK.cyan : CHALK.green
const relName = (r: Relation) =>
  r.type === 'CROSSING_LOOP' ? 'c' : r.type === 'PARALLEL_LOOP' ? 'p' : { PLAIN_HORN: 'h', LEFT_HORN: 'hl', RIGHT_HORN: 'hr' }[r.variant!]

// ── page ───────────────────────────────────────────────────────────────────
const ARROW = `<svg width="28" height="48" viewBox="0 0 36 48" fill="none"><path d="M4 24 Q12 22 24 24" stroke="#FACC15" stroke-width="2.5" stroke-linecap="round" opacity="0.85"/><path d="M24 24 L17 17" stroke="#FACC15" stroke-width="2.5" stroke-linecap="round" opacity="0.85"/><path d="M24 24 L17 31" stroke="#FACC15" stroke-width="2.5" stroke-linecap="round" opacity="0.85"/></svg>`

document.querySelector<HTMLDivElement>('#app')!.innerHTML = `
  <div class="layout">
    <canvas id="draw" class="board" aria-label="Drawing board: drag across dots to draw brush strokes"></canvas>
    <div class="flow-arrow">${ARROW}</div>
    <aside class="panel">
      <div class="panel-top">
        <span class="title">formula</span>
        <button id="undo" title="Undo (Ctrl+Z)">undo</button>
        <button id="redo" title="Redo (Ctrl+Shift+Z)">redo</button>
        <button id="clear">clear</button>
        <button id="copy">copy</button>
      </div>
      <div class="grid-input">
        <span class="label">grid</span>
        <label>N <input id="rows" type="number" min="1" max="16" /></label>
        <label>M <input id="cols" type="number" min="1" max="16" /></label>
      </div>
      <div class="brushes" role="radiogroup" aria-label="Brush">
        ${BRUSHES.map(
          (b) => `<button class="brush" data-brush="${b.key}" role="radio" title="${b.name} (${b.hotkey})">
            <b>${b.key}</b><i style="background:${b.colour}"></i><span>${b.name}</span></button>`,
        ).join('')}
      </div>
      <textarea id="program" spellcheck="false" aria-label="Brush program"></textarea>
      <div id="status" class="status" role="status"></div>
      <span class="label">relations</span>
      <ul id="relations" class="relations"></ul>
      <p id="hint" class="hint"></p>
    </aside>
    <div class="flow-arrow">${ARROW}</div>
    <canvas id="kolam" class="board" aria-label="The compiled kolam"></canvas>
  </div>`

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T
const drawCanvas = $<HTMLCanvasElement>('draw')
const drawBoard = new Board(drawCanvas, 'draw')
const kolamBoard = new Board($<HTMLCanvasElement>('kolam'), 'kolam')
const programEl = $<HTMLTextAreaElement>('program')
const rowsEl = $<HTMLInputElement>('rows')
const colsEl = $<HTMLInputElement>('cols')
const statusEl = $<HTMLDivElement>('status')
const listEl = $<HTMLUListElement>('relations')
const hintEl = $<HTMLParagraphElement>('hint')

// ── state ──────────────────────────────────────────────────────────────────
interface Snapshot { rows: number; cols: number; rels: Relation[] }
interface Pending { brush: BrushKey; dots: Vertex[]; closed: boolean }

let rows = 5
let cols = 5
let rels: Relation[] = []
let brush: BrushKey = 'c'
let selected: number | null = null
let pending: Pending | null = null
let hover: Vertex | null = null
let cursor: XY | null = null // pointer on the draw board, for the rubber-band line
let lastGood: { h: Hypergraph; k: Compiled } | null = null
let textError: string | null = null
let history: string[] = []
let at = -1

const snap = (): string => JSON.stringify({ rows, cols, rels } satisfies Snapshot)
function commit(): void {
  const s = snap()
  if (history[at] === s) return
  history = history.slice(0, at + 1)
  history.push(s)
  at = history.length - 1
}
function restore(s: string): void {
  const v = JSON.parse(s) as Snapshot
  ;({ rows, cols, rels } = v)
  selected = null
  pending = null
  textError = null
  render(true)
}

// ── geometry helpers ───────────────────────────────────────────────────────
const eq = (a: Vertex, b: Vertex) => a[0] === b[0] && a[1] === b[1]

function dotAt(x: number, y: number): Vertex | null {
  const [li, lj] = drawBoard.lattice(x, y)
  const r = Math.round((li - 1) / 2)
  const c = Math.round((lj - 1) / 2)
  if (r < 0 || r >= rows || c < 0 || c >= cols) return null
  return Math.hypot(li - (2 * r + 1), lj - (2 * c + 1)) < 0.8 ? [r, c] : null
}

/** What you "draw" for a relation, in lattice coords: the line through its dots; a horn is an arrow. */
function relShape(r: Relation): XY[][] {
  const L = r.vertices.map(toLattice)
  if (r.type !== 'HORN') return [r.closed ? [...L, L[0]] : L]
  const [a, b] = L
  const f: XY = [(b[0] - a[0]) / 2, (b[1] - a[1]) / 2]
  const left: XY = [-f[1], f[0]]
  const tip: XY = [b[0] + 0.55 * f[0], b[1] + 0.55 * f[1]]
  const sides = r.variant === 'LEFT_HORN' ? [1] : r.variant === 'RIGHT_HORN' ? [-1] : [1, -1]
  return [
    [a, tip],
    ...sides.map((sg): XY[] => [tip, [tip[0] - 0.6 * f[0] + sg * 0.5 * left[0], tip[1] - 0.6 * f[1] + sg * 0.5 * left[1]]]),
  ]
}

function segDist(p: XY, a: XY, b: XY): number {
  const dx = b[0] - a[0]
  const dy = b[1] - a[1]
  const t = Math.max(0, Math.min(1, ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / (dx * dx + dy * dy || 1)))
  return Math.hypot(p[0] - a[0] - t * dx, p[1] - a[1] - t * dy)
}

function relationNear(x: number, y: number): number | null {
  const p = drawBoard.lattice(x, y)
  let best: number | null = null
  let bestD = 0.55
  rels.forEach((r, i) => {
    for (const line of relShape(r))
      for (let k = 0; k + 1 < line.length; k++) {
        const d = segDist(p, line[k], line[k + 1])
        if (d < bestD) [best, bestD] = [i, d]
      }
  })
  return best
}

function pendingRelation(p: Pending): Relation {
  return isHorn(p.brush)
    ? { type: 'HORN', vertices: p.dots.slice(0, 2), variant: VARIANT[p.brush] }
    : { type: p.brush === 'c' ? 'CROSSING_LOOP' : 'PARALLEL_LOOP', vertices: p.dots, closed: p.closed }
}

/** Grow the stroke being drawn toward dot d (fills in straight runs, allows backtracking). */
function extend(p: Pending, d: Vertex | null): boolean {
  if (!d || p.closed) return false
  const last = p.dots[p.dots.length - 1]
  if (eq(d, last)) return false
  if (isHorn(p.brush)) {
    const a = p.dots[0]
    if (eq(d, a)) p.dots = [a]
    else if (adjacent(a, d)) p.dots = [a, d]
    else return false
    return true
  }
  if (p.dots.length >= 2 && eq(d, p.dots[p.dots.length - 2])) {
    p.dots.pop()
    return true
  }
  // closing onto the first dot makes a ring
  if (p.dots.length >= 4 && eq(d, p.dots[0]) && adjacent(last, d)) {
    p.closed = true
    return true
  }
  if (last[0] !== d[0] && last[1] !== d[1]) return false // brushes run along rows and columns
  const dr = Math.sign(d[0] - last[0])
  const dc = Math.sign(d[1] - last[1])
  let cur = last
  let grew = false
  while (!eq(cur, d)) {
    cur = [cur[0] + dr, cur[1] + dc]
    if (p.dots.some((v) => eq(v, cur))) break
    p.dots.push(cur)
    grew = true
  }
  return grew
}

// ── rendering ──────────────────────────────────────────────────────────────
function renderDraw(all: Relation[], bad: Set<number>): void {
  const B = drawBoard
  B.layout(rows, cols)
  B.clear()
  B.dots(CHALK.main, Math.max(3, B.s * 0.16))
  const w = Math.max(3, B.s * 0.2)
  all.forEach((r, i) => {
    const isPending = pending !== null && i === all.length - 1
    const colour = bad.has(i) ? CHALK.red : i === selected ? CHALK.highlight : relColour(r)
    for (const line of relShape(r)) B.chalk(line.map((p) => B.px(p)), colour, isPending ? w * 1.15 : w, i * 31 + 7, bad.has(i))
    if (r.type !== 'HORN' && (i === selected || isPending))
      for (const v of r.vertices) B.disc(toLattice(v), B.s * 0.32, colour, 0.25)
  })
  if (hover && !pending) B.disc(toLattice(hover), B.s * 0.42, CHALK.green, 0.4)
  if (pending) {
    // where the stroke stands: its last dot, and a faint line to the pointer
    const colour = BRUSHES.find((b) => b.key === pending!.brush)!.colour
    const last = toLattice(pending.dots[pending.dots.length - 1])
    B.disc(last, B.s * 0.42, colour, 0.35)
    if (cursor && !pending.closed) {
      const ctx = B.ctx
      const [x0, y0] = B.px(last)
      ctx.save()
      ctx.globalAlpha = 0.45
      ctx.strokeStyle = colour
      ctx.lineWidth = 1.5
      ctx.setLineDash([5, 6])
      ctx.beginPath()
      ctx.moveTo(x0, y0)
      ctx.lineTo(cursor[0], cursor[1])
      ctx.stroke()
      ctx.restore()
    }
  }
}

function renderKolam(k: Compiled | null, h: Hypergraph | null, stale: boolean, selRel: number | null): void {
  const B = kolamBoard
  B.layout(h?.rows ?? rows, h?.cols ?? cols)
  B.clear()
  B.dots(stale ? CHALK.dots : CHALK.main, Math.max(3, B.s * 0.14))
  if (!k) return
  const w = Math.max(2.5, B.s * 0.14)
  k.strokes.forEach((P, si) => {
    bezierSegments(B, P).forEach((seg, i) => {
      const own = k.owners[si][i]
      const colour = stale ? CHALK.dots : own >= 0 && own === selRel ? CHALK.highlight : CHALK.main
      B.chalk(seg, colour, w, si * 1009 + i * 7 + 1)
    })
  })
}

function renderPanel(all: Relation[], problems: Problem[], k: Compiled | null): void {
  if (document.activeElement !== programEl) {
    programEl.value = formatProgram({ rows, cols, relations: all })
    programEl.classList.toggle('invalid', problems.length > 0)
  }
  if (document.activeElement !== rowsEl) rowsEl.value = String(rows)
  if (document.activeElement !== colsEl) colsEl.value = String(cols)
  document.querySelectorAll<HTMLButtonElement>('.brush').forEach((b) => {
    const on = b.dataset.brush === brush
    b.classList.toggle('active', on)
    b.setAttribute('aria-checked', String(on))
  })
  $<HTMLButtonElement>('undo').disabled = at <= 0
  $<HTMLButtonElement>('redo').disabled = at >= history.length - 1

  if (textError) {
    statusEl.className = 'status error'
    statusEl.textContent = textError
  } else if (problems.length) {
    statusEl.className = 'status error'
    statusEl.textContent = problems.slice(0, 3).map((p) => p.message).join('\n')
  } else if (k) {
    statusEl.className = 'status'
    const n = k.strokes.length
    statusEl.textContent = all.length
      ? `${n} stroke${n === 1 ? '' : 's'} · ${k.crossings} crossing${k.crossings === 1 ? '' : 's'} · ${all.length} relation${all.length === 1 ? '' : 's'}`
      : 'empty board: drag across the dots to draw'
  }

  const bad = new Set(problems.map((p) => p.rel))
  listEl.innerHTML = rels
    .map((r, i) => {
      const pts = r.type === 'HORN' ? `[${r.vertices[0]}] → [${r.vertices[1]}]` : `${r.vertices.length} dots${r.closed ? ' ↺' : ''}`
      const cls = [i === selected ? 'selected' : '', bad.has(i) ? 'bad' : ''].join(' ')
      return `<li data-i="${i}" class="${cls}"><i style="background:${relColour(r)}"></i>${relName(r)} ${pts}<small>#${i}</small></li>`
    })
    .join('')

  hintEl.innerHTML =
    selected !== null
      ? `<kbd>#${selected}</kbd> selected · <kbd>c p h l r</kbd> change its brush · <kbd>⌫</kbd> delete · <kbd>esc</kbd> deselect`
      : pending
        ? `Drawing: move across dots, then <kbd>click</kbd> to stop. ${isHorn(pending.brush) ? 'Stop on a dot next to the base.' : 'Move onto the first dot to close a ring.'} <kbd>esc</kbd> cancels.`
        : isHorn(brush)
          ? `Horn: <kbd>click</kbd> the base dot, move to a neighbouring tip dot, <kbd>click</kbd> again.`
          : `<kbd>click</kbd> a dot to start, move across dots, <kbd>click</kbd> again to stop. Click a stroke (between dots) to select it.`
}

function render(fromState = false): void {
  const all = pending && pending.dots.length >= 2 ? [...rels, pendingRelation(pending)] : rels
  const h: Hypergraph = { rows, cols, relations: all }
  const problems = validate(h, false) // a free drawing: only what cannot be drawn blocks it
  let k: Compiled | null = null
  if (!problems.length) {
    try {
      k = compile(h)
      lastGood = { h, k }
    } catch (e) {
      problems.push({ rel: -1, level: 'error', message: (e as Error).message })
    }
  }
  const bad = new Set(problems.map((p) => p.rel))
  renderDraw(all, bad)
  renderKolam(k ?? lastGood?.k ?? null, k ? h : lastGood?.h ?? null, !k, selected)
  if (fromState) textError = null
  renderPanel(all, problems, k)
}

// ── drawing board input ────────────────────────────────────────────────────
function localXY(e: PointerEvent): XY {
  const r = drawCanvas.getBoundingClientRect()
  return [e.clientX - r.left, e.clientY - r.top]
}

// Click (or tap) a dot to start a stroke, move across dots to extend it, click again to stop.
// Touch: tap dot after dot (tap the last dot again to stop), or drag a finger and lift to stop.
let fingerDragged = false

function finish(at: XY | null): void {
  if (!pending) return
  if (pending.dots.length >= 2) {
    rels = [...rels, pendingRelation(pending)]
    selected = null // stay in drawing mode: the next brush key picks a brush, it doesn't edit this stroke
    textError = null
    commit()
  } else if (at) {
    selected = relationNear(...at) // started and stopped on the same dot: select what is there
  }
  pending = null
  fingerDragged = false
}

drawCanvas.addEventListener('pointerdown', (e) => {
  const [x, y] = localXY(e)
  const d = dotAt(x, y)
  cursor = [x, y]
  if (!pending) {
    if (d) pending = { brush, dots: [d], closed: false }
    else selected = relationNear(x, y)
    fingerDragged = false
  } else if (e.pointerType !== 'mouse' && d && !eq(d, pending.dots[pending.dots.length - 1]) && extend(pending, d)) {
    // touch has no hover: each tap on a new dot extends the stroke
  } else {
    finish([x, y])
  }
  render()
})
drawCanvas.addEventListener('pointermove', (e) => {
  const [x, y] = localXY(e)
  const d = dotAt(x, y)
  const touchDown = e.pointerType !== 'mouse' && e.buttons > 0
  const moved = pending && (e.pointerType === 'mouse' || touchDown) ? extend(pending, d) : false
  if (moved && touchDown) fingerDragged = true
  const hoverChanged = (d === null) !== (hover === null) || (d && hover && !eq(d, hover))
  hover = d
  cursor = [x, y]
  if (moved || hoverChanged || pending) render()
})
drawCanvas.addEventListener('pointerup', (e) => {
  if (e.pointerType !== 'mouse' && pending && fingerDragged) {
    finish(null) // a finger drag ends when the finger lifts
    render()
  }
})
drawCanvas.addEventListener('pointerleave', () => {
  hover = null
  cursor = null
  render()
})

// ── panel input ────────────────────────────────────────────────────────────
function setBrush(b: BrushKey): void {
  if (selected !== null) {
    // change the selected stroke's brush (2-dot loops and horns convert into each other)
    const r = rels[selected]
    let next: Relation | null = null
    if (isHorn(b) && (r.type === 'HORN' || (r.vertices.length === 2 && !r.closed)))
      next = { type: 'HORN', vertices: r.vertices.slice(0, 2), variant: VARIANT[b] }
    else if (!isHorn(b)) next = { type: b === 'c' ? 'CROSSING_LOOP' : 'PARALLEL_LOOP', vertices: r.vertices, closed: r.closed }
    if (next) {
      rels = rels.map((x, i) => (i === selected ? next! : x))
      commit()
    } else {
      statusEl.className = 'status error'
      statusEl.textContent = 'only a 2-dot loop can become a horn'
      return
    }
  }
  brush = b
  render()
}

document.querySelectorAll<HTMLButtonElement>('.brush').forEach((el) =>
  el.addEventListener('click', () => setBrush(el.dataset.brush as BrushKey)),
)

listEl.addEventListener('click', (e) => {
  const li = (e.target as HTMLElement).closest('li')
  if (!li) return
  const i = Number(li.dataset.i)
  selected = selected === i ? null : i
  render()
})

programEl.addEventListener('input', () => {
  try {
    const h = parse(programEl.value, false)
    ;({ rows, cols } = h)
    rels = h.relations
    selected = null
    textError = null
    programEl.classList.remove('invalid')
    commit()
  } catch (err) {
    if (!(err instanceof KolamError)) throw err
    textError = err.message
    programEl.classList.add('invalid')
  }
  render()
})
programEl.addEventListener('blur', () => {
  textError = null
  render()
})

function setGrid(): void {
  const r = Math.max(1, Math.min(16, Number(rowsEl.value) || rows))
  const c = Math.max(1, Math.min(16, Number(colsEl.value) || cols))
  if (r === rows && c === cols) return
  rows = r
  cols = c
  commit()
  render()
}
rowsEl.addEventListener('input', setGrid)
colsEl.addEventListener('input', setGrid)

$('undo').addEventListener('click', () => at > 0 && restore(history[--at]))
$('redo').addEventListener('click', () => at < history.length - 1 && restore(history[++at]))
$('clear').addEventListener('click', () => {
  rels = []
  selected = null
  commit()
  render()
})
$('copy').addEventListener('click', async () => {
  const btn = $<HTMLButtonElement>('copy')
  try {
    await navigator.clipboard.writeText(formatProgram(canonical({ rows, cols, relations: rels })))
    btn.textContent = 'copied!'
    btn.classList.add('done')
  } catch {
    btn.textContent = 'failed'
  }
  setTimeout(() => {
    btn.textContent = 'copy'
    btn.classList.remove('done')
  }, 1500)
})

window.addEventListener('keydown', (e) => {
  const t = e.target as HTMLElement
  if (t.closest('input, textarea')) return
  const mod = e.ctrlKey || e.metaKey
  if (mod && e.key.toLowerCase() === 'z') {
    e.preventDefault()
    if (e.shiftKey) $('redo').click()
    else $('undo').click()
    return
  }
  if (mod && e.key.toLowerCase() === 'y') {
    e.preventDefault()
    $('redo').click()
    return
  }
  if (mod || e.altKey) return
  const b = BRUSHES.find((x) => x.hotkey === e.key.toLowerCase())
  if (b) return setBrush(b.key)
  if ((e.key === 'Delete' || e.key === 'Backspace') && selected !== null) {
    rels = rels.filter((_, i) => i !== selected)
    selected = null
    commit()
    render()
  } else if (e.key === 'Escape') {
    pending = null
    selected = null
    render()
  }
})

// ── start ──────────────────────────────────────────────────────────────────
commit() // every page load starts from a clean 5x5 slate
new ResizeObserver(() => render()).observe(document.querySelector('.layout')!)
document.fonts.load(`700 40px 'Schoolbell'`).then(() => {
  drawBoard.invalidateBase()
  kolamBoard.invalidateBase()
  render()
})
