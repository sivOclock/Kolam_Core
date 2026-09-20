// draw.ts — mousedown to start drag, mousemove snaps, mouseup commits spline

import { type LatticePoint, type GridConfig, isAnchor } from './schema'
import { buildLattice, gridOrigin, latticeToCanvas, getSpacing } from './canvas'
import { chalkStroke } from './renderer'
import { theme } from '../styles/theme'

let _cfg: GridConfig
let _scope: any
let _canvas: HTMLCanvasElement

let currentPts: LatticePoint[] = []
let finished: LatticePoint[][] = []
let drawing = false
let _draggedSinceDown = false
let _onLive: ((seq: [number, number][][]) => void) | null = null
let _onCommit: ((seq: [number, number][][]) => void) | null = null

// ── Public API ─────────────────────────────────────────────────────────────

export function initDraw(
  scope: any, canvas: HTMLCanvasElement, cfg: GridConfig,
  onLive?: (seq: [number, number][][]) => void,
  onCommit?: (seq: [number, number][][]) => void,
) {
  _scope = scope; _canvas = canvas; _cfg = cfg
  _onLive = onLive ?? null
  _onCommit = onCommit ?? null

  scope.activate()
  const tool = new scope.Tool()

  tool.onMouseMove = (e: any) => {
    const snap = nearest(e.point)
    if (drawing && snap) {
      const last = currentPts[currentPts.length - 1]
      if (!last || last.li !== snap.li || last.lj !== snap.lj) {
        currentPts = [...currentPts, snap]
        _draggedSinceDown = true
        _onLive?.([...finished, currentPts].map(pts => pts.map(p => [p.li, p.lj])))
        renderAll()
        return
      }
    }
    renderAll(snap ?? undefined)
  }

  tool.onMouseDown = (e: any) => {
    const snap = nearest(e.point)
    if (!snap) return
    const isTouch = typeof e.event?.type === 'string' && e.event.type.indexOf('touch') === 0

    if (!drawing) {
      drawing = true
      currentPts = [snap]
      _draggedSinceDown = false
    } else if (isTouch) {
      // Touch has no hover-drag between discrete taps, so each tap on a new
      // lattice point extends the stroke directly; tapping the same point
      // again (as on desktop) commits it. Mouse behavior below is untouched.
      const last = currentPts[currentPts.length - 1]
      if (last.li !== snap.li || last.lj !== snap.lj) {
        currentPts = [...currentPts, snap]
        _onLive?.([...finished, currentPts].map(pts => pts.map(p => [p.li, p.lj])))
        renderAll()
        return
      }
      drawing = false
      if (currentPts.length >= 2) finished = [...finished, currentPts]
      currentPts = []
      _onCommit?.(finished.map(pts => pts.map(p => [p.li, p.lj])))
    } else {
      drawing = false
      if (currentPts.length >= 2) finished = [...finished, currentPts]
      currentPts = []
      _onCommit?.(finished.map(pts => pts.map(p => [p.li, p.lj])))
    }
    renderAll()
  }

  // Touch-only: if the stroke was built by dragging a finger across the
  // lattice, lifting the finger finishes it immediately — no confirming tap
  // needed. Tap-by-tap strokes (no drag) still wait for the finishing tap,
  // so the two touch gestures don't fight each other. Desktop is untouched.
  tool.onMouseUp = (e: any) => {
    const isTouch = typeof e.event?.type === 'string' && e.event.type.indexOf('touch') === 0
    if (!isTouch || !drawing || !_draggedSinceDown) return
    drawing = false
    if (currentPts.length >= 2) finished = [...finished, currentPts]
    currentPts = []
    _draggedSinceDown = false
    _onCommit?.(finished.map(pts => pts.map(p => [p.li, p.lj])))
    renderAll()
  }
}

export function resetDraw(cfg: GridConfig) {
  _cfg = cfg
  currentPts = []; finished = []; drawing = false; _draggedSinceDown = false
  renderAll()
  _onLive?.([])
  _onCommit?.([])
}

export function undoDraw() {
  if (drawing) { drawing = false; currentPts = []; _draggedSinceDown = false }
  else finished = finished.slice(0, -1)
  renderAll()
  _onCommit?.(finished.map(pts => pts.map(p => [p.li, p.lj])))
}

/** Render externally-provided strokes onto the enc canvas (from seq textarea edit) */
export function renderSeqOnEnc(seq: [number, number][][]) {
  finished = seq.map(stroke => stroke.map(([li, lj]) => ({ li, lj })))
  currentPts = []; drawing = false
  renderAll()
}

// ── Handlers ───────────────────────────────────────────────────────────────

// ── Render ─────────────────────────────────────────────────────────────────

function renderAll(hover?: LatticePoint) {
  _scope.activate()
  const existing = _scope.project.layers.find((l: any) => l.name === 'draw')
  if (existing) existing.remove()
  const layer = new _scope.Layer({ name: 'draw' })
  layer.activate()

  const spacing = getSpacing(_cfg, _canvas.clientWidth, _canvas.clientHeight)
  const rcfg = { ..._cfg, spacing }
  const origin = gridOrigin(rcfg, _canvas.clientWidth, _canvas.clientHeight)

  for (const pts of finished) renderSpline(pts, origin, rcfg)
  if (currentPts.length >= 2) renderSpline(currentPts, origin, rcfg)

  if (hover && !drawing) {
    const hc = new _scope.Path.Circle(lp(hover, origin, rcfg), 8)
    hc.fillColor = new _scope.Color(0.4, 1, 0.4, 0.45)
  }
}

function renderSpline(pts: LatticePoint[], origin: { x: number; y: number }, rcfg: GridConfig) {
  const coords = pts.map(p => lp(p, origin, rcfg))
  const path = new _scope.Path()
  path.moveTo(coords[0])
  for (let i = 1; i < coords.length - 1; i++)
    path.quadraticCurveTo(coords[i], mid(coords[i], coords[i + 1]))
  path.lineTo(coords[coords.length - 1])

  const sampled: any[] = []
  for (let d = 0; d <= path.length; d += 2) {
    const p = path.getPointAt(d); if (p) sampled.push(p)
  }
  path.remove()
  if (sampled.length >= 2)
    chalkStroke(_scope, sampled, theme.chalk.main, 3)

  for (const p of pts) {
    const dot = new _scope.Path.Circle(lp(p, origin, rcfg), isAnchor(p.li, p.lj) ? 4 : 2.5)
    dot.fillColor = new _scope.Color(isAnchor(p.li, p.lj) ? theme.chalk.highlight : theme.chalk.guide)
    dot.opacity = 0.85
  }
}

// ── Helpers ────────────────────────────────────────────────────────────────

function lp(p: LatticePoint, origin: { x: number; y: number }, rcfg: GridConfig) {
  const { x, y } = latticeToCanvas(p.li, p.lj, rcfg, origin)
  return new _scope.Point(x, y)
}

function mid(a: any, b: any) {
  return new _scope.Point((a.x + b.x) / 2, (a.y + b.y) / 2)
}

function nearest(pt: any): LatticePoint | null {
  const spacing = getSpacing(_cfg, _canvas.clientWidth, _canvas.clientHeight)
  const rcfg = { ..._cfg, spacing }
  const origin = gridOrigin(rcfg, _canvas.clientWidth, _canvas.clientHeight)
  const cutoff = spacing / 2   // same as demo: LATTICE_DISTANCE / 2
  let best: LatticePoint | null = null, bestD = cutoff
  for (const p of buildLattice(_cfg)) {
    if (isAnchor(p.li, p.lj)) continue
    const { x, y } = latticeToCanvas(p.li, p.lj, rcfg, origin)
    const d = Math.hypot(pt.x - x, pt.y - y)
    if (d < bestD) { bestD = d; best = p }
  }
  return best
}
