// spline.ts — kolam spline drawing

// A kolam path is a closed chain of quadratic Bézier splines.
// Each spline goes from anchor[i] to anchor[i+1], with the lattice
// point that sits between them as the control point.
//
// Because anchors are at odd lattice indices and the midpoint between
// two adjacent anchors is always an even-index lattice point, the
// control point is simply the average of the two anchor lattice coords.

import { type AnchorDot, type GridConfig } from './schema'
import { latticeToCanvas } from './canvas'
import { chalkStroke } from './renderer'
import { theme } from '../styles/theme'

export interface KolamLoop {
  anchors: AnchorDot[]   // ordered anchor sequence, closed (last→first)
}

// ── Spline geometry ────────────────────────────────────────────────────────

/** Canvas coords for an anchor dot */
function anchorPt(scope: any, a: AnchorDot, cfg: GridConfig, origin: { x: number; y: number }) {
  const { x, y } = latticeToCanvas(a.li, a.lj, cfg, origin)
  return new scope.Point(x, y)
}

/** Control point between two anchors = midpoint in lattice space → canvas */
function controlPt(scope: any, a: AnchorDot, b: AnchorDot, cfg: GridConfig, origin: { x: number; y: number }) {
  const { x, y } = latticeToCanvas((a.li + b.li) / 2, (a.lj + b.lj) / 2, cfg, origin)
  return new scope.Point(x, y)
}

// ── Drawing ────────────────────────────────────────────────────────────────

/**
 * Draw a closed kolam loop as chained quadratic Bézier splines.
 * Returns a group containing the chalk stroke, joint marks, and control-point guides.
 */
export function drawKolamLoop(
  scope: any,
  loop: KolamLoop,
  cfg: GridConfig,
  origin: { x: number; y: number },
  showGuides = true,
): any {
  const { anchors } = loop
  const n = anchors.length
  if (n < 2) return new scope.Group()

  // Build the closed Paper.js path using quadratic Bézier segments
  const path = new scope.Path()
  path.strokeColor = null
  path.fillColor = null

  // We split each quadratic segment into two: anchor → mid-to-next → next-anchor
  // Paper.js quadraticCurveTo(cp, end) adds the curve from current position
  const startPt = anchorPt(scope, anchors[0], cfg, origin)
  path.moveTo(startPt)

  for (let i = 0; i < n; i++) {
    const curr = anchors[i]
    const next = anchors[(i + 1) % n]
    const cp   = controlPt(scope, curr, next, cfg, origin)
    const ep   = anchorPt(scope, next, cfg, origin)
    path.quadraticCurveTo(cp, ep)
  }
  path.closePath()

  // Chalk stroke along the path
  const pathPts = samplePath(path, 2)
  const strokeGroup = chalkStroke(scope, pathPts, theme.chalk.main, 3)
  path.remove()

  const group = new scope.Group([strokeGroup])

  if (showGuides) {
    // Joint marks at every anchor (small filled circle)
    for (let i = 0; i < n; i++) {
      const pt = anchorPt(scope, anchors[i], cfg, origin)
      const joint = new scope.Path.Circle(pt, 4)
      joint.fillColor = new scope.Color(theme.chalk.highlight)
      joint.opacity = 0.85
      group.addChild(joint)
    }

    // Control-point guides at midpoints between consecutive anchors
    for (let i = 0; i < n; i++) {
      const curr = anchors[i]
      const next = anchors[(i + 1) % n]
      const cp   = controlPt(scope, curr, next, cfg, origin)

      // Faint guide dot
      const cpDot = new scope.Path.Circle(cp, 2.5)
      cpDot.fillColor = new scope.Color(theme.chalk.guide)
      cpDot.opacity = 0.5
      group.addChild(cpDot)

      // Faint line from anchor to control point (shows the pull direction)
      const ap = anchorPt(scope, curr, cfg, origin)
      const guideLine = new scope.Path.Line(ap, cp)
      guideLine.strokeColor = new scope.Color(theme.chalk.guide)
      guideLine.strokeWidth = 0.5
      guideLine.opacity = 0.3
      group.addChild(guideLine)
    }
  }

  return group
}

// ── Utility: sample Paper.js path into Point array for chalkStroke ─────────

function samplePath(path: any, step = 2): any[] {
  const len = path.length
  const pts: any[] = []
  for (let d = 0; d <= len; d += step) {
    const pt = path.getPointAt(d)
    if (pt) pts.push(pt)
  }
  return pts
}

// ── Default demo loop ──────────────────────────────────────────────────────

/**
 * Build a simple demo loop that visits anchors in a zigzag order
 * around the grid perimeter (works for any m×n grid ≥ 2×2).
 */
export function buildDemoLoop(anchors: AnchorDot[]): KolamLoop {
  // Walk perimeter: top row L→R, right col T→B, bottom row R→L, left col B→T
  const rows = Math.max(...anchors.map(a => a.row)) + 1
  const cols = Math.max(...anchors.map(a => a.col)) + 1
  const idx = (r: number, c: number) => anchors.find(a => a.row === r && a.col === c)!

  const seq: AnchorDot[] = []
  for (let c = 0; c < cols; c++) seq.push(idx(0, c))
  for (let r = 1; r < rows; r++) seq.push(idx(r, cols - 1))
  for (let c = cols - 2; c >= 0; c--) seq.push(idx(rows - 1, c))
  for (let r = rows - 2; r >= 1; r--) seq.push(idx(r, 0))

  return { anchors: seq }
}
