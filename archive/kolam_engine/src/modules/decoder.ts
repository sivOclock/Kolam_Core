// decoder.ts — live trace while drawing, looping animation on commit

import { type GridConfig } from './schema'
import { gridOrigin, latticeToCanvas, getSpacing } from './canvas'
import { chalkStroke } from './renderer'
import { theme } from '../styles/theme'

let _animId: number | null = null

// Touch/coarse-pointer devices (phones, tablets) auto-scroll the page when
// textarea.setSelectionRange() is called repeatedly, even without focus.
// Skip the text-highlight side effect there; the canvas animation still runs.
const _isTouchDevice = typeof window !== 'undefined' &&
  window.matchMedia?.('(pointer: coarse)').matches

function buildSplinePts(scope: any, coords: any[], closed = false): any[] {
  if (coords.length < 2) return []
  const path = new scope.Path()
  path.moveTo(coords[0])
  for (let i = 1; i < coords.length - 1; i++) {
    const mx = (coords[i].x + coords[i + 1].x) / 2
    const my = (coords[i].y + coords[i + 1].y) / 2
    path.quadraticCurveTo(coords[i], new scope.Point(mx, my))
  }
  if (closed) path.quadraticCurveTo(coords[coords.length - 1], coords[0])
  else path.lineTo(coords[coords.length - 1])
  const pts: any[] = []
  for (let d = 0; d <= path.length; d += 2) { const p = path.getPointAt(d); if (p) pts.push(p) }
  path.remove()
  return pts
}

function clearLayer(scope: any) {
  scope.activate()
  const l = scope.project.layers.find((x: any) => x.name === 'decode-anim')
  if (l) l.remove()
}

/** Scroll a textarea just enough to keep the given char offset's line visible,
 *  without touching focus (avoids mobile keyboard flicker / auto-scroll). */
function scrollTokenIntoView(seqEl: HTMLTextAreaElement, charOffset: number) {
  const lineHeight = parseFloat(getComputedStyle(seqEl).lineHeight) || 16
  const lineIdx = seqEl.value.slice(0, charOffset).split('\n').length - 1
  const lineTop = lineIdx * lineHeight
  const lineBottom = lineTop + lineHeight
  if (lineTop < seqEl.scrollTop) {
    seqEl.scrollTop = lineTop
  } else if (lineBottom > seqEl.scrollTop + seqEl.clientHeight) {
    seqEl.scrollTop = lineBottom - seqEl.clientHeight
  }
}

function toCoords(scope: any, stroke: [number,number][], cfg: GridConfig, origin: any) {
  return stroke.map(([li, lj]) => {
    const { x, y } = latticeToCanvas(li, lj, cfg, origin)
    return new scope.Point(x, y)
  })
}

/** Live trace while drawing */
export function decodeLive(
  scope: any, canvas: HTMLCanvasElement, cfg: GridConfig,
  seq: [number, number][][], seqEl: HTMLTextAreaElement,
  updateSeq = true,
) {
  if (_animId !== null) { cancelAnimationFrame(_animId); _animId = null }
  clearLayer(scope)
  if (seq.length === 0) { seqEl.value = ''; return }

  scope.activate()
  const layer = new scope.Layer({ name: 'decode-anim' })
  layer.activate()
  const spacing = getSpacing(cfg, canvas.clientWidth, canvas.clientHeight)
  const rcfg = { ...cfg, spacing }
  const origin = gridOrigin(rcfg, canvas.clientWidth, canvas.clientHeight)

  for (const stroke of seq) {
    if (stroke.length < 2) continue
    const pts = buildSplinePts(scope, toCoords(scope, stroke, rcfg, origin))
    if (pts.length >= 2) chalkStroke(scope, pts, theme.chalk.guide, 3)
  }

  // Update textarea — strokes separated by ---
  if (updateSeq) {
    const lines = seq.map(stroke => stroke.map(([li, lj]) => `[${li},${lj}]`).join(' → '))
    seqEl.value = lines.join('\n---\n')
    const lastLine = lines[lines.length - 1]
    const lastToken = lastLine.split(' → ').pop()!
    const idx = seqEl.value.lastIndexOf(lastToken)
    if (!_isTouchDevice) {
      seqEl.focus()
      seqEl.setSelectionRange(idx, idx + lastToken.length)
    }
  }
}

/** Looping animation after stroke committed */
export function decodeLoop(
  scope: any, canvas: HTMLCanvasElement, cfg: GridConfig,
  seq: [number, number][][], seqEl: HTMLTextAreaElement,
) {
  if (_animId !== null) { cancelAnimationFrame(_animId); _animId = null }
  if (seq.length === 0) return

  // Build textarea with --- separators and pre-compute char offsets per token
  const strokeLines = seq.map(s => s.map(([li, lj]) => `[${li},${lj}]`).join(' → '))
  const joined = strokeLines.join('\n---\n')
  seqEl.value = joined

  // Char offset for every token across all strokes
  const allOffsets: { start: number; end: number }[] = []
  let pos = 0
  for (let li = 0; li < strokeLines.length; li++) {
    const line = strokeLines[li]
    const tokens = line.split(' → ')
    for (let ti = 0; ti < tokens.length; ti++) {
      const t = tokens[ti]
      const idx = joined.indexOf(t, pos)
      allOffsets.push({ start: idx, end: idx + t.length })
      pos = idx + t.length
      // skip the ' → ' separator between tokens
      if (ti < tokens.length - 1) pos += 3 // ' → '.length
    }
    // skip the '\n---\n' separator between strokes
    if (li < strokeLines.length - 1) pos += 5 // '\n---\n'.length
  }

  const spacing = getSpacing(cfg, canvas.clientWidth, canvas.clientHeight)
  const rcfg = { ...cfg, spacing }
  const origin = gridOrigin(rcfg, canvas.clientWidth, canvas.clientHeight)
  const TRAIL = 60

  // Build per-stroke spline pts + token mappings
  type StrokeData = { pts: any[]; tokenPtIdx: number[] }
  const strokes: StrokeData[] = []
  let tokenOffset = 0
  for (const stroke of seq) {
    if (stroke.length < 2) { tokenOffset += stroke.length; continue }
    const coords = toCoords(scope, stroke, rcfg, origin)
    const first = stroke[0], last = stroke[stroke.length - 1]
    const closed = first[0] === last[0] && first[1] === last[1]
    const pts = buildSplinePts(scope, coords, closed)
    const absTokenPtIdx = stroke.map((_, i) => ({
      ptIdx: Math.floor((i / (stroke.length - 1)) * (pts.length - 1)),
    }))
    strokes.push({ pts, tokenPtIdx: absTokenPtIdx.map(x => x.ptIdx) })
    tokenOffset += stroke.length
  }

  // Flatten: animate stroke by stroke in a loop
  let si = 0   // current stroke index
  let head = 0 // pt index within current stroke

  // Accumulate all previous stroke pts as ghosts
  function frame() {
    clearLayer(scope)
    scope.activate()
    const layer = new scope.Layer({ name: 'decode-anim' })
    layer.activate()

    // Draw all strokes faintly
    for (const sd of strokes) {
      if (sd.pts.length >= 2) chalkStroke(scope, sd.pts, theme.chalk.guide, 1)
    }

    const sd = strokes[si]
    if (!sd) { _animId = requestAnimationFrame(frame); return }

    // Animated trail on current stroke
    const trail = sd.pts.slice(Math.max(0, head - TRAIL), head + 1)
    if (trail.length >= 2) chalkStroke(scope, trail, theme.chalk.guide, 3.5)

    // Head dot
    const dot = new scope.Path.Circle(sd.pts[head], 5)
    dot.fillColor = new scope.Color(theme.chalk.highlight)
    dot.opacity = 0.95

    // Highlight token — compute absolute token index
    const baseTokenIdx = seq.slice(0, si).reduce((s, st) => s + st.length, 0)
    const localTokIdx = sd.tokenPtIdx.findLastIndex((pi: number) => pi <= head)
    const absTokIdx = baseTokenIdx + (localTokIdx >= 0 ? localTokIdx : 0)
    if (allOffsets[absTokIdx] && !_isTouchDevice) {
      const { start, end } = allOffsets[absTokIdx]
      seqEl.setSelectionRange(start, end)
      scrollTokenIntoView(seqEl, start)
    }

    head++
    if (head >= sd.pts.length) {
      head = 0
      si = (si + 1) % strokes.length
    }
    _animId = requestAnimationFrame(frame)
  }

  _animId = requestAnimationFrame(frame)
}

export function stopDecode() {
  if (_animId !== null) { cancelAnimationFrame(_animId); _animId = null }
}
