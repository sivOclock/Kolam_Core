import './styles/challenge.css'
import paper from 'paper'
import { type GridConfig } from './modules/schema'
import { drawGrid, gridOrigin, latticeToCanvas, getSpacing } from './modules/canvas'
import { initDraw, resetDraw } from './modules/draw'
import { areIsomorphic } from './modules/isomorphism'
import { chalkStroke } from './modules/renderer'
import { theme } from './styles/theme'

const CHALLENGES: { label: string; seq: [number,number][][] }[] = [
  { label: 'figure-8', seq: [[[3,2],[4,1],[3,0],[2,1],[3,2],[4,3],[3,4],[2,3],[3,2]]] },
  {
    label: 'diagonal-8',
    seq: [
      [[3,4],[4,5],[3,6],[2,5],[3,4],[4,3],[3,2],[2,3],[3,4]],
      [[3,4],[4,5],[5,4],[4,3],[3,4],[2,5],[1,4],[2,3],[3,4]],
    ],
  },
  {
    label: 'star',
    seq: [
      [[2,2],[2,1],[1,0],[0,0],[0,1],[1,2],[2,2]],
      [[2,2],[1,2],[0,3],[0,4],[1,4],[2,3],[2,2]],
      [[2,2],[2,1],[3,0],[4,0],[4,1],[3,2],[2,2]],
      [[2,2],[3,2],[4,3],[4,4],[3,4],[2,3],[2,2]],
    ],
  },
]
let currentIdx = 0

document.querySelector<HTMLDivElement>('#app')!.innerHTML = `
  <div class="challenge-layout">
    <div class="top-row">
      <div class="panel">
        <div class="panel-label">Target Kolam</div>
        <canvas id="target-canvas"></canvas>
      </div>
      <div class="middle">
        <div class="status idle" id="status">Draw it</div>
        <div class="challenge-num" id="challenge-num">1 / ${CHALLENGES.length}</div>
        <button id="btn-next">Next →</button>
        <button id="btn-undo">Clear</button>
      </div>
      <div class="panel">
        <div class="panel-label">Your Drawing</div>
        <canvas id="draw-canvas"></canvas>
      </div>
    </div>
    <div class="bottom-row">
      <div class="graph-panel">
        <div class="panel-label">Target Graph Signature</div>
        <canvas id="graph-target"></canvas>
      </div>
      <div class="graph-panel">
        <div class="panel-label">Your Graph Signature</div>
        <canvas id="graph-user"></canvas>
      </div>
    </div>
  </div>
`

const targetCanvas = document.getElementById('target-canvas') as HTMLCanvasElement
const drawCanvas   = document.getElementById('draw-canvas')   as HTMLCanvasElement
const graphTarget  = document.getElementById('graph-target')  as HTMLCanvasElement
const graphUser    = document.getElementById('graph-user')    as HTMLCanvasElement
const statusEl     = document.getElementById('status')!
const targetScope  = new paper.PaperScope()
const drawScope    = new paper.PaperScope()
const cfg: GridConfig = { rows: 3, cols: 3, spacing: 40 }

// ── Canvas sizing ──────────────────────────────────────────────────────────

function sizeCanvas(canvas: HTMLCanvasElement) {
  // Only set CSS (layout) size. Paper.js owns canvas.width/height via
  // scope.view.viewSize — setting both independently double-applies DPR.
  canvas.style.width = ''
  canvas.style.height = ''
  const rect = canvas.getBoundingClientRect()
  const w = Math.round(rect.width) || 300
  const h = Math.round(rect.height) || 200
  canvas.style.width = w + 'px'; canvas.style.height = h + 'px'
}

function sizeGraphCanvas(canvas: HTMLCanvasElement) {
  const dpr = window.devicePixelRatio || 1
  canvas.style.width = ''
  canvas.style.height = ''
  const rect = canvas.getBoundingClientRect()
  const w = Math.round(rect.width) || 300
  const h = Math.round(rect.height) || 140
  canvas.width = w * dpr; canvas.height = h * dpr
  canvas.style.width = w + 'px'; canvas.style.height = h + 'px'
}

// ── Target kolam render ────────────────────────────────────────────────────

function renderTarget() {
  const { label, seq } = CHALLENGES[currentIdx]
  targetScope.activate()
  targetScope.project.clear()
  drawGrid(targetScope, targetCanvas, cfg, label)

  const spacing = getSpacing(cfg, targetCanvas.clientWidth, targetCanvas.clientHeight)
  const rcfg = { ...cfg, spacing }
  const origin = gridOrigin(rcfg, targetCanvas.clientWidth, targetCanvas.clientHeight)

  for (const stroke of seq) {
    if (stroke.length < 2) continue
    const coords = stroke.map(([li, lj]) => {
      const { x, y } = latticeToCanvas(li, lj, rcfg, origin)
      return new targetScope.Point(x, y)
    })
    const path = new targetScope.Path()
    path.moveTo(coords[0])
    for (let i = 1; i < coords.length - 1; i++) {
      const mx = (coords[i].x + coords[i+1].x) / 2
      const my = (coords[i].y + coords[i+1].y) / 2
      path.quadraticCurveTo(coords[i], new targetScope.Point(mx, my))
    }
    path.lineTo(coords[coords.length - 1])
    const pts: any[] = []
    for (let d = 0; d <= path.length; d += 2) { const p = path.getPointAt(d); if (p) pts.push(p) }
    path.remove()
    if (pts.length >= 2) chalkStroke(targetScope, pts, theme.chalk.highlight, 3)
  }
}

// ── Graph visualization ────────────────────────────────────────────────────

function buildAdj(strokes: [number,number][][]) {
  const adj = new Map<string, Set<string>>()
  const add = (a: string, b: string) => {
    if (!adj.has(a)) adj.set(a, new Set())
    if (!adj.has(b)) adj.set(b, new Set())
    adj.get(a)!.add(b); adj.get(b)!.add(a)
  }
  for (const s of strokes)
    for (let i = 0; i < s.length - 1; i++) {
      const a = s[i][0]+','+s[i][1], b = s[i+1][0]+','+s[i+1][1]
      if (a !== b) add(a, b)
    }
  return adj
}

function drawGraph(canvas: HTMLCanvasElement, adj: Map<string, Set<string>>, nodeColor: string) {
  const dpr = window.devicePixelRatio || 1
  const ctx = canvas.getContext('2d')!
  const W = canvas.width / dpr, H = canvas.height / dpr
  ctx.clearRect(0, 0, canvas.width, canvas.height)
  ctx.scale(dpr, dpr)

  const nodes = [...adj.keys()]
  if (nodes.length === 0) { ctx.setTransform(1,0,0,1,0,0); return }

  const maxDeg = Math.max(...nodes.map(n => adj.get(n)!.size))
  const sorted = [...nodes].sort((a,b) => adj.get(b)!.size - adj.get(a)!.size)

  // Layout: degree-hub at center, others in circle
  const cx = W / 2, cy = H / 2
  const r = Math.min(W, H) * 0.38
  const pos = new Map<string, {x:number, y:number}>()

  const hubs = sorted.filter(n => adj.get(n)!.size === maxDeg)
  const rest = sorted.filter(n => adj.get(n)!.size !== maxDeg)

  // Hubs in inner circle, rest in outer
  hubs.forEach((n, i) => {
    const angle = (i / Math.max(hubs.length,1)) * Math.PI * 2 - Math.PI/2
    const ir = hubs.length > 1 ? r * 0.3 : 0
    pos.set(n, { x: cx + ir * Math.cos(angle), y: cy + ir * Math.sin(angle) })
  })
  rest.forEach((n, i) => {
    const angle = (i / rest.length) * Math.PI * 2 - Math.PI/2
    pos.set(n, { x: cx + r * Math.cos(angle), y: cy + r * Math.sin(angle) })
  })

  // Edges
  const drawn = new Set<string>()
  ctx.lineWidth = 1.5
  for (const [n, nbrs] of adj) {
    for (const nb of nbrs) {
      const key = [n,nb].sort().join('|')
      if (drawn.has(key)) continue; drawn.add(key)
      const a = pos.get(n)!, b = pos.get(nb)!
      ctx.strokeStyle = nodeColor + '60'
      ctx.beginPath(); ctx.moveTo(a.x, a.y); ctx.lineTo(b.x, b.y); ctx.stroke()
    }
  }

  // Nodes
  for (const n of nodes) {
    const { x, y } = pos.get(n)!
    const deg = adj.get(n)!.size
    const nr = 6 + deg * 3
    const isHub = deg === maxDeg

    // Glow
    const grd = ctx.createRadialGradient(x, y, 0, x, y, nr * 2)
    grd.addColorStop(0, nodeColor + '40')
    grd.addColorStop(1, 'transparent')
    ctx.beginPath(); ctx.arc(x, y, nr * 2, 0, Math.PI*2)
    ctx.fillStyle = grd; ctx.fill()

    // Node circle
    ctx.beginPath(); ctx.arc(x, y, nr, 0, Math.PI*2)
    ctx.fillStyle = isHub ? '#FACC15' : nodeColor
    ctx.strokeStyle = 'rgba(255,255,255,0.3)'
    ctx.lineWidth = 1.5
    ctx.fill(); ctx.stroke()

    // Degree label
    ctx.fillStyle = isHub ? '#0B1F1B' : '#fff'
    ctx.font = `bold ${9 + deg * 1.5}px monospace`
    ctx.textAlign = 'center'; ctx.textBaseline = 'middle'
    ctx.fillText(String(deg), x, y)
  }

  ctx.setTransform(1,0,0,1,0,0)
}

// ── Validation ─────────────────────────────────────────────────────────────

function validate(userSeq: [number,number][][]) {
  drawGraph(graphTarget, buildAdj(CHALLENGES[currentIdx].seq), '#FACC15')
  drawGraph(graphUser,   buildAdj(userSeq), '#22D3EE')

  if (userSeq.length === 0 || userSeq.every(s => s.length < 2)) {
    statusEl.textContent = 'Draw it'
    statusEl.className = 'status idle'
    return
  }
  if (areIsomorphic(CHALLENGES[currentIdx].seq, userSeq)) {
    statusEl.textContent = '✓ Correct!'
    statusEl.className = 'status pass'
  } else {
    statusEl.textContent = '✗ Not yet'
    statusEl.className = 'status fail'
  }
}

// ── Init via ResizeObserver ────────────────────────────────────────────────

let initialized = false
const ro = new ResizeObserver(() => {
  sizeCanvas(targetCanvas); sizeCanvas(drawCanvas)
  sizeGraphCanvas(graphTarget); sizeGraphCanvas(graphUser)

  if (!initialized) {
    initialized = true
    targetScope.setup(targetCanvas)
    drawScope.setup(drawCanvas)
    drawGrid(drawScope, drawCanvas, cfg, 'draw')
    initDraw(drawScope, drawCanvas, cfg,
      (seq) => validate(seq),
      (seq) => validate(seq),
    )
  }

  targetScope.view.viewSize = new targetScope.Size(targetCanvas.clientWidth, targetCanvas.clientHeight)
  drawScope.view.viewSize   = new drawScope.Size(drawCanvas.clientWidth, drawCanvas.clientHeight)
  renderTarget()
  resetDraw(cfg)
  validate([])
})
ro.observe(document.querySelector('.challenge-layout')!)

document.getElementById('btn-undo')!.addEventListener('click', () => { resetDraw(cfg); validate([]) })
document.getElementById('btn-next')!.addEventListener('click', () => {
  currentIdx = (currentIdx + 1) % CHALLENGES.length
  document.getElementById('challenge-num')!.textContent = `${currentIdx + 1} / ${CHALLENGES.length}`
  resetDraw(cfg)
  renderTarget()
  validate([])
})
