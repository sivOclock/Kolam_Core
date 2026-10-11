// Gate compiler: hypergraph -> gate states -> traced closed strokes (port of kolam_press/compiler.py).
// The curve moves diagonally from gate to gate; each gate's state (brushes.json) decides the bounce.

import {
  type GateState,
  type Hypergraph,
  type HornVariant,
  type Point,
  type Vertex,
  KolamError,
  edges,
  gateBetween,
  gateFlip,
  hornFarTurn,
  hornTipPoints,
  loopGateState,
  toLattice,
} from './model'

interface HornTip {
  rel: number
  tipDot: Vertex
  variant: HornVariant
  turn: [Point, Point, Point]
  points: Point[]
}

export interface Compiled {
  strokes: Point[][] // closed control sequences (first point not repeated)
  owners: number[][] // relation index per control point (-1 = none)
  gates: Map<string, [GateState, number]> // non-SPLIT gates -> (state, relation)
  crossings: number
}

const pk = (p: Point) => `${p[0]},${p[1]}`
const DIRS: [number, number][] = [[1, 1], [1, -1], [-1, 1], [-1, -1]]

function dotGates(v: Vertex): Point[] {
  const [li, lj] = toLattice(v)
  return [[li - 1, lj], [li + 1, lj], [li, lj - 1], [li, lj + 1]]
}

export function compile(h: Hypergraph): Compiled {
  // ---- gate configuration: each brush sets the gaps between its consecutive dots
  const gates = new Map<string, [GateState, number]>()
  const tips: HornTip[] = []
  const used = new Set<string>()
  const dotRels = new Map<string, number[]>()
  h.relations.forEach((r, i) => {
    for (const v of r.vertices) {
      used.add(pk(v))
      dotRels.set(pk(v), [...(dotRels.get(pk(v)) ?? []), i])
    }
    if (r.type === 'HORN') {
      const [a, b] = r.vertices
      if (!gates.has(pk(gateBetween(a, b)))) gates.set(pk(gateBetween(a, b)), ['JOIN', i])
      tips.push({ rel: i, tipDot: b, variant: r.variant!, turn: hornFarTurn(a, b), points: hornTipPoints(a, b, r.variant!) })
      return
    }
    const state = loopGateState(r.type)
    for (const [a, b] of edges(r)) if (!gates.has(pk(gateBetween(a, b)))) gates.set(pk(gateBetween(a, b)), [state, i])
  })

  // ---- trace every diagonal link until each stroke closes
  const H = 2 * h.rows
  const W = 2 * h.cols
  const seen = new Set<string>()
  const raw: Point[][] = []
  for (let a = 0; a <= H; a++)
    for (let b = 0; b <= W; b++) {
      if ((a + b) % 2 !== 1) continue
      const g: Point = [a, b]
      for (const d of DIRS) {
        const q: Point = [a + d[0], b + d[1]]
        if (q[0] < 0 || q[0] > H || q[1] < 0 || q[1] > W || seen.has(`${pk(g)}>${pk(q)}`)) continue
        const seq: Point[] = [g]
        let cur = g
        let dd = d
        for (;;) {
          const nxt: Point = [cur[0] + dd[0], cur[1] + dd[1]]
          seen.add(`${pk(cur)}>${pk(nxt)}`)
          seen.add(`${pk(nxt)}>${pk(cur)}`)
          cur = nxt
          const f = gateFlip(gates.get(pk(cur))?.[0] ?? 'SPLIT', cur)
          dd = [dd[0] * f[0], dd[1] * f[1]]
          if (cur[0] === g[0] && cur[1] === g[1] && dd[0] === d[0] && dd[1] === d[1]) break
          seq.push(cur)
          if (seq.length > 4 * (H + 1) * (W + 1)) throw new KolamError('trace did not close')
        }
        raw.push(seq)
      }
    }
  // a dot no brush touches is just a small circle: leave it bare
  const bare = new Set<string>()
  for (let r = 0; r < h.rows; r++)
    for (let c = 0; c < h.cols; c++)
      if (!used.has(pk([r, c]))) bare.add(dotGates([r, c]).map(pk).sort().join('|'))
  const traced = raw.filter((s) => !(s.length === 4 && bare.has(s.map(pk).sort().join('|'))))

  // ---- horn tips replace the far-side turn around their tip dot
  const tipAt = new Map<string, HornTip>()
  for (const t of tips) tipAt.set(`${pk(t.turn[1])}|${[pk(t.turn[0]), pk(t.turn[2])].sort().join('|')}`, t)
  const owner = (g: Point, p: Point, q: Point): number => {
    const own = gates.get(pk(g))
    if (own) return own[1]
    const sr = p[0] + q[0]
    const sc = p[1] + q[1]
    if (sr % 2 || sc % 2) return -1
    const [li, lj] = [sr / 2, sc / 2]
    if (li % 2 === 0 || lj % 2 === 0) return -1
    const dot: Vertex = [(li - 1) / 2, (lj - 1) / 2]
    const rels = dotRels.get(pk(dot)) ?? []
    if (rels.length <= 1) return rels[0] ?? -1
    let best = rels[0]
    let bestD = 99
    for (const gg of dotGates(dot)) {
      const o = gates.get(pk(gg))
      if (o && rels.includes(o[1])) {
        const dist = Math.abs(gg[0] - g[0]) + Math.abs(gg[1] - g[1])
        if (dist < bestD) [best, bestD] = [o[1], dist]
      }
    }
    return best
  }
  const strokes: Point[][] = []
  const owners: number[][] = []
  let placed = 0
  for (const s of traced) {
    const pts: Point[] = []
    const own: number[] = []
    s.forEach((g, i) => {
      const p = s[(i - 1 + s.length) % s.length]
      const q = s[(i + 1) % s.length]
      const t = tipAt.get(`${pk(g)}|${[pk(p), pk(q)].sort().join('|')}`)
      if (t) {
        pts.push(...t.points)
        own.push(...t.points.map(() => t.rel))
        placed++
      } else {
        pts.push(g)
        own.push(owner(g, p, q))
      }
    })
    strokes.push(pts)
    owners.push(own)
  }
  if (placed !== tips.length) throw new KolamError('a horn tip was not placed')
  const crossings = [...gates.values()].filter(([st]) => st === 'CROSS').length
  return { strokes, owners, gates, crossings }
}
