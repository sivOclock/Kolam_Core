// Gate compiler: hypergraph -> gate states -> traced closed strokes (port of kolam_press/compiler.py).
// The curve moves diagonally from gate to gate; each gate's state (brushes.json) decides the bounce.
// Horns set no gate: drawn.ts grows them out of the traced line.

import {
  type GateState,
  type HornDir,
  type HornVariant,
  type Hypergraph,
  type Point,
  type Vertex,
  KolamError,
  edges,
  gateBetween,
  gateFlip,
  hornIsLong,
  hornTip,
  loopGateState,
  toLattice,
} from './model'

export interface HornTip {
  rel: number
  dot: Vertex
  dir: HornDir
  variant: HornVariant
  long: boolean // crescent into an empty spot, else short (spec §10.6)
  point: [number, number] // lattice position of the sharp tip
}

export interface Compiled {
  rows: number
  cols: number
  strokes: Point[][] // closed control sequences (first point not repeated)
  owners: number[][] // relation index per control point (-1 = none)
  gates: Map<string, [GateState, number]> // non-SPLIT gates -> (state, relation)
  tips: HornTip[]
  used: Set<string> // dots some relation uses: the only dots drawn
  crossings: number
}

export const pk = (p: Point) => `${p[0]},${p[1]}`
const DIRS: [number, number][] = [[1, 1], [1, -1], [-1, 1], [-1, -1]]

function dotGates(v: Vertex): Point[] {
  const [li, lj] = toLattice(v)
  return [[li - 1, lj], [li + 1, lj], [li, lj - 1], [li, lj + 1]]
}

export function compile(h: Hypergraph): Compiled {
  // ---- gate configuration: each loop brush sets the gaps between its consecutive dots
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
      const v = r.vertices[0]
      const long = hornIsLong(h, r)
      tips.push({ rel: i, dot: v, dir: r.dir!, variant: r.variant!, long, point: hornTip(v, r.dir!, r.variant!, long) })
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
  const strokes = raw.filter((s) => !(s.length === 4 && bare.has(s.map(pk).sort().join('|'))))

  // ---- owner of each control point: the gate's relation, else the dot the strand turns around
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
  const owners = strokes.map((s) => s.map((g, i) => owner(g, s[(i - 1 + s.length) % s.length], s[(i + 1) % s.length])))
  const crossings = [...gates.values()].filter(([st]) => st === 'CROSS').length
  return { rows: h.rows, cols: h.cols, strokes, owners, gates, tips, used, crossings }
}
