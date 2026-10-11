// Brush model: relations, canonical labels, validation, brush DSL.
// A port of kolam_press/hypergraph.py + dsl.py; reads the shared kolam_press/brushes.json.
// Keep the two in step: the Python side is the source of truth.

import brushes from '../../../kolam_press/brushes.json'

export type Vertex = [number, number] // anchor dot [row, col]
export type Point = [number, number] // lattice point [li, lj]
export type BrushType = 'CROSSING_LOOP' | 'PARALLEL_LOOP' | 'HORN'
export type HornVariant = 'PLAIN_HORN' | 'LEFT_HORN' | 'RIGHT_HORN'
export type GateState = 'CROSS' | 'JOIN' | 'SPLIT'

export interface Relation {
  type: BrushType
  vertices: Vertex[]
  variant?: HornVariant // horns only
  closed?: boolean // loops only: last dot links back to the first
}

export interface Hypergraph {
  rows: number
  cols: number
  relations: Relation[]
}

export class KolamError extends Error {}

export const HORN_VARIANTS: HornVariant[] = ['PLAIN_HORN', 'LEFT_HORN', 'RIGHT_HORN']
const TYPE_ORDER: Record<BrushType, number> = { CROSSING_LOOP: 0, PARALLEL_LOOP: 1, HORN: 2 }
export const SHORT: Record<string, string> = {
  CROSSING_LOOP: 'c',
  PARALLEL_LOOP: 'p',
  PLAIN_HORN: 'h',
  LEFT_HORN: 'hl',
  RIGHT_HORN: 'hr',
}

// ---------------------------------------------------------------- spec helpers
type Spec = typeof brushes
const spec = brushes as Spec

export const toLattice = (v: Vertex): Point => [2 * v[0] + 1, 2 * v[1] + 1]
export const isHGate = (p: Point) => p[0] % 2 === 1 && p[1] % 2 === 0
export const adjacent = (a: Vertex, b: Vertex) => Math.abs(a[0] - b[0]) + Math.abs(a[1] - b[1]) === 1
/** Gate between two 4-adjacent dots. */
export const gateBetween = (a: Vertex, b: Vertex): Point => [a[0] + b[0] + 1, a[1] + b[1] + 1]
const leftOf = (d: [number, number]): [number, number] => [-d[1], d[0]]

export function loopGateState(t: BrushType): GateState {
  return spec.relations[t as 'CROSSING_LOOP' | 'PARALLEL_LOOP'].gate_state as GateState
}

export function gateFlip(state: GateState, g: Point): [number, number] {
  const f = spec.gate_states[state][isHGate(g) ? 'h_gate' : 'v_gate']
  return [f[0], f[1]]
}

export function hornFrame(a: Vertex, b: Vertex) {
  const fwd: [number, number] = [b[0] - a[0], b[1] - a[1]]
  return { o: toLattice(b), f: fwd, l: leftOf(fwd) }
}

export function hornTipPoints(a: Vertex, b: Vertex, variant: HornVariant): Point[] {
  const { o, f, l } = hornFrame(a, b)
  const tpl = spec.relations.HORN.tip_templates[variant] as number[][]
  return tpl.map(([u, v]) => [o[0] + u * f[0] + v * l[0], o[1] + u * f[1] + v * l[1]])
}

/** (left gate, far gate, right gate) of B: the turn a tip template replaces. */
export function hornFarTurn(a: Vertex, b: Vertex): [Point, Point, Point] {
  const { o, f, l } = hornFrame(a, b)
  return [
    [o[0] + l[0], o[1] + l[1]],
    [o[0] + f[0], o[1] + f[1]],
    [o[0] - l[0], o[1] - l[1]],
  ]
}

// ---------------------------------------------------------------- ordering (Python tuple order)
type Cmp = number
function cmpNum(a: number, b: number): Cmp {
  return a < b ? -1 : a > b ? 1 : 0
}
function cmpVerts(a: Vertex[], b: Vertex[]): Cmp {
  for (let i = 0; i < Math.min(a.length, b.length); i++) {
    const c = cmpNum(a[i][0], b[i][0]) || cmpNum(a[i][1], b[i][1])
    if (c) return c
  }
  return cmpNum(a.length, b.length)
}
function cmpRel(a: Relation, b: Relation): Cmp {
  const va = a.variant ?? ''
  const vb = b.variant ?? ''
  return (
    cmpNum(TYPE_ORDER[a.type], TYPE_ORDER[b.type]) ||
    (va < vb ? -1 : va > vb ? 1 : 0) ||
    cmpNum(Number(!!a.closed), Number(!!b.closed)) ||
    cmpVerts(a.vertices, b.vertices)
  )
}
const key = (r: Relation) => JSON.stringify([r.type, r.variant ?? null, !!r.closed, r.vertices])

// ---------------------------------------------------------------- canonical labels
/** A loop's start dot and direction are invisible in an image, so they are normalised away. */
export function canonicalRelation(r: Relation): Relation {
  if (r.type === 'HORN') return { type: r.type, vertices: r.vertices.map((v) => [...v] as Vertex), variant: r.variant }
  const vs = r.vertices
  let best: Vertex[]
  if (!r.closed) {
    const rev = [...vs].reverse()
    best = cmpVerts(vs, rev) <= 0 ? vs : rev
  } else {
    best = vs
    for (const seq of [vs, [...vs].reverse()])
      for (let i = 0; i < seq.length; i++) {
        const rot = [...seq.slice(i), ...seq.slice(0, i)]
        if (cmpVerts(rot, best) < 0) best = rot
      }
  }
  return { type: r.type, vertices: best.map((v) => [...v] as Vertex), closed: !!r.closed }
}

export function canonical(h: Hypergraph): Hypergraph {
  const seen = new Map<string, Relation>()
  for (const r of h.relations) {
    const c = canonicalRelation(r)
    seen.set(key(c), c)
  }
  return { rows: h.rows, cols: h.cols, relations: [...seen.values()].sort(cmpRel) }
}

export function edges(r: Relation): [Vertex, Vertex][] {
  const vs = r.vertices
  const out: [Vertex, Vertex][] = []
  for (let i = 0; i + 1 < vs.length; i++) out.push([vs[i], vs[i + 1]])
  if (r.closed && vs.length > 1) out.push([vs[vs.length - 1], vs[0]])
  return out
}

// ---------------------------------------------------------------- validation (hypergraph.py validate)
export interface Problem {
  rel: number // index into h.relations (-1 = whole kolam)
  level: 'error' | 'label' // error: cannot be drawn; label: draws, but is not a unique dataset label
  message: string
}

const s = (v: Vertex) => `[${v[0]},${v[1]}]`

/** Everything wrong with a hypergraph (port of hypergraph.problems). */
export function problems(h: Hypergraph): Problem[] {
  const out: Problem[] = []
  if (!(h.rows >= 1 && h.rows <= 32 && h.cols >= 1 && h.cols <= 32)) {
    return [{ rel: -1, level: 'error', message: `grid ${h.rows}x${h.cols} out of range 1..32` }]
  }
  const inside = (v: Vertex) => v[0] >= 0 && v[0] < h.rows && v[1] >= 0 && v[1] < h.cols
  const gateState = new Map<string, [GateState, number]>() // straight-step gap -> (state, first relation)
  const loopMembers: Record<string, Map<string, number>> = { CROSSING_LOOP: new Map(), PARALLEL_LOOP: new Map() }
  const hornTargets = new Map<string, number>()
  const tips: [Point, number][] = []
  h.relations.forEach((r, i) => {
    const name = `${SHORT[r.type === 'HORN' ? r.variant ?? 'PLAIN_HORN' : r.type]}#${i}`
    const err = (m: string, level: Problem['level'] = 'error') => out.push({ rel: i, level, message: `${name}: ${m}` })
    const bad = r.vertices.filter((v) => !inside(v))
    if (bad.length) return err(`dots ${bad.map(s).join(' ')} are outside the ${h.rows}x${h.cols} grid`)
    if (new Set(r.vertices.map(s)).size !== r.vertices.length) return err('repeated dot')
    if (r.type === 'HORN') {
      if (!r.variant || !HORN_VARIANTS.includes(r.variant) || r.vertices.length !== 2 || r.closed)
        return err('a horn needs exactly 2 dots and a variant')
      if (!adjacent(r.vertices[0], r.vertices[1])) return err(`horn dots ${s(r.vertices[0])} and ${s(r.vertices[1])} must be neighbours`)
    } else {
      if (r.vertices.length < 2) return err('a loop needs at least 2 dots')
      if (r.closed && (r.vertices.length < 4 || !adjacent(r.vertices[r.vertices.length - 1], r.vertices[0])))
        return err('a closed ring needs >= 4 dots with the last next to the first')
    }
    const state: GateState = r.type === 'HORN' ? 'JOIN' : loopGateState(r.type)
    for (const [a, b] of edges(r)) {
      if (!adjacent(a, b)) {
        err(`${s(a)} and ${s(b)} are not neighbours (diagonal gaps have no gate)`)
        break
      }
      const g = JSON.stringify(gateBetween(a, b))
      const have = gateState.get(g)
      if (have && have[0] !== state) err(`the gap between ${s(a)} and ${s(b)} is already a ${have[0]} of #${have[1]}; one gap can't be both`)
      else if (have) err(`the gap between ${s(a)} and ${s(b)} is also used by #${have[1]}`, 'label')
      if (!have) gateState.set(g, [state, i])
    }
    if (r.type !== 'HORN') {
      for (const v of r.vertices) {
        const m = loopMembers[r.type]
        if (m.has(s(v))) err(`dot ${s(v)} is already in #${m.get(s(v))} (same-brush loops sharing a dot)`, 'label')
        m.set(s(v), i)
      }
    } else {
      const [a, b] = r.vertices
      if (hornTargets.has(s(b))) err(`dot ${s(b)} is already the tip of horn #${hornTargets.get(s(b))}`)
      hornTargets.set(s(b), i)
      const p = hornTipPoints(a, b, r.variant!)[0]
      for (const [q, j] of tips)
        if ((p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2 <= 1) err(`tip too close to the tip of horn #${j}`)
      tips.push([p, i])
    }
  })
  for (const [b, i] of hornTargets) {
    const users = h.relations.flatMap((r, j) => (j !== i && r.vertices.some((v) => s(v) === b) ? [j] : []))
    if (users.length) out.push({ rel: i, level: 'error', message: `horn #${i}: tip dot ${b} must not be used by #${users.join(', #')}` })
  }
  return out
}

/** strict: every dataset rule (decodes to exactly this label); otherwise only what cannot be drawn. */
export function validate(h: Hypergraph, strict = true): Problem[] {
  return problems(h).filter((p) => strict || p.level === 'error')
}

// ---------------------------------------------------------------- brush DSL (dsl.py)
function waypoints(vs: Vertex[], closed: boolean): Vertex[] {
  const seq = closed ? [...vs, vs[0]] : vs
  const out = [seq[0]]
  for (let i = 1; i < seq.length - 1; i++) {
    const [a, b, c] = [seq[i - 1], seq[i], seq[i + 1]]
    if (b[0] - a[0] !== c[0] - b[0] || b[1] - a[1] !== c[1] - b[1]) out.push(b)
  }
  out.push(seq[seq.length - 1])
  return out
}

/** Canonical brush program for a hypergraph (inverse of parse). */
export function formatProgram(h: Hypergraph): string {
  const lines = [`grid(${h.rows}, ${h.cols})`]
  for (const r of canonical(h).relations) {
    const pts = r.type === 'HORN' ? r.vertices : waypoints(r.vertices, !!r.closed)
    const name = SHORT[r.type === 'HORN' ? r.variant! : r.type]
    lines.push(`${name}(${pts.map(s).join(',')})`)
  }
  return lines.join('\n')
}

function expand(wps: Vertex[], where: string): Vertex[] {
  const out: Vertex[] = [wps[0]]
  for (let i = 0; i + 1 < wps.length; i++) {
    const [a, b] = [wps[i], wps[i + 1]]
    if (a[0] === b[0] && a[1] === b[1]) throw new KolamError(`${where}: waypoint ${s(a)} repeated`)
    if (a[0] !== b[0] && a[1] !== b[1])
      throw new KolamError(`${where}: ${s(a)} -> ${s(b)} is diagonal; brushes run along rows/columns (add a corner waypoint)`)
    const dr = Math.sign(b[0] - a[0])
    const dc = Math.sign(b[1] - a[1])
    let cur = a
    while (cur[0] !== b[0] || cur[1] !== b[1]) {
      cur = [cur[0] + dr, cur[1] + dc]
      out.push(cur)
    }
  }
  return out
}

/** Parse a brush program into a canonical, validated hypergraph. */
export function parse(text: string, strict = true): Hypergraph {
  const names = spec.dsl as Record<string, string>
  let grid: [number, number] | null = null
  const rels: Relation[] = []
  text.split('\n').forEach((raw, n) => {
    for (let stmt of raw.split('#')[0].split(';')) {
      stmt = stmt.trim()
      if (!stmt) continue
      const where = `line ${n + 1} \`${stmt}\``
      const m = /^([A-Za-z_]\w*)\s*\((.*)\)$/.exec(stmt)
      if (!m) throw new KolamError(`${where}: expected a brush call like c([0,0],[0,3])`)
      const [, fn, args] = m
      if (fn === 'grid') {
        const g = /^\s*(\d+)\s*,\s*(\d+)\s*$/.exec(args)
        if (!g) throw new KolamError(`${where}: grid(rows, cols) takes two integers`)
        grid = [Number(g[1]), Number(g[2])]
        continue
      }
      if (!(fn in names)) throw new KolamError(`${where}: unknown brush '${fn}'; use c, p, h, hl, hr (or h0/h1/h2)`)
      const kind = names[fn]
      const rest = args.replace(/\[\s*-?\d+\s*,\s*-?\d+\s*\]/g, '').replace(/[\s,]/g, '')
      if (rest) throw new KolamError(`${where}: expected dots like [row, col]`)
      const pts = [...args.matchAll(/\[\s*(-?\d+)\s*,\s*(-?\d+)\s*\]/g)].map((x) => [Number(x[1]), Number(x[2])] as Vertex)
      if (HORN_VARIANTS.includes(kind as HornVariant)) {
        if (pts.length !== 2) throw new KolamError(`${where}: a horn takes exactly two dots: base -> tip`)
        if (!adjacent(pts[0], pts[1])) throw new KolamError(`${where}: horn dots ${s(pts[0])} and ${s(pts[1])} must be neighbours`)
        rels.push({ type: 'HORN', vertices: pts, variant: kind as HornVariant })
      } else {
        if (pts.length < 2) throw new KolamError(`${where}: a loop brush needs at least two dots`)
        const closed = pts.length > 2 && pts[0][0] === pts[pts.length - 1][0] && pts[0][1] === pts[pts.length - 1][1]
        let dots = expand(pts, where)
        if (closed) dots = dots.slice(0, -1)
        rels.push({ type: kind as BrushType, vertices: dots, closed })
      }
    }
  })
  if (!rels.length) throw new KolamError('empty program')
  if (!grid) {
    const vs = rels.flatMap((r) => r.vertices)
    grid = [Math.max(...vs.map((v) => v[0])) + 1, Math.max(...vs.map((v) => v[1])) + 1]
  }
  const h = canonical({ rows: grid[0], cols: grid[1], relations: rels })
  const errs = validate(h, strict)
  if (errs.length) throw new KolamError(errs.map((e) => e.message).join('; '))
  return h
}
