// Brush model: relations, canonical labels, validation, brush DSL.
// A port of kolam_press/hypergraph.py + dsl.py; reads the shared kolam_press/brushes.json.
// Keep the two in step: the Python side is the source of truth.

import brushes from '../../../kolam_press/brushes.json'

export type Vertex = [number, number] // anchor dot [row, col]
export type Point = [number, number] // lattice point [li, lj]
export type BrushType = 'CROSSING_LOOP' | 'PARALLEL_LOOP' | 'HORN'
export type HornVariant = 'PLAIN_HORN' | 'LEFT_HORN' | 'RIGHT_HORN'
export type GateState = 'CROSS' | 'JOIN' | 'SPLIT'
export type HornDir = 'N' | 'E' | 'S' | 'W'

export interface Relation {
  type: BrushType
  vertices: Vertex[] // a horn has exactly one
  variant?: HornVariant // horns only
  closed?: boolean // loops only: last dot links back to the first
  dir?: HornDir // horns only: where the tip points
}

export interface Hypergraph {
  rows: number
  cols: number
  relations: Relation[]
}

export class KolamError extends Error {}

export const HORN_VARIANTS: HornVariant[] = ['PLAIN_HORN', 'LEFT_HORN', 'RIGHT_HORN']
export const HORN_DIRS: HornDir[] = ['N', 'E', 'S', 'W']
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

export const hornDir = (d: HornDir): [number, number] => {
  const f = spec.relations.HORN.dirs[d]
  return [f[0], f[1]]
}
export const hornShape = (long: boolean) => spec.relations.HORN.grow[long ? 'long' : 'short']
/** +1 LEFT_HORN, 0 PLAIN_HORN, -1 RIGHT_HORN */
export const hornCurl = (variant: HornVariant): number => spec.relations.HORN.variants[variant]

/** The spot a horn on dot v points at (may be off the grid). */
export function hornTarget(v: Vertex, d: HornDir): Vertex {
  const f = hornDir(d)
  return [v[0] + f[0], v[1] + f[1]]
}

/** A horn is a long crescent when the spot it points at holds no used dot, else short (spec §10.6). */
export function hornIsLong(h: Hypergraph, r: Relation): boolean {
  const t = hornTarget(r.vertices[0], r.dir!)
  return !h.relations.some((x) => x.vertices.some((v) => v[0] === t[0] && v[1] === t[1]))
}

/** Lattice point of a horn's tip: reach along fwd, turned by the curl. */
export function hornTip(v: Vertex, d: HornDir, variant: HornVariant, long: boolean): [number, number] {
  const o = toLattice(v)
  const f = hornDir(d)
  const l = leftOf(f)
  const sh = hornShape(long)
  const a = hornCurl(variant) * sh.curl
  const cu = sh.reach * Math.cos(a)
  const cv = sh.reach * Math.sin(a)
  return [o[0] + cu * f[0] + cv * l[0], o[1] + cu * f[1] + cv * l[1]]
}

export { leftOf }

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
    cmpVerts(a.vertices, b.vertices) ||
    ((a.dir ?? '') < (b.dir ?? '') ? -1 : (a.dir ?? '') > (b.dir ?? '') ? 1 : 0)
  )
}
const key = (r: Relation) => JSON.stringify([r.type, r.variant ?? null, !!r.closed, r.vertices, r.dir ?? null])

// ---------------------------------------------------------------- canonical labels
/** A loop's start dot and direction are invisible in an image, so they are normalised away. */
export function canonicalRelation(r: Relation): Relation {
  if (r.type === 'HORN')
    return { type: r.type, vertices: r.vertices.map((v) => [...v] as Vertex), variant: r.variant, dir: r.dir }
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

function tidy(rows: number, cols: number, rels: Relation[]): Hypergraph {
  const seen = new Map<string, Relation>()
  for (const r of rels) {
    const c = canonicalRelation(r)
    seen.set(key(c), c)
  }
  return { rows, cols, relations: [...seen.values()].sort(cmpRel) }
}

/** The label of this kolam's picture: loops re-read from the gaps they set (hypergraph.canonical).
 *  Different programs that draw the same picture get the same canonical label. */
export function canonical(h: Hypergraph): Hypergraph {
  if (problems(h).some((p) => p.level === 'error')) return tidy(h.rows, h.cols, h.relations) // cannot be drawn
  const gates = new Map<string, GateState>()
  for (const r of h.relations)
    if (r.type !== 'HORN')
      for (const [a, b] of edges(r)) {
        const g = gateBetween(a, b)
        if (!gates.has(`${g[0]},${g[1]}`)) gates.set(`${g[0]},${g[1]}`, loopGateState(r.type))
      }
  return fromGates(h.rows, h.cols, gates, h.relations.filter((r) => r.type === 'HORN'))
}

const STEPS: [number, number][] = [[-1, 0], [0, 1], [1, 0], [0, -1]] // N E S W

/** Canonical hypergraph from gap states ("li,lj" -> state) and horns: port of hypergraph.from_gates.
 *  A line goes on through a dot with 2 same-state gaps (corners too), straight through a dot with 3
 *  or 4 (the side branch is its own line), and stops before visiting a dot twice. */
export function fromGates(rows: number, cols: number, gates: Map<string, GateState>, horns: Relation[]): Hypergraph {
  const rels: Relation[] = []
  for (const [state, rtype] of [['CROSS', 'CROSSING_LOOP'], ['JOIN', 'PARALLEL_LOOP']] as const) {
    const nb = new Map<string, Set<number>>()
    const add = (v: Vertex, d: number) => nb.set(s(v), (nb.get(s(v)) ?? new Set()).add(d))
    for (const [k, st] of gates) {
      if (st !== state) continue
      const [li, lj] = k.split(',').map(Number)
      if (li % 2) {
        add([(li - 1) / 2, lj / 2 - 1], 1)
        add([(li - 1) / 2, lj / 2], 3)
      } else {
        add([li / 2 - 1, (lj - 1) / 2], 2)
        add([li / 2, (lj - 1) / 2], 0)
      }
    }
    const pair = (v: Vertex, d: number): number | null => {
      const ds = nb.get(s(v))!
      if (ds.size === 2) return [...ds].find((x) => x !== d)!
      return ds.size > 2 && ds.has((d + 2) % 4) ? (d + 2) % 4 : null
    }
    const used = new Set<string>()
    const edge = (v: Vertex, d: number): [Vertex, string] => {
      const w: Vertex = [v[0] + STEPS[d][0], v[1] + STEPS[d][1]]
      return [w, [s(v), s(w)].sort().join('')]
    }
    const trace = (v: Vertex, d0: number): [Vertex[], boolean] => {
      const chain: Vertex[] = [v]
      let cur = v
      let d = d0
      for (;;) {
        const [w, e] = edge(cur, d)
        if (chain.some((x) => x[0] === w[0] && x[1] === w[1])) {
          if (w[0] === v[0] && w[1] === v[1] && chain.length >= 4 && pair(v, (d + 2) % 4) === d0) {
            used.add(e)
            return [chain, true]
          }
          return [chain, false] // would visit a dot twice: stop here
        }
        used.add(e)
        chain.push(w)
        const nd = pair(w, (d + 2) % 4)
        if (nd === null || used.has(edge(w, nd)[1])) return [chain, false]
        cur = w
        d = nd
      }
    }
    const all: [Vertex, number][] = []
    for (const [k, ds] of nb) for (const d of ds) all.push([k.slice(1, -1).split(',').map(Number) as Vertex, d])
    const order = (a: [Vertex, number], b: [Vertex, number]) => cmpNum(a[0][0], b[0][0]) || cmpNum(a[0][1], b[0][1]) || cmpNum(a[1], b[1])
    const ends = all.filter(([v, d]) => pair(v, d) === null).sort(order)
    for (const [v, d] of [...ends, ...all.sort(order)])
      if (!used.has(edge(v, d)[1])) {
        const [chain, closed] = trace(v, d)
        rels.push({ type: rtype, vertices: chain, closed })
      }
  }
  return tidy(rows, cols, [...rels, ...horns])
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
  const horns: [number, Relation][] = []
  h.relations.forEach((r, i) => {
    const name = `${SHORT[r.type === 'HORN' ? r.variant ?? 'PLAIN_HORN' : r.type]}#${i}`
    const err = (m: string, level: Problem['level'] = 'error') => out.push({ rel: i, level, message: `${name}: ${m}` })
    const bad = r.vertices.filter((v) => !inside(v))
    if (bad.length) return err(`dots ${bad.map(s).join(' ')} are outside the ${h.rows}x${h.cols} grid`)
    if (new Set(r.vertices.map(s)).size !== r.vertices.length) return err('repeated dot')
    if (r.type === 'HORN') {
      if (!r.variant || !HORN_VARIANTS.includes(r.variant) || !r.dir || !HORN_DIRS.includes(r.dir) ||
          r.vertices.length !== 1 || r.closed)
        return err('a horn is one dot, a direction (N/E/S/W) and a variant')
      horns.push([i, r])
      return
    }
    if (r.vertices.length < 2) return err('a loop needs at least 2 dots')
    if (r.closed && (r.vertices.length < 4 || !adjacent(r.vertices[r.vertices.length - 1], r.vertices[0])))
      return err('a closed ring needs >= 4 dots with the last next to the first')
    const state = loopGateState(r.type)
    for (const [a, b] of edges(r)) {
      if (!adjacent(a, b)) {
        err(`${s(a)} and ${s(b)} are not neighbours (diagonal gaps have no gate)`)
        break
      }
      const g = JSON.stringify(gateBetween(a, b))
      const have = gateState.get(g)
      if (have && have[0] !== state) err(`the gap between ${s(a)} and ${s(b)} is already a ${have[0]} of #${have[1]}; one gap can't be both`)
      if (!have) gateState.set(g, [state, i])
    }
  })
  // horns: grown out of the wall in front of their dot, tips apart
  const seen = new Map<string, number>()
  const tips: [[number, number], number][] = []
  for (const [i, r] of horns) {
    const v = r.vertices[0]
    const name = `${SHORT[r.variant!]}#${i}`
    const err = (m: string) => out.push({ rel: i, level: 'error', message: `${name}: ${m}` })
    const k = `${s(v)}${r.dir}`
    if (seen.has(k)) {
      err(`dot ${s(v)} already has a horn pointing ${r.dir} (#${seen.get(k)})`)
      continue
    }
    seen.set(k, i)
    const g = gateState.get(JSON.stringify(gateBetween(v, hornTarget(v, r.dir!))))
    if (g) err(`points ${r.dir} along the brush line of #${g[1]}; a horn points at a wall (a gap no loop crosses)`)
    const p = hornTip(v, r.dir!, r.variant!, hornIsLong(h, r))
    for (const [q, j] of tips) if ((p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2 <= 1) err(`tip too close to the tip of horn #${j}`)
    tips.push([p, i])
  }
  return out
}

/** strict: every dataset rule (decodes to exactly this label); otherwise only what cannot be drawn. */
export function validate(h: Hypergraph, strict = true): Problem[] {
  const out = problems(h)
  if (strict && !out.length && JSON.stringify(tidy(h.rows, h.cols, h.relations)) !== JSON.stringify(canonical(h)))
    out.push({ rel: -1, level: 'label', message: 'not the canonical label of its picture (loops are read off the gaps)' })
  return out
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
    if (r.type === 'HORN') lines.push(`${SHORT[r.variant!]}(${s(r.vertices[0])},${r.dir})`)
    else lines.push(`${SHORT[r.type]}(${waypoints(r.vertices, !!r.closed).map(s).join(',')})`)
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
      if (HORN_VARIANTS.includes(kind as HornVariant)) {
        const m = /^\s*\[\s*(-?\d+)\s*,\s*(-?\d+)\s*\]\s*,\s*([NESW])\s*$/.exec(args)
        if (!m)
          throw new KolamError(
            `${where}: a horn is one dot and a direction, like ${fn}([1,1],N) (N/E/S/W); the old two-dot horn h(A,B) is now p(A,B) + h(B,dir)`,
          )
        rels.push({ type: 'HORN', vertices: [[Number(m[1]), Number(m[2])]], variant: kind as HornVariant, dir: m[3] as HornDir })
        continue
      }
      const rest = args.replace(/\[\s*-?\d+\s*,\s*-?\d+\s*\]/g, '').replace(/[\s,]/g, '')
      if (rest) throw new KolamError(`${where}: expected dots like [row, col]`)
      const pts = [...args.matchAll(/\[\s*(-?\d+)\s*,\s*(-?\d+)\s*\]/g)].map((x) => [Number(x[1]), Number(x[2])] as Vertex)
      if (pts.length < 2) throw new KolamError(`${where}: a loop brush needs at least two dots`)
      const closed = pts.length > 2 && pts[0][0] === pts[pts.length - 1][0] && pts[0][1] === pts[pts.length - 1][1]
      let dots = expand(pts, where)
      if (closed) dots = dots.slice(0, -1)
      rels.push({ type: kind as BrushType, vertices: dots, closed })
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
