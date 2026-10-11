// What is drawn for a compiled kolam: a port of kolam_press/geometry.py drawn() (spec §10).
// Parallel runs become capsules, the inner loop of a box is dropped, and horns grow out of the line.
// Drawing only: the compiled strokes (the label side) are not changed. Python is the source of truth.

import type { Compiled, HornTip } from './compiler'
import { type GateState, type Point, hornCurl, hornDir, hornShape, isHGate, leftOf, toLattice } from './model'

export type XY = [number, number]

const CAPSULE_W = 0.75 // parallel walls: distance from the line of dots (a crossing loop's distance)
const CAPSULE_LEAN = 0.5 // a wall point feeding straight into a crossing gap leans in to this
const HORN_SAMPLES = 10 // a stroke that carries a horn is redrawn through this many points per segment

const pk = (p: Point | XY) => `${p[0]},${p[1]}`

/** On-curve samples of a closed chained quadratic Bezier, n per control point. */
export function sampleStroke(P: XY[], n: number): XY[] {
  const k = P.length
  const out: XY[] = []
  for (let i = 0; i < k; i++) {
    const p0 = P[(i - 1 + k) % k]
    const p1 = P[i]
    const p2 = P[(i + 1) % k]
    const a: XY = [(p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2]
    const b: XY = [(p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2]
    for (let j = 0; j < n; j++) {
      const t = j / n
      const u = 1 - t
      out.push([u * u * a[0] + 2 * u * t * p1[0] + t * t * b[0], u * u * a[1] + 2 * u * t * p1[1] + t * t * b[1]])
    }
  }
  return out
}

function isBoxHole(gates: Map<string, [GateState, number]>, s: Point[]): boolean {
  return (
    s.length === 4 &&
    s.every((g) => gates.get(pk(g))?.[0] === 'JOIN') &&
    s.reduce((a, g) => a + g[0], 0) % 8 === 0 &&
    s.reduce((a, g) => a + g[1], 0) % 8 === 0
  )
}

/** Control points that draw parallel runs as capsules (geometry.capsule_points). */
function capsulePoints(gates: Map<string, [GateState, number]>, s: Point[]): { pts: XY[]; src: number[] } {
  const w = CAPSULE_W
  const k = s.length
  const joinDots = new Set<string>()
  for (const [key, [st]] of gates) {
    if (st !== 'JOIN') continue
    const g = key.split(',').map(Number) as Point
    if (isHGate(g)) [[g[0], g[1] - 1], [g[0], g[1] + 1]].forEach((d) => joinDots.add(pk(d as Point)))
    else [[g[0] - 1, g[1]], [g[0] + 1, g[1]]].forEach((d) => joinDots.add(pk(d as Point)))
  }
  const pts: XY[] = []
  const around: ([XY, XY] | null)[] = []
  for (let i = 0; i < k; i++) {
    const g = s[i]
    const p = s[(i - 1 + k) % k]
    const q = s[(i + 1) % k]
    const st = gates.get(pk(g))
    if (st && st[0] === 'JOIN') {
      const feedsX = [p, q].some((n) => gates.get(pk(n))?.[0] === 'CROSS')
      const off = feedsX ? Math.min(w, CAPSULE_LEAN) : w
      if (isHGate(g)) pts.push([g[0] + (p[0] > g[0] ? 1 : -1) * off, g[1]])
      else pts.push([g[0], g[1] + (p[1] > g[1] ? 1 : -1) * off])
      around.push(null)
      continue
    }
    const d: XY = [(p[0] + q[0]) / 2, (p[1] + q[1]) / 2] // the dot the line turns around here
    if (!st && d[0] % 2 === 1 && d[1] % 2 === 1 && joinDots.has(pk(d))) {
      pts.push([d[0] + (g[0] - d[0]) * w, d[1] + (g[1] - d[1]) * w])
      around.push([d, [g[0] - d[0], g[1] - d[1]]])
    } else {
      pts.push([g[0], g[1]])
      around.push(null)
    }
  }
  const out: XY[] = []
  const src: number[] = []
  for (let i = 0; i < k; i++) {
    out.push(pts[i])
    src.push(i)
    const a = around[i]
    const b = around[(i + 1) % k]
    if (a && b && a[0][0] === b[0][0] && a[0][1] === b[0][1]) {
      // turning on one dot's circle: add the 45 degree point
      const [d, u] = a
      const v = b[1]
      const n = Math.hypot(u[0] + v[0], u[1] + v[1])
      out.push([d[0] + (w * (u[0] + v[0])) / n, d[1] + (w * (u[1] + v[1])) / n])
      src.push(i)
    }
  }
  return { pts: out, src }
}

/** Control points actually drawn, and the relation owning each point. */
export function drawn(k: Compiled): { strokes: XY[][]; owners: number[][] } {
  let strokes: XY[][] = []
  let owners: number[][] = []
  k.strokes.forEach((s, i) => {
    if (isBoxHole(k.gates, s)) return
    const { pts, src } = capsulePoints(k.gates, s)
    strokes.push(pts)
    owners.push(src.map((j) => k.owners[i][j]))
  })
  if (k.tips.length) [strokes, owners] = growHorns(k.tips, strokes, owners)
  return { strokes, owners }
}

/** Grow every horn out of the line in front of its dot (geometry.grow_horns). */
function growHorns(tips: HornTip[], strokes: XY[][], owners: number[][]): [XY[][], number[][]] {
  const dense = new Map<number, XY[]>()
  const owns = new Map<number, number[]>()
  const dirs = new Set(tips.map((t) => `${pk(t.dot)}|${pk(hornDir(t.dir))}`))
  for (const t of tips) {
    const o = toLattice(t.dot)
    const f = hornDir(t.dir)
    const l = leftOf(f)
    const front: XY = [o[0] + 0.75 * f[0], o[1] + 0.75 * f[1]]
    const gap = (k: number) =>
      Math.min(...(dense.get(k) ?? sampleStroke(strokes[k], 4)).map((p) => Math.hypot(p[0] - front[0], p[1] - front[1])))
    let k = 0
    for (let j = 1; j < strokes.length; j++) if (gap(j) < gap(k)) k = j
    if (!dense.has(k)) {
      dense.set(k, sampleStroke(strokes[k], HORN_SAMPLES))
      owns.set(k, owners[k].flatMap((x) => Array<number>(HORN_SAMPLES).fill(x)))
    }
    const clip = { 1: dirs.has(`${pk(t.dot)}|${pk(l)}`), [-1]: dirs.has(`${pk(t.dot)}|${pk([-l[0], -l[1]])}`) }
    const [D, own] = grow(dense.get(k)!, owns.get(k)!, t, o, f, l, clip)
    dense.set(k, D)
    owns.set(k, own)
  }
  for (const [k, D] of dense) {
    strokes[k] = D
    owners[k] = owns.get(k)!
  }
  return [strokes, owners]
}

function grow(D: XY[], own: number[], t: HornTip, o: Point, f: [number, number], l: [number, number], clip: Record<number, boolean>): [XY[], number[]] {
  const sh = hornShape(t.long)
  const n = D.length
  const u = D.map((p) => (p[0] - o[0]) * f[0] + (p[1] - o[1]) * f[1])
  const v = D.map((p) => (p[0] - o[0]) * l[0] + (p[1] - o[1]) * l[1])
  let c = 0
  let best = Infinity
  for (let i = 0; i < n; i++) {
    const d = u[i] > 0 ? Math.hypot(u[i] - 0.75, v[i]) : Infinity // the line just in front
    if (d < best) [c, best] = [i, d]
  }
  const seg = D.map((p, i) => Math.hypot(D[(i + 1) % n][0] - p[0], D[(i + 1) % n][1] - p[1]))
  const total = seg.reduce((a, b) => a + b, 0)
  const pos: number[] = []
  seg.reduce((acc, x, i) => ((pos[i] = acc), acc + x), 0)
  const flip = v[(c + 1) % n] < v[c] ? -1 : 1 // so that a > 0 runs to the left
  const a = pos.map((x) => {
    const ahead = (((x - pos[c]) % total) + total) % total
    return (ahead <= total - ahead ? ahead : ahead - total) * flip
  })
  const half: Record<number, number> = {}
  for (const sd of [1, -1]) {
    // a horn beside this one on the same dot: stop at 45 degrees
    let lim = sh.half
    if (clip[sd]) for (let i = 0; i < n; i++) if (Math.sign(a[i]) === sd && sd * v[i] > u[i]) lim = Math.min(lim, Math.abs(a[i]))
    half[sd] = lim
  }
  const T = t.point
  const curl = hornCurl(t.variant)
  const out: XY[] = []
  const ownOut: number[] = []
  for (let i = 0; i < n; i++) {
    if (i === c) {
      out.push([T[0], T[1]], [T[0], T[1]]) // the tip, twice: a sharp corner
      ownOut.push(t.rel, t.rel)
      continue
    }
    const side = a[i] >= 0 ? 1 : -1
    const hs = half[side]
    const w = Math.abs(a[i]) < hs ? (1 - Math.min(Math.abs(a[i]) / hs, 1)) ** sh.q : 0
    const lean = w * (1 - w) * (sh.open * side - curl * sh.bend)
    out.push([D[i][0] + w * (T[0] - D[i][0]) + lean * l[0], D[i][1] + w * (T[1] - D[i][1]) + lean * l[1]])
    ownOut.push(w > 0.15 ? t.rel : own[i])
  }
  return [out, ownOut]
}
