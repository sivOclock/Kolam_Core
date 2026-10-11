// A chalkboard canvas: lattice -> pixel layout, chalk-dust label, chalk brush strokes.
// The chalk look (seeded jittered stamps, dust specks, dusty label) follows archive/kolam_engine/renderer.ts.

import type { Point } from './kolam/model'

export const CHALK = {
  main: '#F9FAFB',
  highlight: '#FACC15',
  pink: '#FB7185',
  cyan: '#22D3EE',
  green: '#6EE7B7',
  red: '#F87171',
  dots: '#9CA3AF',
}

const LABEL_RESERVE = 44 // top band for the chalk-dust label

/** Seeded pseudo-random: the same geometry always gets the same chalk grain. */
export function seededRand(seed: number): () => number {
  let s = Math.abs(Math.floor(seed)) % 2147483646 || 1
  return () => {
    s = (s * 16807) % 2147483647
    return (s - 1) / 2147483646
  }
}

export type XY = [number, number]

export class Board {
  readonly canvas: HTMLCanvasElement
  readonly ctx: CanvasRenderingContext2D
  w = 0
  h = 0
  s = 20 // pixels per lattice unit (dots are 2 units apart)
  ox = 0
  oy = 0
  private base: HTMLCanvasElement | null = null // texture + label, rebuilt on resize
  private rows = 0
  private cols = 0
  private label: string

  constructor(canvas: HTMLCanvasElement, label: string) {
    this.canvas = canvas
    this.ctx = canvas.getContext('2d')!
    this.label = label
  }

  /** Fit the dot grid (rows x cols), plus `pad` lattice units around it, into the board. */
  layout(rows: number, cols: number, pad = 0): void {
    const dpr = window.devicePixelRatio || 1
    const w = this.canvas.clientWidth
    const h = this.canvas.clientHeight
    const resized = w !== this.w || h !== this.h
    if (resized) {
      this.w = w
      this.h = h
      this.canvas.width = Math.round(w * dpr)
      this.canvas.height = Math.round(h * dpr)
      this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
      this.base = null
    }
    this.rows = rows
    this.cols = cols
    const usableH = Math.max(h - LABEL_RESERVE, 1)
    this.s = Math.max(4, Math.min((w * 0.84) / (2 * cols + 2 * pad), (usableH * 0.84) / (2 * rows + 2 * pad)))
    this.ox = (w - 2 * cols * this.s) / 2
    this.oy = LABEL_RESERVE + (usableH - 2 * rows * this.s) / 2
  }

  /** Lattice point (li, lj) -> canvas pixels. */
  px(p: Point | XY): XY {
    return [this.ox + p[1] * this.s, this.oy + p[0] * this.s]
  }

  /** Canvas pixels -> fractional lattice coordinates (li, lj). */
  lattice(x: number, y: number): XY {
    return [(y - this.oy) / this.s, (x - this.ox) / this.s]
  }

  invalidateBase(): void {
    this.base = null
  }

  clear(): void {
    const { ctx } = this
    ctx.clearRect(0, 0, this.w, this.h)
    if (!this.base) this.base = this.buildBase()
    ctx.drawImage(this.base, 0, 0, this.w, this.h)
  }

  /** Grid dots; `only` ("r,c" keys) limits them to the dots in use. */
  dots(colour = CHALK.main, r = 4, only?: Set<string>): void {
    const { ctx } = this
    ctx.globalAlpha = 1
    ctx.fillStyle = colour
    for (let i = 0; i < this.rows; i++)
      for (let j = 0; j < this.cols; j++) {
        if (only && !only.has(`${i},${j}`)) continue
        const [x, y] = this.px([2 * i + 1, 2 * j + 1])
        ctx.beginPath()
        ctx.arc(x, y, r, 0, Math.PI * 2)
        ctx.fill()
      }
  }

  disc(p: Point | XY, r: number, colour: string, alpha = 1): void {
    const [x, y] = this.px(p)
    this.ctx.globalAlpha = alpha
    this.ctx.fillStyle = colour
    this.ctx.beginPath()
    this.ctx.arc(x, y, r, 0, Math.PI * 2)
    this.ctx.fill()
    this.ctx.globalAlpha = 1
  }

  /** Chalk brush along a list of pixel points (stamps jittered dots every ~1.5px). */
  chalk(pts: XY[], colour: string, width = 3, seed = 42, dashed = false): void {
    const { ctx } = this
    const rand = seededRand(seed)
    ctx.fillStyle = colour
    let travelled = 0
    for (let k = 0; k + 1 < pts.length; k++) {
      const [x0, y0] = pts[k]
      const [x1, y1] = pts[k + 1]
      const len = Math.hypot(x1 - x0, y1 - y0)
      const n = Math.max(1, Math.ceil(len / 1.5))
      for (let i = 0; i < n; i++) {
        travelled += len / n
        if (dashed && Math.floor(travelled / 7) % 2) continue
        const t = i / n
        const jx = (rand() - 0.5) * width * 0.45
        const jy = (rand() - 0.5) * width * 0.45
        const r = (width / 2) * (0.6 + rand() * 0.6)
        ctx.globalAlpha = 0.5 + rand() * 0.4
        ctx.beginPath()
        ctx.arc(x0 + (x1 - x0) * t + jx, y0 + (y1 - y0) * t + jy, r, 0, Math.PI * 2)
        ctx.fill()
      }
    }
    ctx.globalAlpha = 1
  }

  private buildBase(): HTMLCanvasElement {
    const dpr = window.devicePixelRatio || 1
    const c = document.createElement('canvas')
    c.width = Math.round(this.w * dpr)
    c.height = Math.round(this.h * dpr)
    const ctx = c.getContext('2d')!
    ctx.scale(dpr, dpr)
    // dust specks over the board
    const rand = seededRand(999)
    const count = Math.floor((this.w * this.h) / 6000)
    ctx.fillStyle = '#fff'
    for (let i = 0; i < count; i++) {
      ctx.globalAlpha = 0.02 + rand() * 0.06
      ctx.beginPath()
      ctx.arc(rand() * this.w, (0.2 + rand() * 0.6) * this.h, 0.5 + rand() * 1.5, 0, Math.PI * 2)
      ctx.fill()
    }
    // chalk-dust label: rasterise the text, then re-draw it as scattered chalk grains
    const off = document.createElement('canvas')
    off.width = Math.max(1, Math.round(this.w))
    off.height = 70
    const o = off.getContext('2d', { willReadFrequently: true })!
    o.fillStyle = '#000'
    o.fillRect(0, 0, off.width, 70)
    o.fillStyle = '#fff'
    o.font = `700 40px 'Schoolbell', cursive`
    o.letterSpacing = '6px'
    o.textAlign = 'center'
    o.textBaseline = 'middle'
    o.fillText(this.label.toUpperCase(), off.width / 2, 35)
    const { data } = o.getImageData(0, 0, off.width, 70)
    const lr = seededRand(this.label.charCodeAt(0) * 7 + 99)
    ctx.fillStyle = CHALK.main
    for (let y = 0; y < 70; y++)
      for (let x = 0; x < off.width; x++) {
        const b = data[(y * off.width + x) * 4]
        if (b < 80 || lr() > 0.75) continue
        ctx.globalAlpha = 0.45 + lr() * 0.45 * (b / 255)
        ctx.beginPath()
        ctx.arc(x + (lr() - 0.5) * 0.6, 4 + y + (lr() - 0.5) * 0.6, 0.5 + lr() * 0.6, 0, Math.PI * 2)
        ctx.fill()
      }
    ctx.globalAlpha = 1
    return c
  }
}

/** Sample a closed chained quadratic Bezier (kolam stroke) into pixel polylines, one per control point. */
export function bezierSegments(board: Board, P: (Point | XY)[]): XY[][] {
  const k = P.length
  const mid = (a: Point | XY, b: Point | XY): XY => [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2]
  const out: XY[][] = []
  for (let i = 0; i < k; i++) {
    const a = board.px(mid(P[(i - 1 + k) % k], P[i]))
    const c = board.px(P[i])
    const b = board.px(mid(P[i], P[(i + 1) % k]))
    const n = Math.max(2, Math.ceil((Math.hypot(c[0] - a[0], c[1] - a[1]) + Math.hypot(b[0] - c[0], b[1] - c[1])) / 3))
    const seg: XY[] = []
    for (let j = 0; j <= n; j++) {
      const t = j / n
      const u = 1 - t
      seg.push([u * u * a[0] + 2 * u * t * c[0] + t * t * b[0], u * u * a[1] + 2 * u * t * c[1] + t * t * b[1]])
    }
    out.push(seg)
  }
  return out
}
