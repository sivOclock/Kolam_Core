// Kolam encoding -> hypergraph (port of kolam_press/encoding.py, spec §12).
// One digit per gap: 0 wall, 1 cross, 2 tube, 3 4 5 plain/left/right horn facing E or S,
// 6 7 8 the same facing W or N. Rows alternate cols digits (up/down gaps) and cols+1 (left/right).

import { type GateState, type Hypergraph, type Relation, HORN_VARIANTS, KolamError, fromGates } from './model'

/** Port of encoding.from_encoding: the rows of one kolam's encoding -> its canonical hypergraph. */
export function fromEncoding(enc: string[]): Hypergraph {
  if (enc.length < 3 || enc.length % 2 === 0)
    throw new KolamError(`an encoding has 2*rows+1 rows (3, 5, 7, ...), not ${enc.length}`)
  const rows = (enc.length - 1) / 2
  const cols = enc[0].length
  const gates = new Map<string, GateState>()
  const horns: Relation[] = []
  enc.forEach((row, i) => {
    if (row.length !== cols + (i % 2))
      throw new KolamError(`encoding row ${i} has ${row.length} digits, wants ${cols + (i % 2)}`)
    for (let k = 0; k < row.length; k++) {
      const x = Number(row[k])
      if (!x) continue
      const j = i % 2 ? 2 * k : 2 * k + 1 // lattice column of this gap
      if (x <= 2) {
        gates.set(`${i},${j}`, x === 1 ? 'CROSS' : 'JOIN')
        continue
      }
      // left/right gap: dots to its left (a) and right (b); up/down gap: dots above (a) and below (b)
      const [a, b]: [[number, number], [number, number]] = i % 2
        ? [[(i - 1) / 2, j / 2 - 1], [(i - 1) / 2, j / 2]]
        : [[i / 2 - 1, (j - 1) / 2], [i / 2, (j - 1) / 2]]
      const fwd = x <= 5
      horns.push({
        type: 'HORN',
        vertices: [fwd ? a : b],
        variant: HORN_VARIANTS[(x - 3) % 3],
        dir: i % 2 ? (fwd ? 'E' : 'W') : fwd ? 'S' : 'N',
      })
    }
  })
  return fromGates(rows, cols, gates, horns)
}

/** Every encoding in a pasted text. Kolams are separated by blank lines (or one per line as a JSON
 *  list or `|`-joined rows); anything after a row's digits (a comment) is ignored. */
export function parseEncodings(text: string): string[][] {
  const out: string[][] = []
  let cur: string[] = []
  const flush = () => {
    if (cur.length) out.push(cur)
    cur = []
  }
  for (const line of text.split('\n')) {
    const t = line.trim()
    if (t.startsWith('[') || t.startsWith('{')) {
      flush()
      try {
        const v = JSON.parse(t)
        const enc = Array.isArray(v) ? v : v.encoding
        if (Array.isArray(enc)) out.push(enc.map(String))
      } catch {
        /* not JSON: skip the line */
      }
      continue
    }
    if (t.includes('|')) {
      flush()
      out.push(t.split('|').map((r) => r.trim()).filter((r) => /^[0-8]+$/.test(r)))
      continue
    }
    const m = t.match(/^[0-8]+/)
    if (m) cur.push(m[0])
    else flush()
  }
  flush()
  return out
}
