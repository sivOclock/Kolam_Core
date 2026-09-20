// schema.ts — types and constants

export interface GridConfig {
  rows: number  // m — number of anchor rows
  cols: number  // n — number of anchor cols
  spacing: number  // pixels between lattice points
}

// A point on the (2m+1)×(2n+1) lattice grid
export interface LatticePoint {
  li: number  // lattice row index 0..2m
  lj: number  // lattice col index 0..2n
}

// An anchor dot — lattice points where li and lj are both odd
export interface AnchorDot {
  li: number  // odd lattice row: 1,3,5,...
  lj: number  // odd lattice col: 1,3,5,...
  row: number  // anchor row index 0..m-1  (li = 2*row+1)
  col: number  // anchor col index 0..n-1  (lj = 2*col+1)
}

export function isAnchor(li: number, lj: number): boolean {
  return li % 2 === 1 && lj % 2 === 1
}
