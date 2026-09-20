// renderer.ts — chalk brush renderer

// Uses `any` for scope since paper.js exports classes not TS interfaces

function buildPath(scope: any, pts: any[]): any {
  const path = new scope.Path()
  if (pts.length < 2) return path
  const mid = (a: any, b: any) => a.add(b).divide(2)
  path.moveTo(pts[0])
  path.lineTo(mid(pts[0], pts[1]))
  for (let i = 1; i < pts.length - 1; i++)
    path.quadraticCurveTo(pts[i], mid(pts[i], pts[i + 1]))
  path.lineTo(pts[pts.length - 1])
  return path
}

// Seeded pseudo-random — same path always produces same grain
export function seededRand(seed: number): () => number {
  let s = seed
  return () => {
    s = (s * 16807 + 0) % 2147483647
    return (s - 1) / 2147483646
  }
}

// Point-sampled chalk brush: walks the path, stamps jittered circles
export function chalkStroke(
  scope: any,
  pts: any[],
  color: string,
  width = 3,
): any {
  const guide = buildPath(scope, pts)
  const len = guide.length
  const group = new scope.Group()
  // seed from first point coords so same geometry = same grain
  const rand = seededRand(Math.round(pts[0]?.x * 13 + pts[0]?.y * 7) || 42)

  for (let d = 0; d <= len; d += 1.5) {
    const pt = guide.getPointAt(d)
    if (!pt) continue
    const jx = (rand() - 0.5) * width * 0.45
    const jy = (rand() - 0.5) * width * 0.45
    const r  = (width / 2) * (0.6 + rand() * 0.6)
    const a  = 0.5 + rand() * 0.4
    const dot = new scope.Path.Circle(new scope.Point(pt.x + jx, pt.y + jy), r)
    const c = new scope.Color(color)
    c.alpha = a
    dot.fillColor = c
    group.addChild(dot)
  }

  guide.remove()
  return group
}

// Subtle chalk dust specks over the board texture
export function drawBoardTexture(scope: any, w: number, h: number): void {
  const rand = seededRand(999)
  const count = Math.floor((w * h) / 6000)
  for (let i = 0; i < count; i++) {
    const dot = new scope.Path.Circle(
      new scope.Point(rand() * w, (0.2 + rand() * 0.6) * h),
      0.5 + rand() * 1.5,
    )
    dot.fillColor = new scope.Color(1, 1, 1, 0.02 + rand() * 0.06)
  }
}
