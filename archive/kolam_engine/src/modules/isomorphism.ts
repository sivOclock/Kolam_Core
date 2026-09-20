// isomorphism.ts — canonical graph certificate for kolam sequence comparison
// Two sequences are isomorphic iff their certificates match (position/direction/order invariant)

type Edge = [string, string]  // sorted node pair

function nodeKey(li: number, lj: number) { return `${li},${lj}` }

/** Build undirected edge set from a flat sequence of [li,lj] points */
function seqToEdges(seq: [number,number][]): Edge[] {
  const edges: Edge[] = []
  for (let i = 0; i < seq.length - 1; i++) {
    const a = nodeKey(...seq[i])
    const b = nodeKey(...seq[i+1])
    if (a === b) continue
    edges.push([a,b].sort() as Edge)
  }
  return edges
}

/** Build adjacency map from all strokes */
function buildGraph(strokes: [number,number][][]): Map<string, Set<string>> {
  const adj = new Map<string, Set<string>>()
  const add = (a: string, b: string) => {
    if (!adj.has(a)) adj.set(a, new Set())
    if (!adj.has(b)) adj.set(b, new Set())
    adj.get(a)!.add(b)
    adj.get(b)!.add(a)
  }
  for (const stroke of strokes)
    for (const [a, b] of seqToEdges(stroke)) add(a, b)
  return adj
}

/**
 * 3-round Weisfeiler-Lehman canonical certificate.
 * Returns a string that is invariant to node relabeling (position-invariant).
 */
export function certificate(strokes: [number,number][][]): string {
  const adj = buildGraph(strokes)
  const nodes = [...adj.keys()]

  // Round 0: label = degree
  let labels = new Map<string, string>(
    nodes.map(n => [n, String(adj.get(n)!.size)])
  )

  // 3 rounds of WL refinement
  for (let r = 0; r < 3; r++) {
    const next = new Map<string, string>()
    for (const n of nodes) {
      const neighborLabels = [...adj.get(n)!]
        .map(nb => labels.get(nb)!)
        .sort()
        .join(',')
      next.set(n, `${labels.get(n)}[${neighborLabels}]`)
    }
    labels = next
  }

  // Certificate = sorted list of all node labels (order-invariant)
  return [...labels.values()].sort().join('|')
}

/** Returns true if two kolam sequences are structurally isomorphic */
export function areIsomorphic(
  a: [number,number][][],
  b: [number,number][][],
): boolean {
  return certificate(a) === certificate(b)
}
