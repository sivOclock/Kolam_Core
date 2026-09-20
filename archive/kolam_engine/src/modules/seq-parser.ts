// seq-parser.ts — parse textarea sequence text into [number,number][][] 

/**
 * Parses text like:
 *   [1,1] → [1,2] → [2,2]
 *   ---
 *   [3,1] → [3,2]
 *
 * Returns array of strokes, each stroke is array of [li, lj] pairs.
 * Invalid tokens are skipped. Strokes with < 2 points are dropped.
 */
export function parseSeq(text: string): [number, number][][] {
  return text
    .split(/\n?---\n?/)
    .map(block =>
      [...block.matchAll(/\[(\d+),(\d+)\]/g)]
        .map(m => [parseInt(m[1]), parseInt(m[2])] as [number, number])
    )
    .filter(stroke => stroke.length >= 2)
}
