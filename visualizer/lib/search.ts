import { edgeKey } from "./graph";
import type { GraphBag, SearchResult } from "./types";

/**
 * Clean branch & bound for k/θ-MDBB:
 * maximize edges(S) s.t. |S_U|,|S_V| >= theta and missing(S) <= k.
 *
 * Not a line-by-line port of opponent.jl — gives a correct reference result.
 */
export function runSearch(
  g: GraphBag,
  k: number,
  theta: number,
): SearchResult {
  const uIds = [...g.U.keys()];
  const vIds = [...g.V.keys()];
  const nodes: { isU: boolean; id: number }[] = [
    ...uIds.map((id) => ({ isU: true, id })),
    ...vIds.map((id) => ({ isU: false, id })),
  ];

  const suffixU = new Array(nodes.length + 1).fill(0);
  const suffixV = new Array(nodes.length + 1).fill(0);
  for (let i = nodes.length - 1; i >= 0; i--) {
    suffixU[i] = suffixU[i + 1] + (nodes[i].isU ? 1 : 0);
    suffixV[i] = suffixV[i + 1] + (nodes[i].isU ? 0 : 1);
  }

  const hasEdge = (u: number, v: number) => g.edges.has(edgeKey(u, v));

  let best: { U: Set<number>; V: Set<number>; edges: number } = {
    U: new Set(),
    V: new Set(),
    edges: 0,
  };
  let bestMissing = 0;
  let nodesExplored = 0;
  const NODE_CAP = 4_000_000;

  function edgeCountBetween(SU: Set<number>, SV: Set<number>) {
    let c = 0;
    for (const u of SU) for (const v of SV) if (hasEdge(u, v)) c++;
    return c;
  }

  function branch(index: number, SU: Set<number>, SV: Set<number>) {
    nodesExplored++;
    if (nodesExplored > NODE_CAP) return;

    if (index === nodes.length) {
      if (SU.size >= theta && SV.size >= theta) {
        const e = edgeCountBetween(SU, SV);
        const missing = SU.size * SV.size - e;
        if (missing <= k && e > best.edges) {
          best = { U: new Set(SU), V: new Set(SV), edges: e };
          bestMissing = missing;
        }
      }
      return;
    }

    const potentialU = SU.size + suffixU[index];
    const potentialV = SV.size + suffixV[index];
    if (potentialU < theta || potentialV < theta) return;

    const upperBound = potentialU * potentialV;
    if (upperBound <= best.edges) return;

    const node = nodes[index];

    if (node.isU) SU.add(node.id);
    else SV.add(node.id);
    const missingSoFar = SU.size * SV.size - edgeCountBetween(SU, SV);
    if (missingSoFar <= k) branch(index + 1, SU, SV);
    if (node.isU) SU.delete(node.id);
    else SV.delete(node.id);

    branch(index + 1, SU, SV);
  }

  branch(0, new Set(), new Set());

  return {
    U: best.U,
    V: best.V,
    edges: best.edges,
    missing: bestMissing,
    k,
    theta,
    truncated: nodesExplored > NODE_CAP,
  };
}
