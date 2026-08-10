import { edgeKey } from "./graph";
import type { GraphBag, NodeRef } from "./types";

export type PartialS = {
  U: Set<number>;
  V: Set<number>;
};

export function emptyPartialS(): PartialS {
  return { U: new Set(), V: new Set() };
}

export function clonePartialS(s: PartialS): PartialS {
  return { U: new Set(s.U), V: new Set(s.V) };
}

export function partialSHas(s: PartialS, isU: boolean, id: number): boolean {
  return isU ? s.U.has(id) : s.V.has(id);
}

export function partialSSize(s: PartialS): number {
  return s.U.size + s.V.size;
}

/** Non-edges in the bipartite induced subgraph G(S). */
export function missingInS(g: GraphBag, s: PartialS): number {
  if (s.U.size === 0 || s.V.size === 0) return 0;
  let missing = 0;
  for (const u of s.U) {
    for (const v of s.V) {
      if (!g.edges.has(edgeKey(u, v))) missing++;
    }
  }
  return missing;
}

/**
 * Extra missing edges incurred by adding `node` to S
 * (non-neighbors of `node` on the opposite side of S).
 */
export function addCostToS(g: GraphBag, s: PartialS, node: NodeRef): number {
  if (node.isU) {
    let cost = 0;
    for (const v of s.V) {
      if (!g.edges.has(edgeKey(node.id, v))) cost++;
    }
    return cost;
  }
  let cost = 0;
  for (const u of s.U) {
    if (!g.edges.has(edgeKey(u, node.id))) cost++;
  }
  return cost;
}

/**
 * Candidate set C for instance (S, ·): nodes v ∉ S such that
 * G(S ∪ {v}) remains a k-defective biclique (paper §3.1).
 */
export function candidatesOfS(
  g: GraphBag,
  s: PartialS,
  k: number,
): { U: Set<number>; V: Set<number> } {
  const missing = missingInS(g, s);
  const budget = k - missing;
  const CU = new Set<number>();
  const CV = new Set<number>();

  for (const id of g.U.keys()) {
    if (s.U.has(id)) continue;
    if (addCostToS(g, s, { isU: true, id }) <= budget) CU.add(id);
  }
  for (const id of g.V.keys()) {
    if (s.V.has(id)) continue;
    if (addCostToS(g, s, { isU: false, id }) <= budget) CV.add(id);
  }
  return { U: CU, V: CV };
}
