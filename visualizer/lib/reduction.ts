import { degreeOf, edgeKey } from "./graph";
import type { GraphBag, ReductionPreview } from "./types";

/**
 * Faithful port of common_neighbor_reduction! from reduction.jl.
 * Same ascending-degree pass, same non-reset accumulator maps.
 */
export function runReduction(
  g: GraphBag,
  k: number,
  theta: number,
): ReductionPreview {
  const U = new Set(g.U.keys());
  const V = new Set(g.V.keys());
  const edges = new Set(g.edges.keys());

  const nodeExists = (isU: boolean, id: number) =>
    isU ? U.has(id) : V.has(id);

  const getNeighbors = (isU: boolean, id: number): number[] => {
    const out: number[] = [];
    if (isU) {
      for (const v of V) if (edges.has(edgeKey(id, v))) out.push(v);
    } else {
      for (const u of U) if (edges.has(edgeKey(u, id))) out.push(u);
    }
    return out;
  };

  const getDegree = (isU: boolean, id: number) =>
    getNeighbors(isU, id).length;

  const remEdge = (u: number, v: number) => edges.delete(edgeKey(u, v));

  const remNode = (isU: boolean, id: number) => {
    if (isU) {
      if (!U.has(id)) return false;
      for (const v of getNeighbors(true, id)) edges.delete(edgeKey(id, v));
      U.delete(id);
    } else {
      if (!V.has(id)) return false;
      for (const u of getNeighbors(false, id)) edges.delete(edgeKey(u, id));
      V.delete(id);
    }
    return true;
  };

  // Ascending-degree order computed once up front (degrees not recomputed).
  const order: { isU: boolean; id: number; deg: number }[] = [];
  for (const u of U) order.push({ isU: true, id: u, deg: degreeOf(g, true, u) });
  for (const v of V)
    order.push({ isU: false, id: v, deg: degreeOf(g, false, v) });
  order.sort((a, b) => a.deg - b.deg);

  // Accumulators allocated once and never reset — matches reduction.jl.
  const commonU = new Map<number, number>();
  const commonV = new Map<number, number>();
  const bump = (isU: boolean, id: number) => {
    const m = isU ? commonU : commonV;
    m.set(id, (m.get(id) || 0) + 1);
  };
  const getCommon = (isU: boolean, id: number) =>
    (isU ? commonU : commonV).get(id) || 0;

  for (const node of order) {
    if (!nodeExists(node.isU, node.id)) continue;

    const neighbors = getNeighbors(node.isU, node.id);

    for (const v of neighbors) {
      const twoHop = getNeighbors(!node.isU, v);
      for (const w of twoHop) bump(node.isU, w);
    }

    for (const v of neighbors) {
      const twoHop = getNeighbors(!node.isU, v);
      let validConnections = 0;
      for (const w of twoHop) {
        if (getCommon(node.isU, w) >= theta - k) validConnections += 1;
      }
      if (validConnections < theta - k) {
        if (node.isU) remEdge(node.id, v);
        else remEdge(v, node.id);
      }
    }

    const degree = getDegree(node.isU, node.id);
    if (degree < theta - k) {
      remNode(node.isU, node.id);
    }

    for (const v of neighbors) {
      if (getDegree(!node.isU, v) < theta - k) {
        remNode(!node.isU, v);
      }
    }
  }

  const removedU = new Set([...g.U.keys()].filter((u) => !U.has(u)));
  const removedV = new Set([...g.V.keys()].filter((v) => !V.has(v)));
  const removedEdges = new Set(
    [...g.edges.keys()].filter((e) => !edges.has(e)),
  );

  return { U, V, edges, removedU, removedV, removedEdges };
}
