import type { GraphBag, GraphSnapshot, Point } from "./types";

export function nodeKey(isU: boolean, id: number): string {
  return `${isU ? "u" : "v"}:${id}`;
}

export function edgeKey(u: number, v: number): string {
  return u + "," + v;
}

export function parseEdgeKey(key: string): [number, number] {
  const [uStr, vStr] = key.split(",");
  return [parseInt(uStr, 10), parseInt(vStr, 10)];
}

export function neighborsOf(
  g: GraphBag,
  isU: boolean,
  id: number,
): number[] {
  const out: number[] = [];
  if (isU) {
    for (const v of g.V.keys()) if (g.edges.has(edgeKey(id, v))) out.push(v);
  } else {
    for (const u of g.U.keys()) if (g.edges.has(edgeKey(u, id))) out.push(u);
  }
  return out;
}

export function degreeOf(g: GraphBag, isU: boolean, id: number): number {
  return neighborsOf(g, isU, id).length;
}

/** Shortest-path distance from `start` out to `maxHop` (bipartite BFS). */
export function hopDistances(
  g: GraphBag,
  isU: boolean,
  id: number,
  maxHop: number,
): Map<string, number> {
  const dist = new Map<string, number>();
  dist.set(nodeKey(isU, id), 0);
  let frontier: { isU: boolean; id: number }[] = [{ isU, id }];
  const limit = Math.max(0, maxHop);
  for (let d = 0; d < limit; d++) {
    const next: { isU: boolean; id: number }[] = [];
    for (const node of frontier) {
      for (const nid of neighborsOf(g, node.isU, node.id)) {
        const key = nodeKey(!node.isU, nid);
        if (dist.has(key)) continue;
        dist.set(key, d + 1);
        next.push({ isU: !node.isU, id: nid });
      }
    }
    frontier = next;
    if (frontier.length === 0) break;
  }
  return dist;
}

export const HOP_COLORS = [
  "#5b52e0",
  "#1a9bb8",
  "#2db85a",
  "#d4a017",
  "#e07030",
  "#e0455a",
  "#7c3aed",
  "#0f766e",
  "#2563eb",
  "#be185d",
];

export function hopColor(distance: number): string {
  return HOP_COLORS[Math.min(Math.max(distance, 0), HOP_COLORS.length - 1)];
}

export function clonePointMap(map: Map<number, Point>): Map<number, Point> {
  return new Map([...map].map(([k, p]) => [k, { x: p.x, y: p.y }]));
}

export function snapshotOf(
  nextUId: number,
  nextVId: number,
  U: Map<number, Point>,
  V: Map<number, Point>,
  edges: Map<string, number>,
): GraphSnapshot {
  return {
    nextUId,
    nextVId,
    U: clonePointMap(U),
    V: clonePointMap(V),
    edges: new Map(edges),
  };
}

export function emptyGraph(): {
  nextUId: number;
  nextVId: number;
  U: Map<number, Point>;
  V: Map<number, Point>;
  edges: Map<string, number>;
} {
  return {
    nextUId: 1,
    nextVId: 1,
    U: new Map(),
    V: new Map(),
    edges: new Map(),
  };
}

/** Demo graph from the original visualizer.html seed. */
export function createDemoGraph(): {
  nextUId: number;
  nextVId: number;
  U: Map<number, Point>;
  V: Map<number, Point>;
  edges: Map<string, number>;
} {
  const U = new Map<number, Point>();
  const V = new Map<number, Point>();
  const edges = new Map<string, number>();
  const edgesSeed: [number, number][] = [
    [1, 1],
    [1, 2],
    [1, 3],
    [2, 1],
    [2, 2],
    [2, 4],
    [3, 2],
    [3, 3],
    [3, 4],
    [4, 1],
    [4, 4],
    [5, 3],
    [5, 4],
  ];
  const ts = Date.now();
  for (const [u, v] of edgesSeed) {
    if (!U.has(u)) U.set(u, { x: 0, y: 0 });
    if (!V.has(v)) V.set(v, { x: 0, y: 0 });
    edges.set(edgeKey(u, v), ts);
  }
  return { nextUId: 6, nextVId: 5, U, V, edges };
}
