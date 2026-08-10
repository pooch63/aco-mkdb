import type { GraphBag, GraphSnapshot, Point } from "./types";

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
