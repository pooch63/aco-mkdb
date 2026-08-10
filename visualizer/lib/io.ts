import { edgeKey } from "./graph";
import type { GraphJson, Point } from "./types";

export function download(filename: string, content: string, mime: string) {
  const blob = new Blob([content], { type: mime });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

export function serializeJson(
  k: number,
  theta: number,
  U: Map<number, Point>,
  V: Map<number, Point>,
  edges: Map<string, number>,
): string {
  const data: GraphJson = {
    k,
    theta,
    U: [...U].map(([id, p]) => ({ id, x: p.x, y: p.y })),
    V: [...V].map(([id, p]) => ({ id, x: p.x, y: p.y })),
    edges: [...edges].map(([key, ts]) => {
      const [u, v] = key.split(",").map(Number);
      return { u, v, timestamp: ts };
    }),
  };
  return JSON.stringify(data, null, 2);
}

export function serializeCsv(edges: Map<string, number>): string {
  const lines = ["u,v"];
  for (const [key] of edges) {
    const [u, v] = key.split(",");
    lines.push(`${u},${v}`);
  }
  return lines.join("\n");
}

export type LoadedGraph = {
  U: Map<number, Point>;
  V: Map<number, Point>;
  edges: Map<string, number>;
  nextUId: number;
  nextVId: number;
  k?: number;
  theta?: number;
  needsRelayout: boolean;
};

export function parseJsonGraph(data: GraphJson): LoadedGraph {
  const U = new Map<number, Point>();
  const V = new Map<number, Point>();
  const edges = new Map<string, number>();
  let maxU = 0;
  let maxV = 0;
  for (const n of data.U || []) {
    U.set(n.id, { x: n.x ?? 0, y: n.y ?? 0 });
    maxU = Math.max(maxU, n.id);
  }
  for (const n of data.V || []) {
    V.set(n.id, { x: n.x ?? 0, y: n.y ?? 0 });
    maxV = Math.max(maxV, n.id);
  }
  for (const e of data.edges || []) {
    edges.set(edgeKey(e.u, e.v), e.timestamp || 0);
  }
  const needsRelayout =
    !data.U ||
    !data.U.length ||
    !data.U[0] ||
    data.U[0].x === undefined;
  return {
    U,
    V,
    edges,
    nextUId: maxU + 1,
    nextVId: maxV + 1,
    k: typeof data.k === "number" ? data.k : undefined,
    theta: typeof data.theta === "number" ? data.theta : undefined,
    needsRelayout,
  };
}

export function parseCsvGraph(text: string): LoadedGraph {
  const U = new Map<number, Point>();
  const V = new Map<number, Point>();
  const edges = new Map<string, number>();
  const lines = text.split(/\r?\n/).filter((l) => l.trim().length > 0);
  let start = 0;
  if (lines.length && isNaN(parseInt(lines[0].split(",")[0], 10))) start = 1;
  let maxU = 0;
  let maxV = 0;
  for (let i = start; i < lines.length; i++) {
    const parts = lines[i].split(",");
    const u = parseInt(parts[0], 10);
    const v = parseInt(parts[1], 10);
    const ts = parts.length > 2 ? parseInt(parts[2], 10) : 0;
    if (Number.isNaN(u) || Number.isNaN(v)) continue;
    if (!U.has(u)) U.set(u, { x: 0, y: 0 });
    if (!V.has(v)) V.set(v, { x: 0, y: 0 });
    edges.set(edgeKey(u, v), Number.isNaN(ts) ? 0 : ts);
    maxU = Math.max(maxU, u);
    maxV = Math.max(maxV, v);
  }
  return {
    U,
    V,
    edges,
    nextUId: maxU + 1,
    nextVId: maxV + 1,
    needsRelayout: true,
  };
}
