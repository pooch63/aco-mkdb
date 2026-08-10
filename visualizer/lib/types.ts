export type Point = { x: number; y: number };

export type EditMode = "move" | "addnode" | "addedge" | "delete";

export type GraphBag = {
  U: Map<number, Point>;
  V: Map<number, Point>;
  edges: Map<string, number>; // "u,v" -> timestamp
};

export type GraphSnapshot = GraphBag & {
  nextUId: number;
  nextVId: number;
};

export type NodeRef = { isU: boolean; id: number };

export type ReductionPreview = {
  U: Set<number>;
  V: Set<number>;
  edges: Set<string>;
  removedU: Set<number>;
  removedV: Set<number>;
  removedEdges: Set<string>;
};

export type SearchResult = {
  U: Set<number>;
  V: Set<number>;
  edges: number;
  missing: number;
  k: number;
  theta: number;
  truncated: boolean;
};

export type LogEntry = {
  id: number;
  time: string;
  msg: string;
  cls?: "ok" | "info" | "err";
};

export type GraphJson = {
  k?: number;
  theta?: number;
  U?: { id: number; x?: number; y?: number }[];
  V?: { id: number; x?: number; y?: number }[];
  edges?: { u: number; v: number; timestamp?: number }[];
};
