"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { createDemoGraph, edgeKey, snapshotOf } from "@/lib/graph";
import {
  addCostToS,
  candidatesOfS,
  clonePartialS,
  emptyPartialS,
  missingInS,
  partialSHas,
  partialSSize,
} from "@/lib/explain";
import {
  download,
  parseCsvGraph,
  parseJsonGraph,
  serializeCsv,
  serializeJson,
} from "@/lib/io";
import { COL_MARGIN, canvasHeight, relayoutAll, TOP_MARGIN } from "@/lib/layout";
import { runReduction } from "@/lib/reduction";
import { runSearch } from "@/lib/search";
import type {
  EditMode,
  ExplainHighlight,
  GraphJson,
  LogEntry,
  NodeRef,
  Point,
  ReductionPreview,
  SearchResult,
} from "@/lib/types";

const MODE_HINTS: Record<EditMode, string> = {
  move: "Drag a node to move it.",
  addnode: "Click inside the U or V column to add a node there.",
  addedge:
    "Click a U node then a V node (or vice-versa) to toggle an edge between them.",
  delete: "Click a node to delete it, or click an edge to delete just that edge.",
  explain:
    "Hover + 1/2/Space: neighbors vs non-neighbors. Hold Shift and click to toggle nodes in S. Use Fade non-candidates to dim nodes outside S∪C.",
};

const HISTORY_CAP = 25;

export function useGraphLab() {
  const [boot] = useState(() => createDemoGraph());
  const [nextUId, setNextUId] = useState(boot.nextUId);
  const [nextVId, setNextVId] = useState(boot.nextVId);
  const [U, setU] = useState(boot.U);
  const [V, setV] = useState(boot.V);
  const [edges, setEdges] = useState(boot.edges);

  const [mode, setModeState] = useState<EditMode>("move");
  const [selected, setSelected] = useState<NodeRef | null>(null);
  const [explainHover, setExplainHover] = useState<NodeRef | null>(null);
  const [explainHighlight, setExplainHighlight] =
    useState<ExplainHighlight | null>(null);
  const [explainS, setExplainS] = useState(emptyPartialS);
  const [fadeNonCandidates, setFadeNonCandidates] = useState(false);
  const [reducedPreview, setReducedPreview] = useState<ReductionPreview | null>(
    null,
  );
  const [solution, setSolution] = useState<SearchResult | null>(null);
  const [history, setHistory] = useState<
    ReturnType<typeof snapshotOf>[]
  >([]);

  const [k, setK] = useState(1);
  const [theta, setTheta] = useState(2);
  const [rnU, setRnU] = useState(8);
  const [rnV, setRnV] = useState(8);
  const [density, setDensity] = useState(0.4);
  const [searchWarn, setSearchWarn] = useState<string | null>(null);
  const [logs, setLogs] = useState<LogEntry[]>([]);
  const [canvasWidth, setCanvasWidth] = useState(800);
  const logId = useRef(0);
  const scrollRef = useRef<HTMLDivElement | null>(null);
  const didInitLayout = useRef(false);
  const graphRef = useRef({ U, V });

  useEffect(() => {
    graphRef.current = { U, V };
  }, [U, V]);

  const log = useCallback((msg: string, cls?: LogEntry["cls"]) => {
    const id = ++logId.current;
    const time = new Date().toLocaleTimeString();
    setLogs((prev) => [{ id, time, msg, cls }, ...prev].slice(0, 200));
  }, []);

  const pushHistory = useCallback(() => {
    setHistory((prev) => {
      const next = [
        ...prev,
        snapshotOf(nextUId, nextVId, U, V, edges),
      ];
      if (next.length > HISTORY_CAP) next.shift();
      return next;
    });
  }, [nextUId, nextVId, U, V, edges]);

  const applyRelayout = useCallback(
    (
      nextU: Map<number, Point>,
      nextV: Map<number, Point>,
      width?: number,
    ) => {
      const w =
        width ??
        Math.max(scrollRef.current?.clientWidth ?? canvasWidth, 500);
      relayoutAll(nextU, nextV, w);
      setCanvasWidth(w);
      setU(new Map(nextU));
      setV(new Map(nextV));
    },
    [canvasWidth],
  );

  // Relayout demo seed once the scroll container is measured.
  useEffect(() => {
    if (didInitLayout.current) return;
    const el = scrollRef.current;
    if (!el) return;
    didInitLayout.current = true;
    const w = Math.max(el.clientWidth, 500);
    const u = new Map(U);
    const v = new Map(V);
    relayoutAll(u, v, w);
    setCanvasWidth(w);
    setU(u);
    setV(v);
    log(
      "ready — demo graph loaded. Generate a random graph or edit this one to get started.",
      "info",
    );
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [log]);

  useEffect(() => {
    const onResize = () => {
      const w = Math.max(scrollRef.current?.clientWidth ?? 500, 500);
      const u = new Map(graphRef.current.U);
      const v = new Map(graphRef.current.V);
      relayoutAll(u, v, w);
      setCanvasWidth(w);
      setU(u);
      setV(v);
    };
    window.addEventListener("resize", onResize);
    return () => window.removeEventListener("resize", onResize);
  }, []);

  const clearOverlays = useCallback(() => {
    setReducedPreview(null);
    setSolution(null);
    setSearchWarn(null);
    setExplainS(emptyPartialS());
    setFadeNonCandidates(false);
  }, []);

  const clearExplainS = useCallback(() => {
    setExplainS(emptyPartialS());
    setFadeNonCandidates(false);
    log("cleared explain S", "info");
  }, [log]);

  const setMode = (m: EditMode) => {
    setModeState(m);
    setSelected(null);
    setExplainHover(null);
    setExplainHighlight(null);
    if (m !== "explain") {
      setExplainS(emptyPartialS());
      setFadeNonCandidates(false);
    }
  };

  const onExplainHover = useCallback(
    (node: NodeRef | null) => {
      if (mode !== "explain") return;
      setExplainHover(node);
      if (!node) setExplainHighlight(null);
    },
    [mode],
  );

  const toggleExplainS = useCallback(
    (isU: boolean, id: number) => {
      if (mode !== "explain") return;
      if (partialSHas(explainS, isU, id)) {
        setExplainS((prev) => {
          const next = clonePartialS(prev);
          if (isU) next.U.delete(id);
          else next.V.delete(id);
          return next;
        });
        log(`removed ${isU ? "U" : "V"}${id} from S`, "info");
        return;
      }
      const cost = addCostToS({ U, V, edges }, explainS, { isU, id });
      const missing = missingInS({ U, V, edges }, explainS);
      if (missing + cost > k) {
        log(
          `cannot add ${isU ? "U" : "V"}${id} to S — would make missing ${missing + cost} > k=${k}`,
          "err",
        );
        return;
      }
      setExplainS((prev) => {
        const next = clonePartialS(prev);
        if (isU) next.U.add(id);
        else next.V.add(id);
        return next;
      });
      log(
        `added ${isU ? "U" : "V"}${id} to S (cost ${cost}, missing→${missing + cost}/${k})`,
        "ok",
      );
    },
    [mode, explainS, U, V, edges, k, log],
  );

  // Explain mode: 1 / 2 / Space while hovering a node.
  useEffect(() => {
    if (mode !== "explain") return;
    const onKey = (ev: KeyboardEvent) => {
      const tag = (ev.target as HTMLElement | null)?.tagName;
      if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT") return;
      if (!explainHover) return;
      // Shift is reserved for building S via click.
      if (ev.shiftKey) return;

      if (ev.key === "1" || ev.key === "n" || ev.key === "N") {
        ev.preventDefault();
        setExplainHighlight("neighbors");
      } else if (ev.key === "2" || ev.key === "x" || ev.key === "X") {
        ev.preventDefault();
        setExplainHighlight("nonneighbors");
      } else if (ev.key === " " || ev.key === "Spacebar") {
        ev.preventDefault();
        setExplainHighlight((prev) =>
          prev === "neighbors" ? "nonneighbors" : "neighbors",
        );
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [mode, explainHover]);

  const clearGraph = () => {
    if (U.size === 0 && V.size === 0) return;
    pushHistory();
    setU(new Map());
    setV(new Map());
    setEdges(new Map());
    setNextUId(1);
    setNextVId(1);
    clearOverlays();
    log("cleared graph", "err");
  };

  const undo = () => {
    setHistory((prev) => {
      if (prev.length === 0) return prev;
      const snap = prev[prev.length - 1];
      setNextUId(snap.nextUId);
      setNextVId(snap.nextVId);
      setU(snap.U);
      setV(snap.V);
      setEdges(snap.edges);
      clearOverlays();
      log("undo", "info");
      return prev.slice(0, -1);
    });
  };

  const generateRandom = () => {
    const nU = Math.max(1, Math.min(60, rnU || 1));
    const nV = Math.max(1, Math.min(60, rnV || 1));
    pushHistory();
    clearOverlays();
    const newU = new Map<number, Point>();
    const newV = new Map<number, Point>();
    const newEdges = new Map<string, number>();
    for (let i = 1; i <= nU; i++) newU.set(i, { x: 0, y: 0 });
    for (let i = 1; i <= nV; i++) newV.set(i, { x: 0, y: 0 });
    let count = 0;
    const ts = Date.now();
    for (let u = 1; u <= nU; u++) {
      for (let v = 1; v <= nV; v++) {
        if (Math.random() < density) {
          newEdges.set(edgeKey(u, v), ts);
          count++;
        }
      }
    }
    setNextUId(nU + 1);
    setNextVId(nV + 1);
    setEdges(newEdges);
    applyRelayout(newU, newV);
    log(
      `generated random graph: |U|=${nU} |V|=${nV} |E|=${count} (density≈${density})`,
      "ok",
    );
  };

  const previewReduction = () => {
    setSolution(null);
    const result = runReduction({ U, V, edges }, k, theta);
    setReducedPreview(result);
    log(
      `reduction preview (k=${k}, θ=${theta}): removed ${result.removedU.size} U-nodes, ${result.removedV.size} V-nodes, ${result.removedEdges.size} edges`,
      "ok",
    );
  };

  const applyReduction = () => {
    if (!reducedPreview) return;
    pushHistory();
    const prev = reducedPreview;
    const newU = new Map(U);
    const newV = new Map(V);
    const newEdges = new Map(edges);
    for (const u of prev.removedU) newU.delete(u);
    for (const v of prev.removedV) newV.delete(v);
    for (const e of prev.removedEdges) newEdges.delete(e);
    setU(newU);
    setV(newV);
    setEdges(newEdges);
    setReducedPreview(null);
    log("reduction applied — graph committed", "ok");
  };

  const runMdbbSearch = () => {
    const total = U.size + V.size;
    if (total > 30) {
      setSearchWarn(
        `Graph has ${total} nodes — exhaustive branch & bound may be slow. Consider running on the reduced graph, or a smaller graph.`,
      );
    } else {
      setSearchWarn(null);
    }
    log(
      `running k/θ-MDBB search (k=${k}, θ=${theta}) on ${U.size}×${V.size} graph…`,
      "info",
    );
    const t0 = performance.now();
    const result = runSearch({ U, V, edges }, k, theta);
    const dt = (performance.now() - t0).toFixed(1);
    setSolution(result);
    if (result.U.size === 0 && result.V.size === 0) {
      log(`no valid S found meeting θ=${theta} within k=${k} (${dt}ms)`, "err");
    } else {
      log(
        `solution: |S_U|=${result.U.size} |S_V|=${result.V.size} edges=${result.edges} missing=${result.missing}/${k} (${dt}ms${result.truncated ? ", search truncated by node cap" : ""})`,
        "ok",
      );
    }
  };

  const clearSolution = () => setSolution(null);

  const saveJson = () => {
    download(
      "graph.json",
      serializeJson(k, theta, U, V, edges),
      "application/json",
    );
    log("saved graph.json", "ok");
  };

  const saveCsv = () => {
    download("edges.csv", serializeCsv(edges), "text/csv");
    log("saved edges.csv (u,v format)", "ok");
  };

  const loadFile = async (file: File) => {
    try {
      const text = await file.text();
      pushHistory();
      clearOverlays();
      let loaded;
      if (file.name.toLowerCase().endsWith(".json")) {
        loaded = parseJsonGraph(JSON.parse(text) as GraphJson);
        if (loaded.k !== undefined) setK(loaded.k);
        if (loaded.theta !== undefined) setTheta(loaded.theta);
      } else {
        loaded = parseCsvGraph(text);
      }
      setNextUId(loaded.nextUId);
      setNextVId(loaded.nextVId);
      setEdges(loaded.edges);
      if (loaded.needsRelayout) {
        applyRelayout(loaded.U, loaded.V);
      } else {
        setU(loaded.U);
        setV(loaded.V);
      }
      log(`loaded ${file.name}`, "ok");
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e);
      log(`failed to load ${file.name}: ${msg}`, "err");
    }
  };

  const addNodeAt = (x: number, y: number, width: number) => {
    if (mode !== "addnode") return;
    pushHistory();
    clearOverlays();
    const isU = x < width / 2;
    if (isU) {
      const id = nextUId;
      setNextUId(id + 1);
      setU((prev) => {
        const next = new Map(prev);
        next.set(id, { x: COL_MARGIN, y: Math.max(TOP_MARGIN, y) });
        return next;
      });
      log(`added U${id}`, "info");
    } else {
      const id = nextVId;
      setNextVId(id + 1);
      setV((prev) => {
        const next = new Map(prev);
        next.set(id, { x: width - COL_MARGIN, y: Math.max(TOP_MARGIN, y) });
        return next;
      });
      log(`added V${id}`, "info");
    }
  };

  const moveNode = (isU: boolean, id: number, y: number) => {
    const setter = isU ? setU : setV;
    setter((prev) => {
      const next = new Map(prev);
      const p = next.get(id);
      if (!p) return prev;
      next.set(id, { ...p, y: Math.max(TOP_MARGIN - 10, y) });
      return next;
    });
  };

  const beginDrag = () => {
    if (mode !== "move") return;
    pushHistory();
  };

  const onNodeClick = (isU: boolean, id: number) => {
    if (mode === "addedge") {
      if (!selected) {
        setSelected({ isU, id });
        return;
      }
      if (selected.isU === isU) {
        setSelected({ isU, id });
        return;
      }
      const u = isU ? id : selected.id;
      const v = isU ? selected.id : id;
      pushHistory();
      clearOverlays();
      const key = edgeKey(u, v);
      setEdges((prev) => {
        const next = new Map(prev);
        if (next.has(key)) {
          next.delete(key);
          log(`removed edge U${u}-V${v}`, "info");
        } else {
          next.set(key, Date.now());
          log(`added edge U${u}-V${v}`, "info");
        }
        return next;
      });
      setSelected(null);
    } else if (mode === "delete") {
      pushHistory();
      clearOverlays();
      if (isU) {
        setEdges((prev) => {
          const next = new Map(prev);
          for (const vv of V.keys()) next.delete(edgeKey(id, vv));
          return next;
        });
        setU((prev) => {
          const next = new Map(prev);
          next.delete(id);
          return next;
        });
        log(`deleted U${id}`, "err");
      } else {
        setEdges((prev) => {
          const next = new Map(prev);
          for (const uu of U.keys()) next.delete(edgeKey(uu, id));
          return next;
        });
        setV((prev) => {
          const next = new Map(prev);
          next.delete(id);
          return next;
        });
        log(`deleted V${id}`, "err");
      }
    }
  };

  const deleteEdge = (key: string) => {
    if (mode !== "delete") return;
    pushHistory();
    clearOverlays();
    setEdges((prev) => {
      const next = new Map(prev);
      next.delete(key);
      return next;
    });
    log(
      `deleted edge ${key.replace(",", "-V").replace(/^/, "U")}`,
      "err",
    );
  };

  const height = canvasHeight(U.size, V.size);
  const width = Math.max(canvasWidth, 500);

  const explainC =
    mode === "explain"
      ? candidatesOfS({ U, V, edges }, explainS, k)
      : { U: new Set<number>(), V: new Set<number>() };
  const explainMissing = missingInS({ U, V, edges }, explainS);

  let viewMode = "original";
  if (reducedPreview) viewMode = "reduction preview";
  if (solution)
    viewMode += (reducedPreview ? " + " : "") + "solution highlighted";
  if (mode === "explain" && partialSSize(explainS) > 0) {
    const sLabel = `S(|U|=${explainS.U.size},|V|=${explainS.V.size},miss=${explainMissing})`;
    viewMode = viewMode === "original" ? sLabel : `${viewMode} + ${sLabel}`;
  }
  if (mode === "explain" && fadeNonCandidates) {
    viewMode += ` + C(|U|=${explainC.U.size},|V|=${explainC.V.size})`;
  }
  if (mode === "explain" && explainHover && explainHighlight) {
    const focus = `${explainHover.isU ? "U" : "V"}${explainHover.id}`;
    viewMode += ` + ${focus} ${explainHighlight}`;
  }

  return {
    // graph
    U,
    V,
    edges,
    mode,
    selected,
    explainHover,
    explainHighlight,
    explainS,
    explainC,
    explainMissing,
    fadeNonCandidates,
    reducedPreview,
    solution,
    // params
    k,
    setK,
    theta,
    setTheta,
    rnU,
    setRnU,
    rnV,
    setRnV,
    density,
    setDensity,
    searchWarn,
    logs,
    canUndo: history.length > 0,
    canApplyReduce: !!reducedPreview,
    canClearSolution: !!solution,
    canClearExplainS: partialSSize(explainS) > 0,
    modeHint: MODE_HINTS[mode],
    stats: {
      u: U.size,
      v: V.size,
      e: edges.size,
      viewMode,
    },
    canvas: { width, height, scrollRef },
    // actions
    setMode,
    onExplainHover,
    toggleExplainS,
    clearExplainS,
    setFadeNonCandidates,
    clearGraph,
    undo,
    generateRandom,
    previewReduction,
    applyReduction,
    runMdbbSearch,
    clearSolution,
    saveJson,
    saveCsv,
    loadFile,
    addNodeAt,
    moveNode,
    beginDrag,
    onNodeClick,
    deleteEdge,
  };
}

export type GraphLabApi = ReturnType<typeof useGraphLab>;
