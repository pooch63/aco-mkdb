"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  createDemoGraph,
  edgeKey,
  hopDistances,
  snapshotOf,
} from "@/lib/graph";
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
import { PLACE_PAD, relayoutAll, worldSize } from "@/lib/layout";
import { runReduction } from "@/lib/reduction";
import { runSearch } from "@/lib/search";
import type {
  EditMode,
  GraphJson,
  LogEntry,
  NodeRef,
  PlaceSide,
  Point,
  ReductionPreview,
  SearchResult,
} from "@/lib/types";

const MODE_HINTS: Record<EditMode, string> = {
  move: "Drag a node anywhere. Click one, then press 1–9 for its hop neighbors.",
  addnode:
    "Click anywhere to drop a node. U and V choose the side. The canvas scrolls well past the window.",
  addedge:
    "Press E, then click a U node and a V node (either order) to toggle the edge.",
  delete: "Click a node to delete it, or click an edge to delete just that edge. D selects this mode.",
  explain:
    "Click a node or press 1–9 to see that many hops. Shift+click toggles S. Fade non-candidates dims nodes outside S∪C.",
};

const HOP_MAX = 24;

function isFormField(target: EventTarget | null): boolean {
  if (!(target instanceof HTMLElement)) return false;
  const tag = target.tagName;
  return (
    tag === "INPUT" ||
    tag === "TEXTAREA" ||
    tag === "SELECT" ||
    target.isContentEditable
  );
}

const HISTORY_CAP = 25;

export function useGraphLab() {
  const [boot] = useState(() => {
    const graph = createDemoGraph();
    relayoutAll(graph.U, graph.V, 1100);
    return graph;
  });
  const [nextUId, setNextUId] = useState(boot.nextUId);
  const [nextVId, setNextVId] = useState(boot.nextVId);
  const [U, setU] = useState(boot.U);
  const [V, setV] = useState(boot.V);
  const [edges, setEdges] = useState(boot.edges);

  const [mode, setModeState] = useState<EditMode>("move");
  const [placeSide, setPlaceSide] = useState<PlaceSide>("U");
  const [selected, setSelected] = useState<NodeRef | null>(null);
  const [lastNode, setLastNode] = useState<NodeRef | null>(null);
  const [hop, setHop] = useState<{ focus: NodeRef; radius: number } | null>(
    null,
  );
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
  const [view, setView] = useState({ w: 960, h: 720 });
  const logId = useRef(0);
  const scrollRef = useRef<HTMLDivElement | null>(null);
  const didInitLayout = useRef(false);
  const pointerRef = useRef<NodeRef | null>(null);
  const pinnedRef = useRef<NodeRef | null>(null);
  const hopRef = useRef<{ focus: NodeRef; radius: number } | null>(null);
  hopRef.current = hop;

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
    (nextU: Map<number, Point>, nextV: Map<number, Point>, width?: number) => {
      const w =
        width ?? Math.max(scrollRef.current?.clientWidth ?? view.w, 280);
      relayoutAll(nextU, nextV, w);
      setU(new Map(nextU));
      setV(new Map(nextV));
    },
    [view.w],
  );

  const graphRef = useRef({ U, V });
  graphRef.current = { U, V };

  // Attach the scrollport. The ref callback is what actually sees the node;
  // a mount effect can run before that ref is set.
  const bindScroll = useCallback(
    (el: HTMLDivElement | null) => {
      scrollRef.current = el;
      if (!el || didInitLayout.current) return;
      didInitLayout.current = true;
      const w = Math.max(el.clientWidth, 280);
      const u = new Map(graphRef.current.U);
      const v = new Map(graphRef.current.V);
      relayoutAll(u, v, w);
      setU(u);
      setV(v);
      log(
        "ready — demo graph loaded. U/V place nodes, E adds edges, 1–9 shows hop neighbors.",
        "info",
      );
    },
    [log],
  );

  useEffect(() => {
    const el = scrollRef.current;
    if (!el) return;
    const measure = () => {
      setView({
        w: Math.max(el.clientWidth, 500),
        h: Math.max(el.clientHeight, 400),
      });
    };
    measure();
    const observer = new ResizeObserver(measure);
    observer.observe(el);
    return () => observer.disconnect();
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
    if (m !== "explain") {
      setExplainS(emptyPartialS());
      setFadeNonCandidates(false);
    }
  };

  const armPlace = useCallback((side: PlaceSide) => {
    setPlaceSide(side);
    setModeState("addnode");
    setSelected(null);
    setExplainS(emptyPartialS());
    setFadeNonCandidates(false);
  }, []);

  const rememberNode = useCallback((node: NodeRef) => {
    pinnedRef.current = node;
    setLastNode(node);
  }, []);

  const setPointer = useCallback(
    (node: NodeRef | null) => {
      pointerRef.current = node;
      if (node) rememberNode(node);
    },
    [rememberNode],
  );

  const showHops = useCallback(
    (radius: number, focus?: NodeRef | null) => {
      const target =
        focus ??
        pointerRef.current ??
        hopRef.current?.focus ??
        pinnedRef.current ??
        null;
      if (!target) return;
      const next = Math.max(1, Math.min(HOP_MAX, radius));
      rememberNode(target);
      const nextHop = { focus: target, radius: next };
      hopRef.current = nextHop;
      setHop(nextHop);
    },
    [rememberNode],
  );

  const clearHops = useCallback(() => {
    hopRef.current = null;
    setHop(null);
  }, []);

  const focusForHops = useCallback(
    (node: NodeRef) => {
      rememberNode(node);
      setHop((prev) => {
        const nextHop = {
          focus: node,
          radius: prev?.radius ?? 1,
        };
        hopRef.current = nextHop;
        return nextHop;
      });
    },
    [rememberNode],
  );

  const stepHop = useCallback((delta: number) => {
    const hovering = pointerRef.current;
    setHop((prev) => {
      const focus = hovering ?? prev?.focus ?? pinnedRef.current;
      if (!focus) return prev;
      const same =
        prev != null &&
        prev.focus.isU === focus.isU &&
        prev.focus.id === focus.id;
      const base = same || !hovering ? (prev?.radius ?? 0) : 0;
      const nextHop = {
        focus,
        radius: Math.max(1, Math.min(HOP_MAX, base + delta)),
      };
      hopRef.current = nextHop;
      return nextHop;
    });
    if (hovering) rememberNode(hovering);
  }, [rememberNode]);

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

  const actionsRef = useRef({
    armPlace,
    setMode,
    showHops,
    clearHops,
    stepHop,
    placeSide,
  });
  actionsRef.current = {
    armPlace,
    setMode,
    showHops,
    clearHops,
    stepHop,
    placeSide,
  };

  useEffect(() => {
    const onKey = (ev: KeyboardEvent) => {
      if (isFormField(ev.target)) return;
      if (ev.metaKey || ev.ctrlKey || ev.altKey) return;
      const actions = actionsRef.current;
      const key = ev.key;

      if (key === "u" || key === "U") {
        ev.preventDefault();
        actions.armPlace("U");
      } else if (key === "v" || key === "V") {
        ev.preventDefault();
        actions.armPlace("V");
      } else if (key === "Tab") {
        ev.preventDefault();
        actions.armPlace(actions.placeSide === "U" ? "V" : "U");
      } else if (key === "e" || key === "E") {
        ev.preventDefault();
        const from = pointerRef.current;
        actions.setMode("addedge");
        if (from) setSelected(from);
      } else if (key === "m" || key === "M") {
        ev.preventDefault();
        actions.setMode("move");
      } else if (key === "d" || key === "D" || key === "Delete") {
        ev.preventDefault();
        actions.setMode("delete");
      } else if (key === "Escape") {
        ev.preventDefault();
        actions.clearHops();
        setSelected(null);
      } else if (key === "0") {
        ev.preventDefault();
        actions.clearHops();
      } else if (/^[1-9]$/.test(key)) {
        ev.preventDefault();
        actions.showHops(parseInt(key, 10));
      } else if (key === "]" || key === "=" || key === "+") {
        ev.preventDefault();
        actions.stepHop(1);
      } else if (key === "[" || key === "-" || key === "_") {
        ev.preventDefault();
        actions.stepHop(-1);
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

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

  const addNodeAt = (x: number, y: number) => {
    if (mode !== "addnode") return;
    pushHistory();
    clearOverlays();
    const point = {
      x: Math.max(PLACE_PAD, x),
      y: Math.max(PLACE_PAD, y),
    };
    if (placeSide === "U") {
      const id = nextUId;
      setNextUId(id + 1);
      setU((prev) => {
        const next = new Map(prev);
        next.set(id, point);
        return next;
      });
      log(`added U${id}`, "info");
    } else {
      const id = nextVId;
      setNextVId(id + 1);
      setV((prev) => {
        const next = new Map(prev);
        next.set(id, point);
        return next;
      });
      log(`added V${id}`, "info");
    }
  };

  const moveNode = (isU: boolean, id: number, x: number, y: number) => {
    const setter = isU ? setU : setV;
    setter((prev) => {
      const next = new Map(prev);
      const p = next.get(id);
      if (!p) return prev;
      next.set(id, {
        x: Math.max(PLACE_PAD, x),
        y: Math.max(PLACE_PAD, y),
      });
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

  const { width, height } = worldSize(U, V, view.w, view.h);

  const hopView = useMemo(() => {
    if (!hop) return null;
    const exists = hop.focus.isU
      ? U.has(hop.focus.id)
      : V.has(hop.focus.id);
    if (!exists) return null;
    const dist = hopDistances(
      { U, V, edges },
      hop.focus.isU,
      hop.focus.id,
      hop.radius,
    );
    const counts = Array.from({ length: hop.radius + 1 }, () => 0);
    for (const d of dist.values()) {
      if (d < counts.length) counts[d] += 1;
    }
    return { focus: hop.focus, radius: hop.radius, dist, counts };
  }, [hop, U, V, edges]);

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
  if (hopView) {
    const focus = `${hopView.focus.isU ? "U" : "V"}${hopView.focus.id}`;
    viewMode += ` + ${hopView.radius}-hop ${focus}`;
  }

  return {
    // graph
    U,
    V,
    edges,
    mode,
    placeSide,
    selected,
    lastNode,
    hopView,
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
    modeHint:
      mode === "addnode"
        ? `Click anywhere to place a ${placeSide} node. Press ${placeSide === "U" ? "V" : "U"} (or Tab) to switch sides. Scroll for more room.`
        : MODE_HINTS[mode],
    stats: {
      u: U.size,
      v: V.size,
      e: edges.size,
      viewMode,
    },
    canvas: { width, height, scrollRef, bindScroll },
    // actions
    setMode,
    armPlace,
    setPointer,
    showHops,
    stepHop,
    clearHops,
    focusForHops,
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
