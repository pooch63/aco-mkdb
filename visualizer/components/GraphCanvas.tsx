"use client";

import type { MouseEvent, ReactNode } from "react";
import type { GraphLabApi } from "@/hooks/useGraphLab";
import { partialSHas } from "@/lib/explain";
import { degreeOf, edgeKey, neighborsOf, parseEdgeKey } from "@/lib/graph";
import { COL_MARGIN, NODE_R } from "@/lib/layout";
import type { ExplainHighlight, GraphBag, NodeRef } from "@/lib/types";

function nodeKey(isU: boolean, id: number): string {
  return `${isU ? "u" : "v"}:${id}`;
}

/** Nodes to emphasize for the hovered focus under the given explain highlight. */
function explainHighlightedNodes(
  g: GraphBag,
  focus: NodeRef,
  highlight: ExplainHighlight,
): Set<string> {
  const out = new Set<string>([nodeKey(focus.isU, focus.id)]);
  const nbrs = new Set(neighborsOf(g, focus.isU, focus.id));
  const own = focus.isU ? g.U : g.V;
  const other = focus.isU ? g.V : g.U;

  if (highlight === "neighbors") {
    for (const id of nbrs) out.add(nodeKey(!focus.isU, id));
  } else {
    for (const id of own.keys()) {
      if (id !== focus.id) out.add(nodeKey(focus.isU, id));
    }
    for (const id of other.keys()) {
      if (!nbrs.has(id)) out.add(nodeKey(!focus.isU, id));
    }
  }
  return out;
}

export function GraphCanvas({ lab }: { lab: GraphLabApi }) {
  const {
    U,
    V,
    edges,
    solution,
    reducedPreview,
    selected,
    explainHover,
    explainHighlight,
    explainS,
    explainC,
    fadeNonCandidates,
    canvas,
  } = lab;
  const { width, height, scrollRef } = canvas;
  const inExplain = lab.mode === "explain";

  const explaining =
    inExplain && explainHover && explainHighlight
      ? explainHighlightedNodes({ U, V, edges }, explainHover, explainHighlight)
      : null;
  const focusKey =
    explainHover != null ? nodeKey(explainHover.isU, explainHover.id) : null;

  const onSvgClick = (ev: MouseEvent<SVGSVGElement>) => {
    if (ev.target !== ev.currentTarget) return;
    if (lab.mode !== "addnode") return;
    const rect = ev.currentTarget.getBoundingClientRect();
    const scaleX = width / rect.width;
    const x = (ev.clientX - rect.left) * scaleX;
    const y = (ev.clientY - rect.top) * scaleX;
    lab.addNodeAt(x, y, width);
  };

  const onEdgeClick = (ev: MouseEvent<SVGLineElement>, key: string) => {
    ev.stopPropagation();
    lab.deleteEdge(key);
  };

  const onNodeMouseDown = (
    ev: MouseEvent,
    isU: boolean,
    id: number,
  ) => {
    if (lab.mode !== "move") return;
    ev.preventDefault();
    ev.stopPropagation();
    lab.beginDrag();

    const svg = (ev.currentTarget as SVGGElement).ownerSVGElement;
    if (!svg) return;

    const onMove = (mv: globalThis.MouseEvent) => {
      const rect = svg.getBoundingClientRect();
      const scaleX = width / rect.width;
      const y = (mv.clientY - rect.top) * scaleX;
      lab.moveNode(isU, id, y);
    };
    const onUp = () => {
      window.removeEventListener("mousemove", onMove);
      window.removeEventListener("mouseup", onUp);
    };
    window.addEventListener("mousemove", onMove);
    window.addEventListener("mouseup", onUp);
  };

  const edgeLines: ReactNode[] = [];
  for (const key of edges.keys()) {
    const [u, v] = parseEdgeKey(key);
    const pu = U.get(u);
    const pv = V.get(v);
    if (!pu || !pv) continue;

    const isRemoved = reducedPreview?.removedEdges.has(key);
    const inSolution = solution && solution.U.has(u) && solution.V.has(v);
    const inS = inExplain && explainS.U.has(u) && explainS.V.has(v);

    let stroke = "var(--edge)";
    let strokeWidth = 1.4;
    let dash = "none";
    let opacity = 1;
    if (isRemoved) {
      stroke = "var(--removed)";
      dash = "4,3";
      opacity = 0.55;
    } else if (inS) {
      stroke = "var(--s-color)";
      strokeWidth = 2.6;
    } else if (inSolution) {
      stroke = "var(--solution)";
      strokeWidth = 2.4;
    }

    if (explaining && explainHover && explainHighlight === "neighbors") {
      const touchesFocus =
        (explainHover.isU && explainHover.id === u) ||
        (!explainHover.isU && explainHover.id === v);
      if (touchesFocus) {
        stroke = "var(--explain)";
        strokeWidth = 2.4;
        opacity = 1;
      } else if (!inS) {
        opacity = Math.min(opacity, 0.18);
      }
    } else if (explaining && !inS) {
      opacity = Math.min(opacity, 0.18);
    } else if (fadeNonCandidates && inExplain) {
      const uKeep = explainS.U.has(u) || explainC.U.has(u);
      const vKeep = explainS.V.has(v) || explainC.V.has(v);
      if (!uKeep || !vKeep) opacity = Math.min(opacity, 0.12);
    }

    edgeLines.push(
      <line
        key={key}
        x1={pu.x}
        y1={pu.y}
        x2={pv.x}
        y2={pv.y}
        stroke={stroke}
        strokeWidth={strokeWidth}
        strokeDasharray={dash}
        opacity={opacity}
        data-edge={key}
        onClick={(ev) => onEdgeClick(ev, key)}
        style={{ cursor: lab.mode === "delete" ? "pointer" : undefined }}
      />,
    );
  }

  const missingLines: ReactNode[] = [];
  if (solution) {
    for (const u of solution.U) {
      for (const v of solution.V) {
        if (!edges.has(edgeKey(u, v))) {
          const pu = U.get(u);
          const pv = V.get(v);
          if (!pu || !pv) continue;
          missingLines.push(
            <line
              key={`miss-${u}-${v}`}
              x1={pu.x}
              y1={pu.y}
              x2={pv.x}
              y2={pv.y}
              stroke="var(--solution-missing)"
              strokeWidth={1.6}
              strokeDasharray="3,3"
              opacity={explaining || fadeNonCandidates ? 0.2 : 0.85}
            />,
          );
        }
      }
    }
  }

  // Missing edges inside explain S (always shown while S has both sides).
  if (inExplain && explainS.U.size > 0 && explainS.V.size > 0) {
    for (const u of explainS.U) {
      for (const v of explainS.V) {
        if (edges.has(edgeKey(u, v))) continue;
        const pu = U.get(u);
        const pv = V.get(v);
        if (!pu || !pv) continue;
        missingLines.push(
          <line
            key={`s-miss-${u}-${v}`}
            x1={pu.x}
            y1={pu.y}
            x2={pv.x}
            y2={pv.y}
            stroke="var(--s-missing)"
            strokeWidth={1.8}
            strokeDasharray="3,3"
            opacity={0.9}
          />,
        );
      }
    }
  }

  function drawNode(isU: boolean, id: number, pos: { x: number; y: number }) {
    const isRemoved =
      reducedPreview &&
      (isU ? reducedPreview.removedU.has(id) : reducedPreview.removedV.has(id));
    const inSolution =
      solution && (isU ? solution.U.has(id) : solution.V.has(id));
    const isSelected =
      selected && selected.isU === isU && selected.id === id;
    const nk = nodeKey(isU, id);
    const isFocus = focusKey === nk;
    const isExplainHit = explaining?.has(nk) ?? false;
    const isExplainDim = explaining != null && !isExplainHit;
    const inS = inExplain && partialSHas(explainS, isU, id);
    const inC =
      inExplain &&
      (isU ? explainC.U.has(id) : explainC.V.has(id));
    const fadeOut =
      fadeNonCandidates && inExplain && !inS && !inC;

    let fill = isU ? "var(--u-color)" : "var(--v-color)";
    let opacity = 1;
    let strokeColor = "none";
    let strokeW = 0;
    if (isRemoved) {
      fill = "#c8ced8";
      opacity = 0.55;
      strokeColor = "var(--removed)";
      strokeW = 1.5;
    }
    if (inSolution) {
      strokeColor = "var(--solution)";
      strokeW = 3;
    }
    if (isSelected) {
      strokeColor = "#1a1d24";
      strokeW = 2.5;
    }

    // S is always highlighted in explain mode.
    if (inS) {
      strokeColor = "var(--s-color)";
      strokeW = 3.5;
      opacity = 1;
    } else if (fadeNonCandidates && inC) {
      strokeColor = "var(--c-color)";
      strokeW = 2.5;
    }

    if (inExplain && explainHover && isFocus && !explainHighlight && !inS) {
      strokeColor = "var(--explain)";
      strokeW = 2.5;
    }
    if (explaining) {
      if (isFocus) {
        strokeColor = inS ? "var(--s-color)" : "var(--explain)";
        strokeW = 3.5;
        opacity = 1;
      } else if (isExplainHit) {
        if (!inS) {
          strokeColor = "var(--explain)";
          strokeW = 2.5;
        }
        opacity = 1;
      } else if (isExplainDim && !inS) {
        opacity = fadeOut ? 0.08 : 0.18;
        if (!inC || !fadeNonCandidates) {
          strokeW = Math.min(strokeW, 0);
          if (!inS) strokeColor = "none";
        }
      }
    } else if (fadeOut) {
      opacity = 0.1;
      strokeW = 0;
      strokeColor = "none";
    }

    const deg = degreeOf({ U, V, edges }, isU, id);
    const title =
      `${isU ? "U" : "V"}${id}  deg=${deg}` +
      (isRemoved ? "  [removed by reduction]" : "") +
      (inSolution ? "  [in solution]" : "") +
      (inS ? "  [in S]" : "") +
      (inC ? "  [in C]" : "") +
      (isFocus && explainHighlight
        ? `  [explain: ${explainHighlight}]`
        : "");

    return (
      <g
        key={`${isU ? "u" : "v"}${id}`}
        transform={`translate(${pos.x},${pos.y})`}
        style={{ cursor: "pointer" }}
        onMouseDown={(ev) => onNodeMouseDown(ev, isU, id)}
        onMouseEnter={() => lab.onExplainHover({ isU, id })}
        onMouseLeave={() => lab.onExplainHover(null)}
        onClick={(ev) => {
          ev.stopPropagation();
          if (inExplain && ev.shiftKey) {
            lab.toggleExplainS(isU, id);
            return;
          }
          lab.onNodeClick(isU, id);
        }}
      >
        <title>{title}</title>
        <circle
          r={NODE_R}
          fill={fill}
          opacity={opacity}
          stroke={strokeColor}
          strokeWidth={strokeW}
        />
        <text
          x={0}
          y={3.5}
          textAnchor="middle"
          className="node-label"
          fill={isRemoved || fadeOut || (isExplainDim && !inS) ? "#6b7380" : "#ffffff"}
          fontWeight={700}
          fontSize={10}
          opacity={fadeOut || (isExplainDim && !inS) ? 0.45 : 1}
        >
          {id}
        </text>
      </g>
    );
  }

  return (
    <div className="canvas-scroll" ref={scrollRef}>
      <svg
        width={width}
        height={height}
        viewBox={`0 0 ${width} ${height}`}
        onClick={onSvgClick}
      >
        <text x={COL_MARGIN} y={20} className="col-label" textAnchor="middle">
          U
        </text>
        <text
          x={width - COL_MARGIN}
          y={20}
          className="col-label"
          textAnchor="middle"
        >
          V
        </text>
        {edgeLines}
        {missingLines}
        {[...U].map(([id, pos]) => drawNode(true, id, pos))}
        {[...V].map(([id, pos]) => drawNode(false, id, pos))}
      </svg>
    </div>
  );
}
