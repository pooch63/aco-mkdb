"use client";

import type { MouseEvent, ReactNode } from "react";
import type { GraphLabApi } from "@/hooks/useGraphLab";
import { partialSHas } from "@/lib/explain";
import { degreeOf, edgeKey, hopColor, nodeKey, parseEdgeKey } from "@/lib/graph";
import { NODE_R } from "@/lib/layout";

function clientToSvg(
  svg: SVGSVGElement,
  clientX: number,
  clientY: number,
  width: number,
  height: number,
) {
  const rect = svg.getBoundingClientRect();
  const scaleX = rect.width > 0 ? width / rect.width : 1;
  const scaleY = rect.height > 0 ? height / rect.height : 1;
  return {
    x: (clientX - rect.left) * scaleX,
    y: (clientY - rect.top) * scaleY,
  };
}

export function GraphCanvas({ lab }: { lab: GraphLabApi }) {
  const {
    U,
    V,
    edges,
    solution,
    reducedPreview,
    selected,
    explainS,
    explainC,
    fadeNonCandidates,
    hopView,
    canvas,
  } = lab;
  const { width, height, bindScroll } = canvas;
  const inExplain = lab.mode === "explain";
  const hops = hopView?.dist ?? null;

  const onSvgClick = (ev: MouseEvent<SVGSVGElement>) => {
    const target = ev.target as Element;
    const background =
      target === ev.currentTarget || target.classList.contains("canvas-bg");
    if (!background || lab.mode !== "addnode") return;
    const point = clientToSvg(
      ev.currentTarget,
      ev.clientX,
      ev.clientY,
      width,
      height,
    );
    lab.addNodeAt(point.x, point.y);
  };

  const onEdgeClick = (ev: MouseEvent<SVGLineElement>, key: string) => {
    ev.stopPropagation();
    lab.deleteEdge(key);
  };

  const onNodeMouseDown = (ev: MouseEvent, isU: boolean, id: number) => {
    if (lab.mode !== "move") return;
    ev.preventDefault();
    ev.stopPropagation();

    const svg = (ev.currentTarget as SVGGElement).ownerSVGElement;
    if (!svg) return;
    const origin = clientToSvg(svg, ev.clientX, ev.clientY, width, height);
    let moved = false;
    let saved = false;

    const onMove = (mv: globalThis.MouseEvent) => {
      const point = clientToSvg(svg, mv.clientX, mv.clientY, width, height);
      if (!moved && Math.hypot(point.x - origin.x, point.y - origin.y) < 4) {
        return;
      }
      moved = true;
      if (!saved) {
        saved = true;
        lab.beginDrag();
      }
      lab.moveNode(isU, id, point.x, point.y);
    };
    const onUp = () => {
      window.removeEventListener("mousemove", onMove);
      window.removeEventListener("mouseup", onUp);
      if (!moved) lab.focusForHops({ isU, id });
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
    const du = hops?.get(nodeKey(true, u));
    const dv = hops?.get(nodeKey(false, v));
    const onHop =
      hops != null &&
      du != null &&
      dv != null &&
      Math.abs(du - dv) === 1;

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

    if (onHop) {
      stroke = hopColor(Math.max(du ?? 0, dv ?? 0));
      strokeWidth = 2.6;
      opacity = 1;
    } else if (hops && !inS) {
      opacity = Math.min(opacity, 0.1);
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
              opacity={hops || fadeNonCandidates ? 0.2 : 0.85}
            />,
          );
        }
      }
    }
  }

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
    const hopDist = hops?.get(nk);
    const inS = inExplain && partialSHas(explainS, isU, id);
    const inC =
      inExplain && (isU ? explainC.U.has(id) : explainC.V.has(id));
    const fadeOut = fadeNonCandidates && inExplain && !inS && !inC && hopDist == null;

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
    if (inS) {
      strokeColor = "var(--s-color)";
      strokeW = 3.5;
      opacity = 1;
    } else if (fadeNonCandidates && inC) {
      strokeColor = "var(--c-color)";
      strokeW = 2.5;
    }

    if (hops) {
      if (hopDist == null) {
        if (!inS) opacity = Math.min(opacity, 0.16);
      } else {
        opacity = 1;
        if (!inS) {
          strokeColor = hopColor(hopDist);
          strokeW = hopDist === 0 ? 3.5 : 2.5;
        }
      }
    } else if (fadeOut) {
      opacity = 0.1;
      strokeW = 0;
      strokeColor = "none";
    }

    if (isSelected) {
      strokeColor = "#1a1d24";
      strokeW = 2.5;
      opacity = 1;
    }

    const deg = degreeOf({ U, V, edges }, isU, id);
    const title =
      `${isU ? "U" : "V"}${id}  deg=${deg}` +
      (hopDist != null ? `  hop=${hopDist}` : "") +
      (isRemoved ? "  [removed by reduction]" : "") +
      (inSolution ? "  [in solution]" : "") +
      (inS ? "  [in S]" : "") +
      (inC ? "  [in C]" : "");

    return (
      <g
        key={`${isU ? "u" : "v"}${id}`}
        transform={`translate(${pos.x},${pos.y})`}
        style={{ cursor: lab.mode === "move" ? "grab" : "pointer" }}
        onMouseDown={(ev) => onNodeMouseDown(ev, isU, id)}
        onMouseEnter={() => lab.setPointer({ isU, id })}
        onMouseLeave={() => lab.setPointer(null)}
        onClick={(ev) => {
          ev.stopPropagation();
          if (inExplain && ev.shiftKey) {
            lab.toggleExplainS(isU, id);
            return;
          }
          if (inExplain) lab.focusForHops({ isU, id });
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
          fill={isRemoved || fadeOut || (hops && hopDist == null && !inS) ? "#6b7380" : "#ffffff"}
          fontWeight={700}
          fontSize={10}
          opacity={fadeOut || (hops && hopDist == null && !inS) ? 0.45 : 1}
        >
          {id}
        </text>
        {hopDist != null && (
          <g className="hop-badge" transform={`translate(${NODE_R - 2},${-NODE_R + 2})`}>
            <circle r={7.5} fill={hopColor(hopDist)} stroke="#ffffff" strokeWidth={1.5} />
            <text
              y={3}
              textAnchor="middle"
              fill="#ffffff"
              fontSize={8}
              fontWeight={700}
            >
              {hopDist}
            </text>
          </g>
        )}
      </g>
    );
  }

  const cursor =
    lab.mode === "addnode" || lab.mode === "addedge" ? "crosshair" : "default";

  return (
    <div className="canvas-scroll" ref={bindScroll}>
      <svg
        width={width}
        height={height}
        viewBox={`0 0 ${width} ${height}`}
        onClick={onSvgClick}
        style={{ cursor }}
      >
        <defs>
          <pattern
            id="place-grid"
            width="48"
            height="48"
            patternUnits="userSpaceOnUse"
          >
            <circle cx="1" cy="1" r="1.15" className="grid-dot" />
          </pattern>
        </defs>
        <rect
          className="canvas-bg"
          x={0}
          y={0}
          width={width}
          height={height}
          fill="url(#place-grid)"
        />
        {edgeLines}
        {missingLines}
        {[...U].map(([id, pos]) => drawNode(true, id, pos))}
        {[...V].map(([id, pos]) => drawNode(false, id, pos))}
      </svg>
    </div>
  );
}
