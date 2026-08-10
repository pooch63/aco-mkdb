"use client";

import type { MouseEvent, ReactNode } from "react";
import type { GraphLabApi } from "@/hooks/useGraphLab";
import { degreeOf, edgeKey, parseEdgeKey } from "@/lib/graph";
import { COL_MARGIN, NODE_R } from "@/lib/layout";

export function GraphCanvas({ lab }: { lab: GraphLabApi }) {
  const { U, V, edges, solution, reducedPreview, selected, canvas } = lab;
  const { width, height, scrollRef } = canvas;

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

    let stroke = "var(--edge)";
    let strokeWidth = 1.4;
    let dash = "none";
    let opacity = 1;
    if (isRemoved) {
      stroke = "var(--removed)";
      dash = "4,3";
      opacity = 0.55;
    } else if (inSolution) {
      stroke = "var(--solution)";
      strokeWidth = 2.4;
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
              opacity={0.85}
            />,
          );
        }
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

    let fill = isU ? "var(--u-color)" : "var(--v-color)";
    let opacity = 1;
    let strokeColor = "none";
    let strokeW = 0;
    if (isRemoved) {
      fill = "#2a2f3a";
      opacity = 0.45;
      strokeColor = "var(--removed)";
      strokeW = 1.5;
    }
    if (inSolution) {
      strokeColor = "var(--solution)";
      strokeW = 3;
    }
    if (isSelected) {
      strokeColor = "#fff";
      strokeW = 2.5;
    }

    const deg = degreeOf({ U, V, edges }, isU, id);
    const title =
      `${isU ? "U" : "V"}${id}  deg=${deg}` +
      (isRemoved ? "  [removed by reduction]" : "") +
      (inSolution ? "  [in solution]" : "");

    return (
      <g
        key={`${isU ? "u" : "v"}${id}`}
        transform={`translate(${pos.x},${pos.y})`}
        style={{ cursor: "pointer" }}
        onMouseDown={(ev) => onNodeMouseDown(ev, isU, id)}
        onClick={(ev) => {
          ev.stopPropagation();
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
          fill={isRemoved ? "#5a6270" : "#0d0f13"}
          fontWeight={700}
          fontSize={10}
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
