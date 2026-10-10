"use client";

import type { GraphLabApi } from "@/hooks/useGraphLab";
import { hopColor } from "@/lib/graph";

function sideLabel(isU: boolean, id: number) {
  return `${isU ? "U" : "V"}${id}`;
}

export function CanvasHud({ lab }: { lab: GraphLabApi }) {
  const subject = lab.hopView?.focus ?? lab.lastNode;
  const radius = lab.hopView?.radius ?? null;
  const hopButtons = [1, 2, 3, 4, 5, 6, 7, 8];
  if (radius != null && radius > 8 && !hopButtons.includes(radius)) {
    hopButtons.push(radius);
  }

  return (
    <div className="canvas-hud">
      <div className="hud-row">
        <button
          type="button"
          className={
            lab.mode === "addnode" && lab.placeSide === "U" ? "active-u" : ""
          }
          onClick={() => lab.armPlace("U")}
        >
          <kbd>U</kbd> place U
        </button>
        <button
          type="button"
          className={
            lab.mode === "addnode" && lab.placeSide === "V" ? "active-v" : ""
          }
          onClick={() => lab.armPlace("V")}
        >
          <kbd>V</kbd> place V
        </button>
        <button
          type="button"
          className={lab.mode === "addedge" ? "active" : ""}
          onClick={() => lab.setMode("addedge")}
        >
          <kbd>E</kbd> edge
        </button>
        <button
          type="button"
          className={lab.mode === "move" ? "active" : ""}
          onClick={() => lab.setMode("move")}
        >
          <kbd>M</kbd> move
        </button>
        <button
          type="button"
          className={lab.mode === "delete" ? "active" : ""}
          onClick={() => lab.setMode("delete")}
        >
          <kbd>D</kbd> delete
        </button>
        <span className="hud-note">Tab switches U/V · scroll for more room</span>
      </div>
      <div className="hud-row">
        <span className="hud-note">
          {subject
            ? `Hops of ${sideLabel(subject.isU, subject.id)}`
            : "Hover or click a node, then pick a hop distance"}
        </span>
        {hopButtons.map((n) => (
          <button
            key={n}
            type="button"
            className={`hop-n ${radius === n ? "active" : ""}`}
            disabled={!subject}
            title={`${n}-hop neighbors (key ${n <= 9 ? n : "]"})`}
            onClick={() => lab.showHops(n)}
          >
            {n}
          </button>
        ))}
        <button
          type="button"
          disabled={!subject}
          title="One fewer hop"
          onClick={() => lab.stepHop(-1)}
        >
          <kbd>[</kbd>
        </button>
        <button
          type="button"
          disabled={!subject}
          title="One more hop"
          onClick={() => lab.stepHop(1)}
        >
          <kbd>]</kbd>
        </button>
        <button
          type="button"
          disabled={!lab.hopView}
          onClick={lab.clearHops}
        >
          <kbd>Esc</kbd> clear
        </button>
        {lab.hopView && (
          <span className="hop-counts">
            {lab.hopView.counts.slice(1).map((count, index) => (
              <span key={index + 1}>
                <i style={{ background: hopColor(index + 1) }} />
                {index + 1}:{count}
              </span>
            ))}
          </span>
        )}
      </div>
    </div>
  );
}
