"use client";

import { useRef } from "react";
import type { GraphLabApi } from "@/hooks/useGraphLab";
import { Button } from "./Button";
import { Section } from "./Section";

export function Sidebar({ lab }: { lab: GraphLabApi }) {
  const fileRef = useRef<HTMLInputElement>(null);

  return (
    <div className="sidebar">
      <Section title="Parameters">
        <div className="row">
          <label htmlFor="kInput">k (missing-edge budget)</label>
          <input
            type="number"
            id="kInput"
            min={0}
            step={1}
            value={lab.k}
            onChange={(e) =>
              lab.setK(Math.max(0, parseInt(e.target.value || "0", 10) || 0))
            }
          />
        </div>
        <div className="row">
          <label htmlFor="thetaInput">θ (min side size)</label>
          <input
            type="number"
            id="thetaInput"
            min={1}
            step={1}
            value={lab.theta}
            onChange={(e) =>
              lab.setTheta(
                Math.max(1, parseInt(e.target.value || "1", 10) || 1),
              )
            }
          />
        </div>
      </Section>

      <Section title="Edit Mode">
        <div className="btnrow">
          <Button
            wide
            active={lab.mode === "move"}
            title="Drag nodes to reposition"
            onClick={() => lab.setMode("move")}
          >
            Move
          </Button>
          <Button
            wide
            active={lab.mode === "addnode"}
            title="Click U/V column to add a node"
            onClick={() => lab.setMode("addnode")}
          >
            Add Node
          </Button>
          <Button
            wide
            active={lab.mode === "addedge"}
            title="Click two nodes on opposite sides"
            onClick={() => lab.setMode("addedge")}
          >
            Add Edge
          </Button>
          <Button
            wide
            active={lab.mode === "delete"}
            title="Click a node or edge to remove it"
            onClick={() => lab.setMode("delete")}
          >
            Delete
          </Button>
        </div>
        <div className="hint">{lab.modeHint}</div>
        <div className="btnrow" style={{ marginTop: 8 }}>
          <Button wide danger onClick={lab.clearGraph}>
            Clear graph
          </Button>
          <Button wide disabled={!lab.canUndo} onClick={lab.undo}>
            Undo
          </Button>
        </div>
      </Section>

      <Section title="Random Graph">
        <div className="row">
          <label htmlFor="rnU">|U|</label>
          <input
            type="number"
            id="rnU"
            min={1}
            max={60}
            value={lab.rnU}
            onChange={(e) => lab.setRnU(parseInt(e.target.value, 10) || 1)}
          />
        </div>
        <div className="row">
          <label htmlFor="rnV">|V|</label>
          <input
            type="number"
            id="rnV"
            min={1}
            max={60}
            value={lab.rnV}
            onChange={(e) => lab.setRnV(parseInt(e.target.value, 10) || 1)}
          />
        </div>
        <div className="row">
          <label htmlFor="rDensity">edge density</label>
          <input
            type="range"
            id="rDensity"
            min={0.05}
            max={1}
            step={0.05}
            value={lab.density}
            onChange={(e) => lab.setDensity(parseFloat(e.target.value))}
          />
          <span
            style={{
              fontFamily: "var(--mono)",
              fontSize: 11,
              width: 32,
              textAlign: "right",
            }}
          >
            {lab.density}
          </span>
        </div>
        <Button
          wide
          primary
          style={{ width: "100%" }}
          onClick={lab.generateRandom}
        >
          Generate random graph
        </Button>
      </Section>

      <Section title="Reduction Phase">
        <div className="btnrow">
          <Button wide onClick={lab.previewReduction}>
            Preview reduction
          </Button>
          <Button
            wide
            disabled={!lab.canApplyReduce}
            onClick={lab.applyReduction}
          >
            Apply (commit)
          </Button>
        </div>
        <div className="hint">
          Ports <code>common_neighbor_reduction!</code> from reduction.jl using
          the current k/θ. Preview overlays removed nodes/edges in red without
          changing the graph; Apply commits it.
        </div>
      </Section>

      <Section title="k/θ-MDBB Search">
        <Button
          wide
          primary
          style={{ width: "100%" }}
          onClick={lab.runMdbbSearch}
        >
          Run search
        </Button>
        <Button
          wide
          disabled={!lab.canClearSolution}
          style={{ width: "100%", marginTop: 6 }}
          onClick={lab.clearSolution}
        >
          Clear solution
        </Button>
        <div className="hint">
          Finds S=(S<sub>U</sub>,S<sub>V</sub>) maximizing edges with |S
          <sub>U</sub>|,|S<sub>V</sub>|≥θ and missing edges ≤ k, via branch
          &amp; bound over the graph currently shown on canvas.
        </div>
        {lab.searchWarn && (
          <div className="hint warn">{lab.searchWarn}</div>
        )}
      </Section>

      <Section title="File">
        <div className="btnrow">
          <Button wide onClick={lab.saveJson}>
            Save .json
          </Button>
          <Button wide onClick={lab.saveCsv}>
            Save .csv
          </Button>
        </div>
        <div className="btnrow" style={{ marginTop: 6 }}>
          <label className="filelabel wide" style={{ flex: 1 }}>
            <Button
              wide
              style={{ width: "100%" }}
              onClick={() => fileRef.current?.click()}
            >
              Load file…
            </Button>
            <input
              ref={fileRef}
              type="file"
              accept=".json,.csv"
              onChange={(e) => {
                const file = e.target.files?.[0];
                if (file) void lab.loadFile(file);
                e.target.value = "";
              }}
            />
          </label>
        </div>
        <div className="hint">
          .csv matches load.jl&apos;s <code>user_id,item_id,timestamp</code>{" "}
          format. .json preserves layout + k/θ.
        </div>
      </Section>

      <Section title="Log">
        <div className="log">
          {lab.logs.map((entry) => (
            <div key={entry.id} className={entry.cls}>
              [{entry.time}] {entry.msg}
            </div>
          ))}
        </div>
      </Section>
    </div>
  );
}
