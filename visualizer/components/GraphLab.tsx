"use client";

import { useGraphLab } from "@/hooks/useGraphLab";
import { GraphCanvas } from "./GraphCanvas";
import { Header } from "./Header";
import { Sidebar } from "./Sidebar";
import { Toolbar } from "./Toolbar";

export function GraphLab() {
  const lab = useGraphLab();

  return (
    <div className="graph-lab">
      <Header />
      <div className="layout">
        <Sidebar lab={lab} />
        <div className="canvas-wrap">
          <Toolbar stats={lab.stats} />
          <GraphCanvas lab={lab} />
        </div>
      </div>
    </div>
  );
}
