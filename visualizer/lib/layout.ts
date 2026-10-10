import type { Point } from "./types";

export const NODE_R = 13;
export const COL_MARGIN = 110;
export const ROW_SPACING = 56;
export const TOP_MARGIN = 64;
/** Empty workspace beyond the current drawing, so new nodes have room. */
export const WORLD_MIN_W = 3200;
export const WORLD_MIN_H = 2200;
export const PLACE_PAD = 28;

function columnX(width: number, isU: boolean): number {
  const margin = Math.min(COL_MARGIN, Math.max(48, width * 0.08));
  return isU ? margin : Math.max(margin + 80, width - margin);
}

export function layoutColumn(
  map: Map<number, Point>,
  width: number,
  isU: boolean,
) {
  const ids = [...map.keys()].sort((a, b) => a - b);
  const x = columnX(width, isU);
  ids.forEach((id, i) => {
    map.set(id, { x, y: TOP_MARGIN + i * ROW_SPACING });
  });
}

export function relayoutAll(
  U: Map<number, Point>,
  V: Map<number, Point>,
  width: number,
) {
  const w = Math.max(width, 280);
  layoutColumn(U, w, true);
  layoutColumn(V, w, false);
}

/** Canvas big enough to scroll around in, and to hold every placed node. */
export function worldSize(
  U: Map<number, Point>,
  V: Map<number, Point>,
  viewW: number,
  viewH: number,
): { width: number; height: number } {
  let maxX = 0;
  let maxY = 0;
  for (const p of U.values()) {
    maxX = Math.max(maxX, p.x);
    maxY = Math.max(maxY, p.y);
  }
  for (const p of V.values()) {
    maxX = Math.max(maxX, p.x);
    maxY = Math.max(maxY, p.y);
  }
  return {
    width: Math.max(WORLD_MIN_W, viewW, Math.ceil(maxX + 360)),
    height: Math.max(WORLD_MIN_H, viewH, Math.ceil(maxY + 360)),
  };
}
