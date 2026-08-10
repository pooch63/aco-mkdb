import type { Point } from "./types";

export const NODE_R = 13;
export const COL_MARGIN = 90;
export const ROW_SPACING = 42;
export const TOP_MARGIN = 46;

export function layoutColumn(
  map: Map<number, Point>,
  width: number,
  isU: boolean,
) {
  const ids = [...map.keys()].sort((a, b) => a - b);
  const x = isU ? COL_MARGIN : width - COL_MARGIN;
  ids.forEach((id, i) => {
    map.set(id, { x, y: TOP_MARGIN + i * ROW_SPACING });
  });
}

export function relayoutAll(
  U: Map<number, Point>,
  V: Map<number, Point>,
  width: number,
) {
  const w = Math.max(width, 500);
  layoutColumn(U, w, true);
  layoutColumn(V, w, false);
}

export function canvasHeight(
  uSize: number,
  vSize: number,
): number {
  const n = Math.max(uSize, vSize, 1);
  return TOP_MARGIN * 2 + n * ROW_SPACING;
}
