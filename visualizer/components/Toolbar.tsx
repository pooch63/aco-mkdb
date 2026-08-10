export function Toolbar({
  stats,
}: {
  stats: { u: number; v: number; e: number; viewMode: string };
}) {
  return (
    <div className="toolbar">
      <div className="stats">
        <span>
          |U|=<b>{stats.u}</b>
        </span>
        <span>
          |V|=<b>{stats.v}</b>
        </span>
        <span>
          |E|=<b>{stats.e}</b>
        </span>
        <span>view: {stats.viewMode}</span>
      </div>
      <div className="legend">
        <div className="item">
          <span className="swatch" style={{ background: "var(--u-color)" }} />U
          node
        </div>
        <div className="item">
          <span className="swatch" style={{ background: "var(--v-color)" }} />V
          node
        </div>
        <div className="item">
          <span className="swatch" style={{ background: "var(--edge)" }} />
          edge
        </div>
        <div className="item">
          <span className="swatch" style={{ background: "var(--removed)" }} />
          removed by reduction
        </div>
        <div className="item">
          <span className="swatch" style={{ background: "var(--solution)" }} />
          solution S
        </div>
        <div className="item">
          <span
            className="swatch"
            style={{ background: "var(--solution-missing)" }}
          />
          missing edge in S
        </div>
      </div>
    </div>
  );
}
