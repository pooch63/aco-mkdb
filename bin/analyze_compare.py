#!/usr/bin/env python3
"""
analyze_compare.py — Analyze a compare-methods.jl JSON output file.

Usage:
    python bin/analyze_compare.py results/my_ab.json
    python bin/analyze_compare.py results/my_ab.json --sort=dataset
    python bin/analyze_compare.py results/my_ab.json --sort=pct --top=10
    python bin/analyze_compare.py results/my_ab.json --csv
    python bin/analyze_compare.py results/my_ab.json --filter=challenger
    python bin/analyze_compare.py results/my_ab.json --plot
    python bin/analyze_compare.py results/my_ab.json --plot=results/heat_vs_density.png

Options:
    --sort=FIELD          Sort per-graph table by: dataset (default), edges, pct, delta, verdict
    --top=N               Only show the top N rows (after sorting) in the per-graph table
    --filter=VERDICT      Only show rows with this verdict: challenger, baseline, tie
    --csv                 Emit a CSV instead of the human-readable report
    --plot[=PATH]         Plot graph density vs percent change in heat mean (saves to PATH or <stem>_heat_density.png)
    --density-type=TYPE   Density type for heat plot: reduced (default) or original
    --title=TITLE         Custom title for the correlation plot
    --no-color            Disable ANSI color output
"""

import json, sys, os, statistics, csv, io

# ── ANSI helpers ──────────────────────────────────────────────────────────────
USE_COLOR = True

def _c(code, text): return f"\033[{code}m{text}\033[0m" if USE_COLOR else text
def green(t):  return _c("32", t)
def red(t):    return _c("31", t)
def yellow(t): return _c("33", t)
def bold(t):   return _c("1",  t)
def dim(t):    return _c("2",  t)
def cyan(t):   return _c("36", t)

# ── Stat helpers ──────────────────────────────────────────────────────────────
def safe_pct(val):
    if val is None: return None
    try: return float(val)
    except (TypeError, ValueError): return None

def median(vals):
    if not vals: return None
    s = sorted(vals); n = len(s); mid = n // 2
    return s[mid] if n % 2 else (s[mid-1] + s[mid]) / 2.0

def pct_str(v, plus=True):
    if v is None: return "n/a"
    sign = "+" if (plus and v >= 0) else ""
    return f"{sign}{v:.2f}%"

def fmt_time(s):
    if s is None: return "n/a"
    return f"{s*1000:.1f} ms" if s < 1 else f"{s:.3f} s"

def get_graph_density(row, density_type="reduced"):
    """Extract bipartite edge density |E| / (|U|·|V|)."""
    if density_type == "reduced":
        if row.get("reduced_density") is not None:
            return float(row["reduced_density"])
        r_u, r_v, r_e = row.get("reduced_nU"), row.get("reduced_nV"), row.get("reduced_edges")
        if r_u is not None and r_v is not None and r_e is not None:
            denom = int(r_u) * int(r_v)
            return float(r_e) / denom if denom > 0 else 0.0
    nu, nv, edges = row.get("nU"), row.get("nV"), row.get("edge_count")
    if nu is not None and nv is not None and edges is not None:
        denom = int(nu) * int(nv)
        return float(edges) / denom if denom > 0 else 0.0
    r_u, r_v, r_e = row.get("reduced_nU"), row.get("reduced_nV"), row.get("reduced_edges")
    if r_u is not None and r_v is not None and r_e is not None:
        denom = int(r_u) * int(r_v)
        return float(r_e) / denom if denom > 0 else 0.0
    return None

def get_heat_stats(row):
    """Extract normalized heat statistics (biclique, total, sampled)."""
    heat = row.get("heat")
    if not heat:
        heat = row.get("challenger", {}).get("meta", {}).get("heat_stats")
    if not heat:
        heat = row.get("baseline", {}).get("meta", {}).get("heat_stats")
    if not heat or not isinstance(heat, dict):
        return None

    b_mean = heat.get("biclique_mean")
    if b_mean is None:
        b_mean = heat.get("biclique_heat_mean")
    t_mean = heat.get("total_graph_mean")
    if t_mean is None:
        t_mean = heat.get("full_graph_mean")
    s_mean = heat.get("sampled_graph_mean")

    if b_mean is None or t_mean is None:
        return None

    pct_change = heat.get("percent_change")
    if pct_change is None:
        pct_change = ((float(b_mean) - float(t_mean)) / float(t_mean) * 100.0) if float(t_mean) != 0 else 0.0

    diff = heat.get("difference")
    if diff is None:
        diff = float(b_mean) - float(t_mean)

    return {
        "biclique_mean": float(b_mean),
        "total_graph_mean": float(t_mean),
        "sampled_graph_mean": float(s_mean) if s_mean is not None else None,
        "percent_change": float(pct_change),
        "difference": float(diff),
        "raw": heat,
    }

def pearson_correlation(xs, ys):
    """Compute Pearson correlation coefficient r, R^2, slope, and intercept."""
    if len(xs) != len(ys) or len(xs) < 2:
        return None, None, None, None
    n = len(xs)
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    var_x = sum((x - mean_x) ** 2 for x in xs)
    var_y = sum((y - mean_y) ** 2 for y in ys)
    if var_x == 0 or var_y == 0:
        return 0.0, 0.0, 0.0, mean_y
    cov_xy = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    r = cov_xy / ((var_x * var_y) ** 0.5)
    r2 = r ** 2
    slope = cov_xy / var_x
    intercept = mean_y - slope * mean_x
    return r, r2, slope, intercept

# ── JSON loading ──────────────────────────────────────────────────────────────
def load_json(path):
    with open(path) as f: return json.load(f)

def get_results(data):
    if "results" in data and isinstance(data["results"], list):
        return data["results"]
    if "dataset" in data and "verdict" in data:
        return [data]
    return []

# ── Core analysis ─────────────────────────────────────────────────────────────
def analyze(data, results, density_type="reduced"):
    ok = [r for r in results if r.get("status", "ok") == "ok"]

    chal_wins   = sum(1 for r in ok if str(r.get("verdict","")) == "challenger")
    base_wins   = sum(1 for r in ok if str(r.get("verdict","")) == "baseline")
    ties        = sum(1 for r in ok if str(r.get("verdict","")) == "tie")
    paper_beats = sum(1 for r in ok if r.get("challenger_beats_baseline") is True)

    pcts_wins = [p for r in ok if str(r.get("verdict","")) == "challenger"
                 for p in [safe_pct(r.get("edge_pct_increase"))] if p is not None]
    all_pcts  = [p for r in ok for p in [safe_pct(r.get("edge_pct_increase"))] if p is not None]

    chal_feasible = sum(1 for r in ok if r.get("challenger",{}).get("theta_feasible") is True)
    base_feasible = sum(1 for r in ok if r.get("baseline",{}).get("theta_feasible") is True)

    chal_times = [r["challenger_wall_time_s"] for r in ok if r.get("challenger_wall_time_s") is not None]
    base_times = [r["baseline_wall_time_s"]   for r in ok if r.get("baseline_wall_time_s") is not None]
    chal_edges = [e for r in ok for e in [r.get("challenger",{}).get("final_edges")] if e is not None]
    base_edges = [e for r in ok for e in [r.get("baseline",{}).get("final_edges")]   if e is not None]

    heat_points = []
    for r in ok:
        dens = get_graph_density(r, density_type)
        h = get_heat_stats(r)
        if dens is not None and h is not None:
            heat_points.append({
                "dataset": str(r.get("dataset", "unknown")),
                "density": dens,
                "biclique_mean": h["biclique_mean"],
                "total_graph_mean": h["total_graph_mean"],
                "sampled_graph_mean": h["sampled_graph_mean"],
                "percent_change": h["percent_change"],
                "difference": h["difference"],
            })

    densities = [p["density"] for p in heat_points]
    pct_changes = [p["percent_change"] for p in heat_points]
    diffs = [p["difference"] for p in heat_points]

    heat_corr_pct = pearson_correlation(densities, pct_changes)
    heat_corr_diff = pearson_correlation(densities, diffs)

    n = len(ok)
    return dict(
        n_total=len(results), n_ok=n, n_errors=len(results)-n,
        chal_wins=chal_wins, base_wins=base_wins, ties=ties, paper_beats=paper_beats,
        chal_win_rate=100.0*chal_wins/n if n else 0.0,
        base_win_rate=100.0*base_wins/n if n else 0.0,
        tie_rate=100.0*ties/n if n else 0.0,
        mean_pct_all=statistics.mean(all_pcts) if all_pcts else None,
        median_pct_all=median(all_pcts),
        mean_pct_wins=statistics.mean(pcts_wins) if pcts_wins else None,
        median_pct_wins=median(pcts_wins),
        pct_stdev=statistics.stdev(all_pcts) if len(all_pcts) > 1 else None,
        chal_feasible=chal_feasible, base_feasible=base_feasible,
        chal_feasible_rate=100.0*chal_feasible/n if n else 0.0,
        base_feasible_rate=100.0*base_feasible/n if n else 0.0,
        mean_chal_time=statistics.mean(chal_times) if chal_times else None,
        mean_base_time=statistics.mean(base_times) if base_times else None,
        median_chal_time=median(chal_times), median_base_time=median(base_times),
        mean_chal_edges=statistics.mean(chal_edges) if chal_edges else None,
        mean_base_edges=statistics.mean(base_edges) if base_edges else None,
        heat_points=heat_points,
        heat_corr_pct=heat_corr_pct,
        heat_corr_diff=heat_corr_diff,
        density_type=density_type,
        ok_rows=ok,
    )

# ── Report ────────────────────────────────────────────────────────────────────
def get_label(data, side, fallback):
    for src in [data.get(side) or {}, data.get(f"{side}_spec") or {}]:
        if src.get("label"): return src["label"]
    return fallback

def print_report(path, data, results, stats, *, sort_by="dataset", top_n=None, filter_verdict=None):
    chal_label = get_label(data, "challenger", "challenger")
    base_label  = get_label(data, "baseline",   "baseline")
    n = stats["n_ok"]

    print(bold("=" * 70))
    print(bold(f"  compare-methods analysis: {os.path.basename(path)}"))
    print(bold("=" * 70))

    meta = [f"{k}={data[k]}" for k in ("name","k","theta","seed","reduction","prefix")
            if k in data and data[k] is not None]
    if meta: print(dim("  " + "  ".join(meta)))
    print(f"\n  {cyan('baseline')}   : {bold(base_label)}")
    print(f"  {cyan('challenger')} : {bold(chal_label)}")
    err_s = f"  {red(str(stats['n_errors']) + ' errors')}" if stats["n_errors"] else ""
    print(f"  graphs       : {n} ok{err_s}")

    # Win/Loss
    print()
    print(bold("── Win / Loss Summary ─────────────────────────────────────"))
    rf = "  {:<34} {:>6}  {:>7}"
    print(dim(rf.format("Outcome", "Count", "Rate")))
    print(dim("  " + "-" * 50))
    print(rf.format(green(f"Challenger ({chal_label}) wins"), green(str(stats["chal_wins"])), green(f"{stats['chal_win_rate']:.1f}%")))
    print(rf.format(red(f"Baseline ({base_label}) wins"),    red(str(stats["base_wins"])),   red(f"{stats['base_win_rate']:.1f}%")))
    print(rf.format("Ties",                                   str(stats["ties"]),             f"{stats['tie_rate']:.1f}%"))
    print(rf.format("Paper beats (strict θ-feasible+edges)", str(stats["paper_beats"]),
                    f"{100.0*stats['paper_beats']/n:.1f}%" if n else "n/a"))

    # Edge quality
    print()
    print(bold("── Edge Quality (challenger vs baseline) ──────────────────"))
    qf = "  {:<44} {:>12}"
    print(qf.format("Mean   edge % increase (all graphs):",          pct_str(stats["mean_pct_all"])))
    print(qf.format("Median edge % increase (all graphs):",          pct_str(stats["median_pct_all"])))
    print(qf.format("Mean   edge % increase (challenger-win graphs):",pct_str(stats["mean_pct_wins"])))
    print(qf.format("Median edge % increase (challenger-win graphs):",pct_str(stats["median_pct_wins"])))
    if stats["pct_stdev"] is not None:
        print(qf.format("Std-dev edge % (all graphs):", f"{stats['pct_stdev']:.2f}%"))
    print(qf.format("Mean challenger edges:", f"{stats['mean_chal_edges']:.1f}" if stats["mean_chal_edges"] is not None else "n/a"))
    print(qf.format("Mean baseline edges:",   f"{stats['mean_base_edges']:.1f}"  if stats["mean_base_edges"]  is not None else "n/a"))

    # Feasibility
    print()
    print(bold("── θ-Feasibility ──────────────────────────────────────────"))
    ff = "  {:<44} {:>6}  {:>7}"
    print(ff.format(f"Challenger ({chal_label}) feasible:", str(stats["chal_feasible"]), f"{stats['chal_feasible_rate']:.1f}%"))
    print(ff.format(f"Baseline ({base_label}) feasible:",   str(stats["base_feasible"]), f"{stats['base_feasible_rate']:.1f}%"))

    # Runtime
    print()
    print(bold("── Wall-clock Runtime ─────────────────────────────────────"))
    tf = "  {:<44} {:>12} {:>12}"
    print(dim(tf.format("", "mean", "median")))
    print(tf.format(f"Challenger ({chal_label}):", fmt_time(stats["mean_chal_time"]), fmt_time(stats["median_chal_time"])))
    print(tf.format(f"Baseline ({base_label}):",   fmt_time(stats["mean_base_time"]), fmt_time(stats["median_base_time"])))

    # Heat analysis if present
    hp = stats.get("heat_points", [])
    if hp:
        print()
        print(bold("── Heat Analysis (Biclique vs Total Graph Heat) ───────────"))
        hf = "  {:<44} {:>12}"
        print(hf.format("Graphs with heat statistics:", str(len(hp))))
        b_means = [p["biclique_mean"] for p in hp]
        t_means = [p["total_graph_mean"] for p in hp]
        pcts = [p["percent_change"] for p in hp]
        diffs = [p["difference"] for p in hp]
        print(hf.format("Mean   biclique heat mean:", f"{statistics.mean(b_means):.4f}"))
        print(hf.format("Mean   total graph heat mean:", f"{statistics.mean(t_means):.4f}"))
        print(hf.format("Mean   heat % change (vs total):", pct_str(statistics.mean(pcts))))
        print(hf.format("Median heat % change (vs total):", pct_str(median(pcts))))
        print(hf.format("Mean   heat difference (vs total):", f"{statistics.mean(diffs):+.4f}"))

        r_pct, r2_pct, slope_pct, intercept_pct = stats["heat_corr_pct"]
        if r_pct is not None:
            sign = "+" if r_pct >= 0 else ""
            print(hf.format(f"Pearson r ({stats['density_type']} density vs %Δ):", f"{sign}{r_pct:.4f}"))
            print(hf.format("R² coefficient of determination:", f"{r2_pct:.4f}"))
            if slope_pct is not None:
                s_sign = "+" if intercept_pct >= 0 else "-"
                print(hf.format("Linear regression fit:", f"y={slope_pct:.2f}x{s_sign}{abs(intercept_pct):.2f}"))
        r_diff, r2_diff, _, _ = stats["heat_corr_diff"]
        if r_diff is not None:
            sign = "+" if r_diff >= 0 else ""
            print(hf.format(f"Pearson r ({stats['density_type']} density vs diff):", f"{sign}{r_diff:.4f}"))

    # Per-graph table
    print()
    print(bold("── Per-graph Results ──────────────────────────────────────"))
    rows = list(stats["ok_rows"])
    if filter_verdict:
        rows = [r for r in rows if str(r.get("verdict","")) == filter_verdict]
        if not rows:
            print(f"  (no rows with verdict={filter_verdict!r})")
            print(bold("=" * 70))
            return

    def sort_key(r):
        if sort_by == "edges":  return r.get("edge_count", 0)
        if sort_by == "pct":    p = safe_pct(r.get("edge_pct_increase")); return -p if p is not None else float("inf")
        if sort_by == "delta":  return -(r.get("edge_delta") or 0)
        if sort_by == "verdict": return {"challenger":0,"baseline":1,"tie":2}.get(str(r.get("verdict","")),3)
        return str(r.get("dataset",""))

    rows.sort(key=sort_key)
    if top_n is not None: rows = rows[:top_n]

    ds_w = max((len(str(r.get("dataset",""))) for r in rows), default=7)
    ds_w = max(ds_w, 7)
    has_heat = any(get_heat_stats(r) is not None for r in rows)

    if has_heat:
        hdr = f"  {'Dataset':<{ds_w}}  {'|E|':>7}  {'Dens':>7}  {'B-edges':>7}  {'C-edges':>7}  {'Δ%':>8}  {'verdict':>10}  {'Heat-B':>7}  {'Heat-Tot':>8}  {'HeatΔ%':>8}"
    else:
        hdr = f"  {'Dataset':<{ds_w}}  {'|E|':>7}  {'B-edges':>7}  {'C-edges':>7}  {'Δedges':>7}  {'Δ%':>8}  {'verdict':>10}  {'B-feas':>6}  {'C-feas':>6}"
    print(dim(hdr))
    print(dim("  " + "-" * (len(hdr) - 2)))

    for r in rows:
        ds    = str(r.get("dataset","?"))
        edges = r.get("edge_count","?")
        be    = r.get("baseline",{}).get("final_edges","?")
        ce    = r.get("challenger",{}).get("final_edges","?")
        delta = r.get("edge_delta","?")
        pct   = safe_pct(r.get("edge_pct_increase"))
        verd  = str(r.get("verdict","?"))
        pct_s = pct_str(pct) if pct is not None else "n/a"
        delta_s = f"+{delta}" if isinstance(delta, int) and delta > 0 else str(delta)
        if verd == "challenger": verd_s = green(f"{'challenger':>10}")
        elif verd == "baseline": verd_s = red(f"{'baseline':>10}")
        else:                    verd_s = yellow(f"{'tie':>10}")

        if has_heat:
            dens = get_graph_density(r, stats["density_type"])
            dens_s = f"{dens:.4f}" if dens is not None else "n/a"
            h = get_heat_stats(r)
            if h:
                hb_s = f"{h['biclique_mean']:.2f}"
                ht_s = f"{h['total_graph_mean']:.2f}"
                hp_s = pct_str(h["percent_change"])
            else:
                hb_s, ht_s, hp_s = "n/a", "n/a", "n/a"
            print(f"  {ds:<{ds_w}}  {str(edges):>7}  {dens_s:>7}  {str(be):>7}  {str(ce):>7}  {pct_s:>8}  {verd_s}  {hb_s:>7}  {ht_s:>8}  {hp_s:>8}")
        else:
            bf = "✓" if r.get("baseline",{}).get("theta_feasible") else "✗"
            cf = "✓" if r.get("challenger",{}).get("theta_feasible") else "✗"
            print(f"  {ds:<{ds_w}}  {str(edges):>7}  {str(be):>7}  {str(ce):>7}  {delta_s:>7}  {pct_s:>8}  {verd_s}    {bf:>4}    {cf:>4}")

    print(bold("=" * 70))

# ── CSV output ────────────────────────────────────────────────────────────────
def print_csv(results, density_type="reduced"):
    out = io.StringIO()
    w = csv.writer(out)
    w.writerow(["dataset","edge_count","reduced_nU","reduced_nV","reduced_edges","density",
                "baseline_edges","baseline_feasible","baseline_time_s",
                "challenger_edges","challenger_feasible","challenger_time_s",
                "edge_delta","edge_pct_increase","verdict","challenger_beats_baseline",
                "biclique_heat_mean","total_graph_mean","sampled_graph_mean","heat_diff","heat_pct_change"])
    for r in results:
        if r.get("status","ok") != "ok": continue
        b = r.get("baseline",{}); c = r.get("challenger",{})
        dens = get_graph_density(r, density_type)
        h = get_heat_stats(r)
        w.writerow([r.get("dataset"), r.get("edge_count"), r.get("reduced_nU"), r.get("reduced_nV"),
                    r.get("reduced_edges"), f"{dens:.6f}" if dens is not None else "",
                    b.get("final_edges"), b.get("theta_feasible"), r.get("baseline_wall_time_s"),
                    c.get("final_edges"), c.get("theta_feasible"), r.get("challenger_wall_time_s"),
                    r.get("edge_delta"), r.get("edge_pct_increase"), r.get("verdict"),
                    r.get("challenger_beats_baseline"),
                    f"{h['biclique_mean']:.4f}" if h else "",
                    f"{h['total_graph_mean']:.4f}" if h else "",
                    f"{h['sampled_graph_mean']:.4f}" if (h and h['sampled_graph_mean'] is not None) else "",
                    f"{h['difference']:.4f}" if h else "",
                    f"{h['percent_change']:.4f}" if h else ""])
    print(out.getvalue(), end="")

# ── Plotting ──────────────────────────────────────────────────────────────────
def plot_heat_density(stats, output_path, density_type="reduced", title=None):
    """Plot graph density vs percent change in heat mean using matplotlib."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("Error: matplotlib is required for plotting.", file=sys.stderr)
        print("Please activate your virtual environment with matplotlib installed: pip install matplotlib", file=sys.stderr)
        sys.exit(1)

    points = stats.get("heat_points", [])
    if not points:
        print("Error: no heat and density data points found in JSON to plot.", file=sys.stderr)
        print("Make sure compare-methods.jl was run with heat recording enabled.", file=sys.stderr)
        return

    densities = [p["density"] for p in points]
    pct_changes = [p["percent_change"] for p in points]
    labels = [p["dataset"].split("/")[-1] for p in points]

    r, r2, slope, intercept = stats.get("heat_corr_pct", (None, None, None, None))

    fig, ax = plt.subplots(figsize=(9, 6), dpi=150)
    ax.scatter(densities, pct_changes, color="#1976d2", edgecolor="#0d47a1", s=65, alpha=0.85, zorder=3, label=f"Observations (N={len(points)})")

    # Annotate points with dataset basenames
    for x, y, lab in zip(densities, pct_changes, labels):
        ax.annotate(lab, (x, y), xytext=(5, 4), textcoords="offset points", fontsize=8, alpha=0.8)

    # Linear trendline if valid
    if slope is not None and len(densities) >= 2:
        x_min, x_max = min(densities), max(densities)
        x_span = x_max - x_min
        x_pad = x_span * 0.05 if x_span > 0 else 0.01
        x_vals = [max(0.0, x_min - x_pad), x_max + x_pad]
        y_vals = [slope * x + intercept for x in x_vals]
        sign = "+" if intercept >= 0 else "-"
        fit_label = f"Fit: y = {slope:.2f}x {sign} {abs(intercept):.2f}\n(r = {r:+.3f}, R² = {r2:.3f})"
        ax.plot(x_vals, y_vals, color="#d32f2f", linestyle="--", linewidth=1.8, zorder=2, label=fit_label)

    density_label = "Reduced Graph Density $|E_R| / (|U_R| \\cdot |V_R|)$" if density_type == "reduced" else "Original Graph Density $|E| / (|U| \\cdot |V|)$"
    ax.set_xlabel(density_label, fontsize=11, fontweight="bold")
    ax.set_ylabel("Heat Mean % Change: (Biclique - Total) / Total (%)", fontsize=11, fontweight="bold")

    if title:
        plot_title = title
    elif r is not None:
        plot_title = f"Correlation: Graph Density vs Biclique Heat % Change (r = {r:+.3f})"
    else:
        plot_title = "Graph Density vs Biclique Heat % Change"
    ax.set_title(plot_title, fontsize=13, fontweight="bold", pad=12)

    ax.axhline(0, color="gray", linestyle=":", linewidth=0.8, alpha=0.7)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="best", framealpha=0.9, fontsize=9.5)
    plt.tight_layout()

    out_dir = os.path.dirname(os.path.abspath(output_path))
    if out_dir: os.makedirs(out_dir, exist_ok=True)
    plt.savefig(output_path)
    plt.close(fig)
    print(f"Saved correlation plot to: {output_path}")

# ── Entry point ───────────────────────────────────────────────────────────────
def main():
    global USE_COLOR
    args = sys.argv[1:]
    if not args or "-h" in args or "--help" in args:
        print(__doc__); sys.exit(0)

    path = None; sort_by = "dataset"; top_n = None; filter_verdict = None
    emit_csv = False; do_plot = False; plot_path = None
    density_type = "reduced"; plot_title = None

    for a in args:
        if   a.startswith("--sort="):         sort_by = a.split("=",1)[1]
        elif a.startswith("--top="):          top_n = int(a.split("=",1)[1])
        elif a.startswith("--filter="):       filter_verdict = a.split("=",1)[1]
        elif a == "--csv":                    emit_csv = True
        elif a == "--no-color":               USE_COLOR = False
        elif a.startswith("--density-type="): density_type = a.split("=",1)[1]
        elif a.startswith("--title="):        plot_title = a.split("=",1)[1]
        elif a.startswith("--plot="):
            do_plot = True
            plot_path = a.split("=",1)[1]
        elif a in ("--plot", "--plot-heat"):
            do_plot = True
        elif not a.startswith("-"):
            path = a

    if path is None:
        print("Error: no JSON path provided.\n", file=sys.stderr); print(__doc__, file=sys.stderr); sys.exit(1)
    if not os.path.isfile(path):
        print(f"Error: file not found: {path}", file=sys.stderr); sys.exit(1)

    data = load_json(path)
    results = get_results(data)
    if not results:
        print("No results found in JSON.", file=sys.stderr); sys.exit(1)

    if emit_csv:
        print_csv(results, density_type=density_type); return

    stats = analyze(data, results, density_type=density_type)
    print_report(path, data, results, stats, sort_by=sort_by, top_n=top_n, filter_verdict=filter_verdict)

    if do_plot:
        if not plot_path:
            base_stem = os.path.splitext(os.path.basename(path))[0]
            plot_path = f"{base_stem}_heat_density.png"
        plot_heat_density(stats, plot_path, density_type=density_type, title=plot_title)

if __name__ == "__main__":
    main()
