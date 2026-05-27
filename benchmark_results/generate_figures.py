"""
generate_figures.py
-------------------
Generates all benchmark visualisation figures for the Zenodo dataset:
  Benchmark Solver Results for Workflow Scheduling on Heterogeneous HPC Systems

Design choices:
  - Okabe-Ito colorblind-safe palette (works for deuteranopia, protanopia, tritanopia)
  - Every filled element also carries a distinct hatch pattern -> readable in greyscale print
  - Every line/marker combination uses a distinct shape + linestyle -> readable in B&W
  - Gap reference: MILP (PuLP) optimal solutions only

Figures produced (saved to ./figures/):
  fig01_boxplot_objective_{mode}.pdf/png     -- objective value boxplots
  fig02_boxplot_makespan_{mode}.pdf/png      -- makespan boxplots
  fig03_scalability_{mode}.pdf/png           -- median objective vs graph size
  fig04_walltime_{mode}.pdf/png              -- wall-clock time (log scale)
  fig05_gap_to_pulp_{mode}.pdf/png           -- relative gap vs MILP (PuLP) optimal
  fig06_scatter_rnc{N}_{mode}.pdf/png        -- per-instance makespan (sorted scatter)
  fig07_node_util_rnc{N}_hetero.pdf/png      -- node utilisation heatmap
  fig08_solve_status.pdf/png                 -- solve-status breakdown
  fig09_overview.pdf/png                     -- combined summary dashboard

Dependencies: pandas, matplotlib, numpy
Run: python3 generate_figures.py
"""

import json
import glob
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
HERE   = os.path.dirname(os.path.abspath(__file__))
BASE   = os.path.join(os.path.dirname(HERE), "main_results")
FIGDIR = os.path.join(HERE, "figures")
os.makedirs(FIGDIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Style  -- Okabe-Ito colorblind-safe palette (2008)
# 6 entries chosen from the 8-color Okabe-Ito set for maximum separation.
# Gray values (0.299R+0.587G+0.114B) are sufficiently spread so hatches
# provide a second distinguishing dimension for greyscale / mono print.
# ---------------------------------------------------------------------------
SOLVERS = ["milp_ortools", "milp_gurobi", "cpsat", "heft", "gnnrl_teacher", "gnnrl_self"]

SOLVER_LABELS = {
    "milp_ortools":  "MILP\n(PuLP)",
    "milp_gurobi":   "MILP\n(Gurobi)",
    "cpsat":         "CP-SAT",
    "heft":          "HEFT",
    "gnnrl_teacher": "GNNRL\n(Teacher)",
    "gnnrl_self":    "GNNRL\n(Self)",
}
SOLVER_SHORT = {
    "milp_ortools":  "PuLP",
    "milp_gurobi":   "Gurobi",
    "cpsat":         "CP-SAT",
    "heft":          "HEFT",
    "gnnrl_teacher": "GNNRL-T",
    "gnnrl_self":    "GNNRL-S",
}

# Okabe-Ito palette -- approx greyscale: 87, 119, 106, 162, 151, 158
# Hatches make the three light colors distinguishable in mono print.
COLORS = {
    "milp_ortools":  "#0072B2",   # blue         grey ~87
    "milp_gurobi":   "#D55E00",   # vermilion    grey ~119
    "cpsat":         "#009E73",   # bluish green grey ~106
    "heft":          "#E69F00",   # golden       grey ~162
    "gnnrl_teacher": "#CC79A7",   # rose         grey ~151
    "gnnrl_self":    "#56B4E9",   # sky blue     grey ~158
}

# Hatch patterns -- each unique, clearly different in print
HATCHES = {
    "milp_ortools":  "///",
    "milp_gurobi":   "xxx",
    "cpsat":         "...",
    "heft":          "|||",
    "gnnrl_teacher": "\\\\",
    "gnnrl_self":    "---",
}

# Markers -- distinct shapes for scatter and line plots
MARKERS = {
    "milp_ortools":  "o",    # circle
    "milp_gurobi":   "s",    # square
    "cpsat":         "^",    # triangle-up
    "heft":          "D",    # diamond
    "gnnrl_teacher": "P",    # filled plus
    "gnnrl_self":    "*",    # star
}

# Line styles -- distinct dashes for line plots
LINESTYLES = {
    "milp_ortools":  "-",
    "milp_gurobi":   "--",
    "cpsat":         ":",
    "heft":          "-.",
    "gnnrl_teacher": (0, (5, 1)),
    "gnnrl_self":    (0, (3, 1, 1, 1)),
}

SIZES       = ["rnc50", "rnc100", "rnc300"]
MODES       = ["homo", "hetero"]
SIZE_LABELS = {
    "rnc50":  "RNC-50\n(50 tasks)",
    "rnc100": "RNC-100\n(100 tasks)",
    "rnc300": "RNC-300\n(300 tasks)",
}
DPI   = 150
FIG_W = 10.5

plt.rcParams.update({
    "font.family":     "DejaVu Sans",
    "axes.linewidth":  0.8,
    "grid.linewidth":  0.5,
    "grid.alpha":      0.38,
    "grid.linestyle":  "--",
})

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_record(path, solver):
    try:
        with open(path) as f:
            d = json.load(f)
    except Exception:
        return None

    fname  = os.path.basename(path)
    parts  = fname.replace(".json", "").split("_")
    # variation = "rand0042" (just the number part, without mode suffix)
    variation = parts[0]

    rec = {"variation": variation, "solver": solver}

    if "objective" in d and isinstance(d["objective"], dict):
        rec["objective"]   = d["objective"].get("value")
        rec["makespan"]    = d["objective"].get("makespan")
        rec["usage_term"]  = d["objective"].get("usage_term")
        rec["wall_time_s"] = d.get("solver", {}).get("wall_time_s")
        rec["status"]      = d.get("status", "unknown")
        rec["node_util"]   = d.get("node_utilization", {})
    else:
        rec["objective"]   = d.get("objective")
        rec["makespan"]    = d.get("makespan")
        rec["usage_term"]  = d.get("usage_term")
        rec["wall_time_s"] = d.get("wall_time_s")
        rec["status"]      = "solved"
        rec["node_util"]   = {}

    return rec


def load_all():
    rows = []
    for solver in SOLVERS:
        for size in SIZES:
            for mode in MODES:
                cell_dir = os.path.join(BASE, solver, f"{size}_{mode}")
                if not os.path.isdir(cell_dir):
                    continue
                for path in glob.glob(os.path.join(cell_dir, "*.json")):
                    rec = load_record(path, solver)
                    if rec is None:
                        continue
                    rec["size"] = size
                    rec["mode"] = mode
                    rows.append(rec)

    df = pd.DataFrame(rows)
    df = df[df["objective"].notna() & (df["objective"] > 0)].copy()
    df = df[df["status"] != "infeasible"].copy()
    df["solver_short"] = df["solver"].map(SOLVER_SHORT)
    return df


print("Loading data ...")
df = load_all()
print(f"  {len(df):,} records across {df['solver'].nunique()} solvers")

# ---------------------------------------------------------------------------
# PuLP reference table (best objective per variation/size/mode, optimal only)
# ---------------------------------------------------------------------------
pulp_optimal = (
    df[(df["solver"] == "milp_ortools") & (df["status"] == "optimal")]
    .groupby(["size", "mode", "variation"])["objective"]
    .min()
    .reset_index()
    .rename(columns={"objective": "pulp_ref"})
)
print(f"  PuLP optimal reference rows: {len(pulp_optimal)}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def savefig(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIGDIR, f"{name}.{ext}"),
                    dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {name}")


def _style_ax(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y")


def _apply_box_style(bp, solvers_in_order):
    """Apply color + hatch + black edge to every box patch."""
    for patch, solver in zip(bp["boxes"], solvers_in_order):
        patch.set_facecolor(COLORS[solver])
        patch.set_hatch(HATCHES[solver])
        patch.set_edgecolor("black")
        patch.set_linewidth(1.1)
        patch.set_alpha(0.88)
    for med in bp["medians"]:
        med.set_color("white")
        med.set_linewidth(2.2)
    for elem in bp["whiskers"] + bp["caps"]:
        elem.set_color("black")
        elem.set_linewidth(0.9)
    for flier in bp["fliers"]:
        flier.set_marker("o")
        flier.set_markersize(2.2)
        flier.set_alpha(0.45)
        flier.set_markerfacecolor("gray")
        flier.set_markeredgecolor("gray")


def solver_legend_handles(solvers=None):
    if solvers is None:
        solvers = SOLVERS
    return [
        Patch(facecolor=COLORS[s], hatch=HATCHES[s], edgecolor="black",
              label=SOLVER_SHORT[s], alpha=0.88)
        for s in solvers
    ]


def line_legend_handles(solvers=None):
    if solvers is None:
        solvers = SOLVERS
    return [
        Line2D([0], [0],
               color=COLORS[s],
               marker=MARKERS[s],
               linestyle=LINESTYLES[s],
               linewidth=1.8,
               markersize=6,
               label=SOLVER_SHORT[s])
        for s in solvers
    ]


# ---------------------------------------------------------------------------
# Fig 01 / 02 -- Boxplots: objective and makespan
# ---------------------------------------------------------------------------
def make_boxplots(metric, ylabel, fig_prefix):
    for mode in MODES:
        sub = df[df["mode"] == mode]
        fig, axes = plt.subplots(1, 3, figsize=(FIG_W * 1.05, 4.2), sharey=False)
        fig.suptitle(
            f"{ylabel} Distribution by Solver  |  {mode.capitalize()} mode",
            fontsize=11, fontweight="bold", y=1.02,
        )

        for ax, size in zip(axes, SIZES):
            cell = sub[sub["size"] == size]
            data = [cell[cell["solver"] == s][metric].dropna().values for s in SOLVERS]

            bp = ax.boxplot(
                data, patch_artist=True,
                medianprops=dict(color="white", linewidth=2.2),
                widths=0.58,
            )
            _apply_box_style(bp, SOLVERS)
            ax.set_xticks(range(1, len(SOLVERS) + 1))
            ax.set_xticklabels([SOLVER_SHORT[s] for s in SOLVERS], fontsize=7.2,
                                rotation=30, ha="right")
            ax.set_title(SIZE_LABELS[size], fontsize=9)
            ax.set_ylabel(ylabel, fontsize=8)
            ax.tick_params(axis="y", labelsize=7.5)
            _style_ax(ax)

        fig.legend(handles=solver_legend_handles(), loc="lower center",
                   ncol=6, fontsize=7.5, framealpha=0.9,
                   bbox_to_anchor=(0.5, -0.07))
        fig.tight_layout()
        savefig(fig, f"{fig_prefix}_{mode}")


print("\nFig 01 -- Objective boxplots ...")
make_boxplots("objective", "Objective value", "fig01_boxplot_objective")
print("Fig 02 -- Makespan boxplots ...")
make_boxplots("makespan", "Makespan (s)", "fig02_boxplot_makespan")


# ---------------------------------------------------------------------------
# Fig 03 -- Scalability: median objective vs graph size
# ---------------------------------------------------------------------------
def make_scalability():
    xs = [50, 100, 300]
    for mode in MODES:
        sub = df[df["mode"] == mode]
        fig, ax = plt.subplots(figsize=(6.5, 4.2))

        for solver in SOLVERS:
            ys, yl, yu = [], [], []
            for size in SIZES:
                vals = sub[(sub["solver"] == solver) & (sub["size"] == size)
                           ]["objective"].dropna()
                if len(vals) == 0:
                    ys.append(np.nan); yl.append(0); yu.append(0)
                else:
                    med = vals.median()
                    ys.append(med)
                    yl.append(med - vals.quantile(0.25))
                    yu.append(vals.quantile(0.75) - med)

            ax.errorbar(
                xs, ys, yerr=[yl, yu],
                color=COLORS[solver],
                marker=MARKERS[solver],
                linestyle=LINESTYLES[solver],
                linewidth=1.8, markersize=7, capsize=3.5,
                label=SOLVER_SHORT[solver],
            )

        ax.set_xlabel("Number of tasks", fontsize=9)
        ax.set_ylabel("Median objective value  (IQR bars)", fontsize=8.5)
        ax.set_title(f"Scalability: Objective vs Graph Size  |  {mode.capitalize()} mode",
                     fontsize=10, fontweight="bold")
        ax.set_xticks(xs)
        ax.set_xticklabels(["50\n(RNC-50)", "100\n(RNC-100)", "300\n(RNC-300)"], fontsize=8)
        ax.tick_params(axis="y", labelsize=7.5)
        ax.legend(handles=line_legend_handles(), fontsize=7.5, ncol=2,
                  framealpha=0.88)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid()
        fig.tight_layout()
        savefig(fig, f"fig03_scalability_{mode}")

print("Fig 03 -- Scalability plots ...")
make_scalability()


# ---------------------------------------------------------------------------
# Fig 04 -- Wall-clock time (log scale)
# ---------------------------------------------------------------------------
def make_walltime():
    for mode in MODES:
        sub = df[df["mode"] == mode]
        fig, axes = plt.subplots(1, 3, figsize=(FIG_W * 1.05, 4.2), sharey=False)
        fig.suptitle(
            f"Solver Wall-Clock Time (log scale)  |  {mode.capitalize()} mode",
            fontsize=11, fontweight="bold", y=1.02,
        )

        for ax, size in zip(axes, SIZES):
            cell = sub[sub["size"] == size]
            data = [cell[cell["solver"] == s]["wall_time_s"].dropna().values for s in SOLVERS]
            bp = ax.boxplot(data, patch_artist=True,
                            medianprops=dict(color="white", linewidth=2.2),
                            widths=0.58)
            _apply_box_style(bp, SOLVERS)
            ax.set_yscale("log")
            ax.set_xticks(range(1, len(SOLVERS) + 1))
            ax.set_xticklabels([SOLVER_SHORT[s] for s in SOLVERS], fontsize=7.2,
                                rotation=30, ha="right")
            ax.set_title(SIZE_LABELS[size], fontsize=9)
            ax.set_ylabel("Wall time (s, log scale)", fontsize=8)
            ax.tick_params(axis="y", labelsize=7.5)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.grid(axis="y", which="both")

        fig.legend(handles=solver_legend_handles(), loc="lower center",
                   ncol=6, fontsize=7.5, framealpha=0.9,
                   bbox_to_anchor=(0.5, -0.07))
        fig.tight_layout()
        savefig(fig, f"fig04_walltime_{mode}")

print("Fig 04 -- Wall-time boxplots ...")
make_walltime()


# ---------------------------------------------------------------------------
# Fig 05 -- Gap to MILP (PuLP) optimal reference
# ---------------------------------------------------------------------------
def make_gap():
    other_solvers = [s for s in SOLVERS if s != "milp_ortools"]

    for mode in MODES:
        sub = df[df["mode"] == mode].copy()
        # Take best (min) objective per variation per solver to handle duplicate runs
        agg = (sub.groupby(["solver", "size", "variation"])["objective"]
                  .min().reset_index())
        ref  = pulp_optimal[pulp_optimal["mode"] == mode]
        merged = agg[agg["solver"] != "milp_ortools"].merge(
            ref[["size", "variation", "pulp_ref"]],
            on=["size", "variation"], how="inner"
        )
        if merged.empty:
            print(f"  WARNING: no overlap for gap ({mode})")
            continue

        merged["gap_pct"] = (
            (merged["objective"] - merged["pulp_ref"]) / merged["pulp_ref"] * 100.0
        )

        fig, axes = plt.subplots(1, 3, figsize=(FIG_W * 1.05, 4.2), sharey=False)
        fig.suptitle(
            f"Relative Gap to MILP (PuLP) Optimal  |  {mode.capitalize()} mode",
            fontsize=11, fontweight="bold", y=1.02,
        )

        for ax, size in zip(axes, SIZES):
            cell = merged[merged["size"] == size]
            n_ref = len(ref[ref["size"] == size])
            data  = [cell[cell["solver"] == s]["gap_pct"].dropna().values
                     for s in other_solvers]

            bp = ax.boxplot(data, patch_artist=True,
                            medianprops=dict(color="white", linewidth=2.2),
                            widths=0.58)
            _apply_box_style(bp, other_solvers)
            ax.axhline(0, color="black", linestyle="--", linewidth=1.1,
                       alpha=0.7, zorder=0, label="MILP (PuLP) reference (0%)")
            ax.set_xticks(range(1, len(other_solvers) + 1))
            ax.set_xticklabels([SOLVER_SHORT[s] for s in other_solvers],
                                fontsize=7.2, rotation=30, ha="right")
            ax.set_title(f"{SIZE_LABELS[size]}\n(n={n_ref} MILP-PuLP optimal)", fontsize=8.5)
            ax.set_ylabel("Gap to MILP (PuLP) optimal (%)", fontsize=8)
            ax.tick_params(axis="y", labelsize=7.5)
            _style_ax(ax)

        fig.legend(
            handles=solver_legend_handles(other_solvers) +
                    [Line2D([0],[0], color="black", linestyle="--",
                            linewidth=1.1, label="MILP (PuLP) reference (0%)")],
            loc="lower center", ncol=6, fontsize=7.5,
            framealpha=0.9, bbox_to_anchor=(0.5, -0.10),
        )
        fig.tight_layout()
        savefig(fig, f"fig05_gap_to_pulp_{mode}")

print("Fig 05 -- Gap-to-PuLP plots ...")
make_gap()


# ---------------------------------------------------------------------------
# Fig 06 -- Per-instance sorted scatter (makespan)
# ---------------------------------------------------------------------------
def make_sorted_scatter():
    for mode in MODES:
        for size in SIZES:
            sub = df[(df["mode"] == mode) & (df["size"] == size)].copy()
            # Best (min) makespan per variation per solver
            piv = (sub.groupby(["solver", "variation"])["makespan"]
                      .min().unstack(level=0))

            # Sort by PuLP optimal if present, else by Gurobi, else mean
            if "milp_ortools" in piv.columns:
                piv = piv.sort_values("milp_ortools")
            elif "milp_gurobi" in piv.columns:
                piv = piv.sort_values("milp_gurobi")
            else:
                piv = piv.assign(_mean=piv.mean(axis=1)).sort_values("_mean").drop("_mean", axis=1)

            xs = np.arange(len(piv))
            fig, ax = plt.subplots(figsize=(min(FIG_W * 1.25, 13), 4.2))

            for solver in SOLVERS:
                if solver not in piv.columns:
                    continue
                ys = piv[solver].values
                ax.scatter(
                    xs, ys,
                    color=COLORS[solver],
                    marker=MARKERS[solver],
                    label=SOLVER_SHORT[solver],
                    s=14, alpha=0.72,
                    edgecolors="none",
                    linewidths=0,
                    zorder=3,
                )

            ax.set_xlabel("Workflow instance (sorted by MILP-PuLP makespan)", fontsize=9)
            ax.set_ylabel("Makespan (s)", fontsize=9)
            ax.set_title(
                f"Per-Instance Makespan  |  {size.upper()}  {mode.capitalize()} mode",
                fontsize=10, fontweight="bold",
            )
            ax.legend(handles=line_legend_handles(), fontsize=7.5, ncol=3,
                      framealpha=0.88, markerscale=1.4)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.grid(axis="y")
            fig.tight_layout()
            savefig(fig, f"fig06_scatter_{size}_{mode}")

print("Fig 06 -- Per-instance scatter plots ...")
make_sorted_scatter()


# ---------------------------------------------------------------------------
# Fig 07 -- Node utilisation heatmap (hetero mode)
# ---------------------------------------------------------------------------
def make_node_util_heatmap():
    HETERO_NODES = [
        "iot_1", "edge_1", "cloud_1",
        "scc_cascadelake_1", "scc_skylake_1", "emmy_p3",
        "grete_p3_gpu", "grete_p1_gpu",
    ]
    NODE_SHORT = {
        "iot_1": "IoT-1", "edge_1": "Edge-1", "cloud_1": "Cloud-1",
        "scc_cascadelake_1": "CascadeLake", "scc_skylake_1": "Skylake",
        "emmy_p3": "Emmy-P3",
        "grete_p3_gpu": "Grete-P3\n(GPU)", "grete_p1_gpu": "Grete-P1\n(GPU)",
    }

    for size in SIZES:
        matrix = np.zeros((len(SOLVERS), len(HETERO_NODES)))

        for si, solver in enumerate(SOLVERS):
            cell_dir = os.path.join(BASE, solver, f"{size}_hetero")
            if not os.path.isdir(cell_dir):
                continue
            node_sums  = {n: 0.0 for n in HETERO_NODES}
            node_files = 0
            for path in glob.glob(os.path.join(cell_dir, "*.json")):
                try:
                    d = json.load(open(path))
                except Exception:
                    continue
                if "task_to_node" in d:
                    assignments = list(d["task_to_node"].values())
                    total = len(assignments)
                    for n in HETERO_NODES:
                        node_sums[n] += sum(1 for v in assignments if v == n) / max(total, 1) * 100
                elif "node_utilization" in d:
                    util = d["node_utilization"]
                    total = sum(util.values()) or 1
                    for n in HETERO_NODES:
                        node_sums[n] += util.get(n, 0) / total * 100
                else:
                    continue
                node_files += 1

            if node_files > 0:
                for ni, n in enumerate(HETERO_NODES):
                    matrix[si, ni] = node_sums[n] / node_files

        fig, ax = plt.subplots(figsize=(9.5, 3.8))
        im = ax.imshow(matrix, aspect="auto", cmap="YlOrRd", vmin=0, vmax=100)

        ax.set_xticks(range(len(HETERO_NODES)))
        ax.set_xticklabels([NODE_SHORT[n] for n in HETERO_NODES],
                            rotation=35, ha="right", fontsize=7.8)
        ax.set_yticks(range(len(SOLVERS)))
        ax.set_yticklabels([SOLVER_SHORT[s] for s in SOLVERS], fontsize=8.2)
        ax.set_title(
            f"Mean Node Utilisation (% of tasks assigned per file)  |  "
            f"{size.upper()} Hetero",
            fontsize=10, fontweight="bold",
        )

        for i in range(len(SOLVERS)):
            for j in range(len(HETERO_NODES)):
                v = matrix[i, j]
                if v >= 1:
                    ax.text(j, i, f"{v:.0f}%", ha="center", va="center",
                            fontsize=6.8,
                            color="white" if v > 55 else "black",
                            fontweight="bold")

        cb = fig.colorbar(im, ax=ax, fraction=0.024, pad=0.02)
        cb.set_label("% of tasks", fontsize=8)
        cb.ax.tick_params(labelsize=7.5)
        fig.tight_layout()
        savefig(fig, f"fig07_node_util_{size}_hetero")

print("Fig 07 -- Node utilisation heatmaps ...")
make_node_util_heatmap()


# ---------------------------------------------------------------------------
# Fig 08 -- Solve-status breakdown
# ---------------------------------------------------------------------------
def make_solve_status():
    STATUS_COLORS  = {
        "optimal":    "#009E73",
        "feasible":   "#56B4E9",
        "not solved": "#E69F00",
        "infeasible": "#D55E00",
        "solved":     "#009E73",
    }
    STATUS_HATCHES = {
        "optimal":    "///",
        "feasible":   "...",
        "not solved": "|||",
        "infeasible": "xxx",
        "solved":     "///",
    }
    STATUS_ORDER  = ["optimal", "feasible", "not solved", "infeasible"]
    # Only solvers that carry a status field
    stat_solvers  = ["milp_ortools", "milp_gurobi", "cpsat", "heft"]
    # Order: mode outer, size inner -> row 0 = homo, row 1 = hetero; columns = rnc50/100/300
    cells         = [(sz, mo) for mo in MODES for sz in SIZES]

    fig, axes = plt.subplots(2, 3, figsize=(FIG_W * 1.05, 6.2),
                             sharex=False, sharey=True)
    fig.suptitle("Solve Status per Solver and Graph Size",
                 fontsize=11, fontweight="bold", y=1.01)

    ROW_LABELS = {0: "Homogeneous (3-node)", 1: "Heterogeneous (8-node)"}

    for ax_idx, (size, mode) in enumerate(cells):
        row = ax_idx // 3
        col = ax_idx % 3
        ax  = axes[row, col]
        sub = df[(df["size"] == size) & (df["mode"] == mode)]
        bottoms = np.zeros(len(stat_solvers))

        for st in STATUS_ORDER:
            heights = []
            for s in stat_solvers:
                sv    = sub[sub["solver"] == s]
                total = len(sv)
                # GNNRL "solved" counts as "optimal" visually
                count = sv["status"].isin([st, "solved"] if st == "optimal" else [st]).sum()
                heights.append(count / total * 100 if total > 0 else 0)

            ax.bar(
                range(len(stat_solvers)), heights,
                bottom=bottoms,
                color=STATUS_COLORS[st],
                hatch=STATUS_HATCHES[st],
                edgecolor="black", linewidth=0.6,
                alpha=0.88,
            )
            bottoms += np.array(heights)

        ax.set_xticks(range(len(stat_solvers)))
        ax.set_xticklabels([SOLVER_SHORT[s] for s in stat_solvers], fontsize=7.5)
        ax.set_ylim(0, 108)
        ax.tick_params(axis="y", labelsize=7.5)
        ax.set_title(size.upper(), fontsize=9)
        _style_ax(ax)

        # Left-side row label on first column only
        if col == 0:
            ax.set_ylabel(ROW_LABELS[row] + "\n% of instances", fontsize=7.8)
        else:
            ax.set_ylabel("")

    handles = [
        Patch(facecolor=STATUS_COLORS[st], hatch=STATUS_HATCHES[st],
              edgecolor="black", label=st, alpha=0.88)
        for st in STATUS_ORDER
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4,
               fontsize=8, framealpha=0.9, bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout()
    savefig(fig, "fig08_solve_status")

print("Fig 08 -- Solve-status breakdown ...")
make_solve_status()


# ---------------------------------------------------------------------------
# Fig 09 -- Overview dashboard (4 panels)
# ---------------------------------------------------------------------------
def make_overview():
    xs = [50, 100, 300]
    fig = plt.figure(figsize=(FIG_W * 1.15, 8.2))
    gs  = gridspec.GridSpec(2, 2, hspace=0.48, wspace=0.38)
    # Row 0 = Homo, Row 1 = Hetero; columns = Objective, Makespan
    panels = [
        ("homo",  "objective", "Median Objective"),
        ("homo",  "makespan",  "Median Makespan (s)"),
        ("hetero","objective", "Median Objective"),
        ("hetero","makespan",  "Median Makespan (s)"),
    ]
    ROW_LABELS = {0: "Homogeneous (3-node)", 1: "Heterogeneous (8-node)"}

    for idx, (mode, metric, title) in enumerate(panels):
        row = idx // 2
        col = idx % 2
        ax  = fig.add_subplot(gs[row, col])
        sub = df[df["mode"] == mode]

        for solver in SOLVERS:
            ys = []
            for size in SIZES:
                vals = sub[(sub["solver"]==solver) & (sub["size"]==size)][metric].dropna()
                ys.append(vals.median() if len(vals) > 0 else np.nan)
            ax.plot(xs, ys,
                    color=COLORS[solver],
                    marker=MARKERS[solver],
                    linestyle=LINESTYLES[solver],
                    linewidth=1.8, markersize=6,
                    label=SOLVER_SHORT[solver])

        ax.set_xticks(xs)
        ax.set_xticklabels(["50\n(RNC-50)", "100\n(RNC-100)", "300\n(RNC-300)"], fontsize=7.5)
        ax.set_xlabel("Tasks", fontsize=8.5)
        ax.set_title(title, fontsize=9.5, fontweight="bold")
        ax.tick_params(labelsize=7.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid()

        # Left column: row label as ylabel; right column: plain metric label
        if col == 0:
            ylabel = ("Median objective" if metric == "objective"
                      else "Median makespan (s)")
            ax.set_ylabel(f"{ROW_LABELS[row]}\n{ylabel}", fontsize=8.2)
        else:
            ax.set_ylabel(
                "Median objective" if metric == "objective" else "Median makespan (s)",
                fontsize=8.5)

    fig.legend(handles=line_legend_handles(), loc="lower center",
               ncol=6, fontsize=8, framealpha=0.9,
               bbox_to_anchor=(0.5, -0.04))
    fig.suptitle(
        "Benchmark Overview: Objective and Makespan Scalability\n"
        "All 6 Solvers  |  rnc50, rnc100, rnc300  |  Homo and Hetero",
        fontsize=11, fontweight="bold", y=1.01,
    )
    savefig(fig, "fig09_overview")

print("Fig 09 -- Overview dashboard ...")
make_overview()


# ---------------------------------------------------------------------------
# Summary CSV
# ---------------------------------------------------------------------------
def write_summary():
    rows = []
    for solver in SOLVERS:
        for size in SIZES:
            for mode in MODES:
                sub = df[(df["solver"]==solver) & (df["size"]==size) & (df["mode"]==mode)]
                if sub.empty:
                    continue
                obj = sub["objective"].dropna()
                ms  = sub["makespan"].dropna()
                wt  = sub["wall_time_s"].dropna()
                rows.append({
                    "solver":         SOLVER_SHORT[solver],
                    "size":           size,
                    "mode":           mode,
                    "n_instances":    len(sub),
                    "obj_mean":       round(obj.mean(),   3) if len(obj) else "",
                    "obj_median":     round(obj.median(), 3) if len(obj) else "",
                    "obj_std":        round(obj.std(),    3) if len(obj) else "",
                    "ms_mean":        round(ms.mean(),    3) if len(ms) else "",
                    "ms_median":      round(ms.median(),  3) if len(ms) else "",
                    "ms_std":         round(ms.std(),     3) if len(ms) else "",
                    "wt_mean_s":      round(wt.mean(),    4) if len(wt) else "",
                    "n_optimal":      (sub["status"] == "optimal").sum(),
                    "n_feasible":     (sub["status"].isin(["feasible","solved"])).sum(),
                    "n_not_solved":   (sub["status"] == "not solved").sum(),
                })
    out = pd.DataFrame(rows)
    path = os.path.join(FIGDIR, "summary_stats.csv")
    out.to_csv(path, index=False)
    print(f"  Saved summary_stats.csv ({len(out)} rows)")

print("Summary CSV ...")
write_summary()

print("\nAll figures written to:", FIGDIR)
for f in sorted(os.listdir(FIGDIR)):
    kb = os.path.getsize(os.path.join(FIGDIR, f)) // 1024
    print(f"  {f:55s} {kb:4d} KB")
