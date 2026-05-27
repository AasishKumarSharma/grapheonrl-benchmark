# Benchmark Results: Multi-Solver Workflow Scheduling on Heterogeneous HPC Systems

This directory contains the figure generation script and all analysis figures for the
**Phase I Small Scale Benchmark Tests in Edge Device** of the GNNRL workflow scheduling project.

The raw result data (6,480 JSON files) is archived on Zenodo:

**Benchmark Results Dataset DOI:** https://doi.org/10.5281/zenodo.20419279

---

## What Is Here

| Item | Description |
|---|---|
| `generate_figures.py` | Reproduces all 21 figures from the raw JSON result files |
| `figures/` | All 21 analysis figures in PDF (publication quality) and PNG (screen) formats |
| `figures/summary_stats.csv` | Per-cell summary statistics (median objective, makespan, wall time) |

---

## Figures

| Figure | Description |
|---|---|
| `fig01_boxplot_objective_homo/hetero` | Objective value distribution per solver, homo and hetero systems |
| `fig02_boxplot_makespan_homo/hetero` | Makespan distribution per solver, homo and hetero systems |
| `fig03_scalability_homo/hetero` | Median objective scalability across rnc50/100/300 |
| `fig04_walltime_homo/hetero` | Solver wall time distribution per cell |
| `fig05_gap_to_pulp_homo/hetero` | Optimality gap relative to MILP (PuLP) optimal baseline |
| `fig06_scatter_rnc50/100/300_homo/hetero` | Per-instance objective scatter: GNNRL vs exact solvers |
| `fig07_node_util_rnc50/100/300_hetero` | Node utilisation heatmaps for heterogeneous systems |
| `fig08_solve_status` | Solve status breakdown (optimal/feasible/not solved/infeasible) |
| `fig09_overview` | Dashboard: median objective and makespan scalability, all 6 solvers |

All figures use the Okabe-Ito (2008) colorblind-safe palette with distinct hatches and
markers for mono-print safety.

---

## Reproducing the Figures

1. Download the raw JSON result files from Zenodo:
   https://doi.org/10.5281/zenodo.20419279

2. Extract `benchmark_solver_results_main.zip` to a folder named `main_results/`
   at the same level as `generate_figures.py`:

   ```
   benchmark_results/
   +-- generate_figures.py
   +-- main_results/          <-- extracted from benchmark_solver_results_main.zip
   |   +-- milp_ortools/
   |   +-- milp_gurobi/
   |   +-- cpsat/
   |   +-- heft/
   |   +-- gnnrl_self/
   |   +-- gnnrl_teacher/
   +-- figures/               <-- output written here
   ```

3. Install dependencies:
   ```bash
   pip install pandas numpy matplotlib
   ```

4. Run the script:
   ```bash
   python generate_figures.py
   ```

   All 21 PDF and 21 PNG figures plus `summary_stats.csv` are written to `figures/`.

---

## Solvers

| Solver | Type | Notes |
|---|---|---|
| MILP (PuLP) | Exact (MILP) | PuLP library, CBC back-end; used as optimality gap reference |
| MILP (Gurobi) | Exact (MILP) | Gurobi solver; some runs interrupted before proven optimal |
| CP-SAT | Exact (CP) | OR-Tools CP-SAT solver |
| HEFT | Heuristic | Heterogeneous Earliest Finish Time |
| GNNRL (self) | Learned | GNN-RL model evaluated on its own training distribution |
| GNNRL (teacher) | Learned | GNN-RL model trained with teacher guidance |

---

## System Models

| Mode | Nodes | Node types |
|---|---|---|
| Homogeneous | 3 | hpc_node_1, hpc_node_2, hpc_node_3 |
| Heterogeneous | 8 | iot_1, edge_1, cloud_1, scc_cascadelake_1, scc_skylake_1, emmy_p3, grete_p3_gpu, grete_p1_gpu |

---

## Objective Function

All solvers minimise the same weighted objective:

```
Objective = alpha * makespan + beta * usage_term    (alpha = beta = 1.0)
```

---

## Execution Environment (Phase I)

All runs were executed on an edge device:

| Property | Value |
|---|---|
| CPU | Intel Core i5-1145G7 (11th Gen, Tiger Lake) |
| Architecture | x86_64, 4 cores / 8 threads, 2.60 GHz / 4.40 GHz |
| TDP | 15 W |
| RAM | 16 GB |
| OS | Ubuntu 22.04.5 LTS |
| Python | 3.10.12 |

Phase II cluster validation on real HPC infrastructure will be published as a separate dataset.

---

## Citation

If you use these results or figures, please cite the benchmark results dataset:

```bibtex
@dataset{Sharma2026BenchmarkResults,
  author    = {Sharma, Aasish Kumar and Kunkel, Julian Martin},
  title     = {Standard Task Graph ({STG}) Multi-Solver Benchmark Results
               for Workflow Scheduling in Heterogeneous High Performance
               Computing ({HPC}) Systems},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.20419279},
  url       = {https://doi.org/10.5281/zenodo.20419279}
}
```
