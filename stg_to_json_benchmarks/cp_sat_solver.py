#!/usr/bin/env python3
"""
cp_sat_solver.py — Feature-aware CP-SAT baseline for workflow scheduling (OR-Tools)

Implements the SAME mathematical model as milp_solver_revised.py (Thesis Chapter, RQ2):

    min  alpha * Σ_{j∈T} Σ_{i∈N} U_ij * x_ij  +  beta * C_max

    s.t. (1)  Σ_{i∈N} x_ij = 1                               ∀ j ∈ T
         (2)  x_ij ≤ a_ij                                     ∀ i ∈ N, j ∈ T
         (3)  f_j ≥ s_j + Σ_{i∈N} p_ij * x_ij                ∀ j ∈ T
         (4)  C_max ≥ f_j                                      ∀ j ∈ T
         (5)  s_j ≥ f_{j'} + Σ_{i',i} D_{i',i}^{(j',j)} * y_{i',i}^{(j',j)}  ∀(j',j)∈δ
         (6)  y_{i',i}^{(j',j)} = x_{i'j'} · x_{ij}  (linearized)

CP-SAT solves a discretized integer version with precision TIME_SCALE (milliseconds).
CP-SAT typically outperforms CBC on scheduling due to specialized propagators and
incremental constraint solving.

References:
    - Topcuoglu et al. (2002), TPDS — HEFT / scheduling taxonomy
    - Sharma & Kunkel (2025), COMPSAC — Workflow-Driven Modeling (exact formulation)
    - Nemhauser & Wolsey (1988), Integer Programming — LP/IP theory
    - OR-Tools CP-SAT: https://developers.google.com/optimization/reference/python/sat/python/cp_model

Author: Generated for Aasish Kumar Sharma, PhD Thesis — University of Göttingen
"""

from __future__ import annotations

import argparse
import json
import math
import re
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

try:
    from ortools.sat.python import cp_model
except ImportError:
    raise SystemExit(
        "OR-Tools is required: pip install ortools --break-system-packages"
    )

# ─────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────

# Discretisation precision: 1 unit = 1 ms → 0.001 s
# Increase if numeric precision is needed (at cost of solving time).
TIME_SCALE = 1000   # float seconds → integer milliseconds
MAX_T      = 10_000_000  # 10 000 s upper bound in original scale → 10^7 units

GPU_MARKERS = {
    "gpu", "cuda", "opencl", "v100", "a100", "h100", "rtx", "tesla",
    "nvidia_v100", "nvidia_a100", "nvidia_h100",
}

DEFAULT_TASK = {
    "cores": 1, "memory_mb": 1024,
    "features": ["cpu"], "data_mb": 0, "duration": 10.0,
}


# ─────────────────────────────────────────────────────────────
# Utility helpers  (shared with MILP solver, duplicated for
# self-contained deployment)
# ─────────────────────────────────────────────────────────────

def ts_compact() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def norm_token(x: str) -> str:
    x = str(x).strip().lower()
    x = re.sub(r"[^a-z0-9]+", "_", x)
    return re.sub(r"_+", "_", x).strip("_")


def norm_feature_set(lst: Any) -> Set[str]:
    if not lst:
        return set()
    if isinstance(lst, (set, tuple, list)):
        return {norm_token(v) for v in lst}
    return {norm_token(lst)}


def to_mb(value: float, unit: str) -> float:
    unit = unit.strip().upper()
    if unit == "MB": return float(value)
    if unit == "GB": return float(value) * 1024.0
    raise ValueError(f"Unsupported unit: {unit}")


def node_net_rate_mb_s(node: dict, default: float = 1000.0) -> float:
    rates = node.get("data_transfer_rate", {}) or {}
    if not isinstance(rates, dict) or not rates:
        return default
    vals = [float(v) for v in rates.values() if str(v).replace(".", "", 1).isdigit()]
    return max(vals) if vals else default


def infer_device(features: Set[str]) -> str:
    return "GPU" if (features & GPU_MARKERS) else "CPU"


def node_has_gpu(node: "Node") -> bool:
    if "gpu" in node.features:
        return True
    ks = {norm_token(k) for k in (node.processing_speed or {}).keys()}
    return "gpu" in ks


def pick_speed(task_feat: Set[str], node_proc_speed: dict) -> Tuple[float, str]:
    proc = node_proc_speed or {}
    norm_map: Dict[str, float] = {}
    for k, v in proc.items():
        try:
            norm_map[norm_token(k)] = float(v)
        except Exception:
            continue
    candidates = [f for f in task_feat if f in norm_map]
    if candidates:
        best = max(candidates, key=lambda kk: norm_map.get(kk, 0.0))
        return max(1e-9, norm_map[best]), best
    dev = infer_device(task_feat)
    if dev == "GPU" and "gpu" in norm_map:
        return max(1e-9, norm_map["gpu"]), "GPU"
    if "cpu" in norm_map:
        return max(1e-9, norm_map["cpu"]), "CPU"
    return 1.0, "default_1.0"


def float_to_int(x: float, scale: int = TIME_SCALE) -> int:
    """Convert float seconds to integer time units."""
    return max(0, int(round(x * scale)))


def int_to_float(x: int, scale: int = TIME_SCALE) -> float:
    return x / scale


# ─────────────────────────────────────────────────────────────
# Data model
# ─────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Task:
    tid: str
    duration: float
    cores: float
    mem_mb: float
    features: Set[str]
    deps: List[str]
    data_mb: float


@dataclass(frozen=True)
class Node:
    nid: str
    tier: str
    cores: float
    mem_mb: float
    features: Set[str]
    processing_speed: dict
    data_transfer_rate: dict


# ─────────────────────────────────────────────────────────────
# Loaders  (identical to MILP solver)
# ─────────────────────────────────────────────────────────────

def load_workflow(path: str, task_mem_unit: str, task_data_unit: str) -> Tuple[str, Dict[str, Task]]:
    wf = json.loads(Path(path).read_text())
    if "workflows" in wf:
        wname = list(wf["workflows"].keys())[0]
        raw_tasks = wf["workflows"][wname]["tasks"]
    elif "tasks" in wf:
        wname = Path(path).stem
        raw_tasks = wf["tasks"]
    else:
        raise ValueError("Workflow JSON must contain 'tasks' or 'workflows'.")
    tasks: Dict[str, Task] = {}
    for tid, t in raw_tasks.items():
        tasks[tid] = Task(
            tid=tid,
            duration=float(t.get("duration", DEFAULT_TASK["duration"])),
            cores=float(t.get("cores", DEFAULT_TASK["cores"])),
            mem_mb=to_mb(float(t.get("memory_required", DEFAULT_TASK["memory_mb"])), task_mem_unit),
            features=norm_feature_set(t.get("features", DEFAULT_TASK["features"])),
            deps=list(t.get("dependencies", [])) if t.get("dependencies") else [],
            data_mb=to_mb(float(t.get("data", DEFAULT_TASK["data_mb"])), task_data_unit),
        )
    return wname, tasks


def load_nodes(path: str, node_mem_unit: str) -> Dict[str, Node]:
    nd = json.loads(Path(path).read_text())
    raw_nodes = nd.get("nodes", nd)
    if not isinstance(raw_nodes, dict):
        raise ValueError("Nodes JSON must be an object or contain 'nodes' object.")
    nodes: Dict[str, Node] = {}
    for nid, n in raw_nodes.items():
        if not isinstance(n, dict):
            continue
        feats = norm_feature_set(n.get("features", []))
        feats.add("cpu")
        nodes[nid] = Node(
            nid=nid,
            tier=str(n.get("tier", "HPC")),
            cores=float(n.get("cores", 1.0)),
            mem_mb=to_mb(float(n.get("memory", 0.0)), node_mem_unit),
            features=feats,
            processing_speed=n.get("processing_speed", {}) or {},
            data_transfer_rate=n.get("data_transfer_rate", {}) or {},
        )
    return nodes


# ─────────────────────────────────────────────────────────────
# Parameter builders  (identical semantics to MILP solver)
# ─────────────────────────────────────────────────────────────

def build_a(tasks: Dict[str, Task], nodes: Dict[str, Node]) -> Dict[Tuple[str, str], int]:
    a: Dict[Tuple[str, str], int] = {}
    for j, t in tasks.items():
        dev = infer_device(t.features)
        for i, n in nodes.items():
            feat_ok = t.features.issubset(n.features)
            res_ok = (t.cores <= n.cores) and (t.mem_mb <= n.mem_mb)
            if dev == "GPU":
                feat_ok = feat_ok and node_has_gpu(n)
            a[(i, j)] = int(feat_ok and res_ok)
    return a


def build_p_U(
    tasks: Dict[str, Task],
    nodes: Dict[str, Node],
    a: Dict[Tuple[str, str], int],
    mem_weight: float = 0.0,
) -> Tuple[Dict[Tuple[str, str], float], Dict[Tuple[str, str], float], Dict[Tuple[str, str], str]]:
    p: Dict[Tuple[str, str], float] = {}
    U: Dict[Tuple[str, str], float] = {}
    speed_key: Dict[Tuple[str, str], str] = {}
    for j, t in tasks.items():
        for i, n in nodes.items():
            if a[(i, j)] == 0:
                p[(i, j)] = 1e12
                U[(i, j)] = 1e12
                speed_key[(i, j)] = "infeasible"
                continue
            speed, key = pick_speed(t.features, n.processing_speed)
            speed_key[(i, j)] = str(key)
            p_ij = max(1e-6, t.duration / speed)
            p[(i, j)] = p_ij
            mem_gb = t.mem_mb / 1024.0
            U[(i, j)] = (t.cores * p_ij) + (mem_weight * mem_gb * p_ij)
    return p, U, speed_key


def build_D(tasks: Dict[str, Task], nodes: Dict[str, Node]) -> Dict[Tuple[str, str, str, str], float]:
    D: Dict[Tuple[str, str, str, str], float] = {}
    net = {nid: node_net_rate_mb_s({"data_transfer_rate": nodes[nid].data_transfer_rate})
           for nid in nodes}
    for j, t in tasks.items():
        for jp in t.deps:
            if jp not in tasks:
                continue
            data_mb = max(0.0, tasks[jp].data_mb)
            for ip in nodes:
                for i in nodes:
                    D[(ip, i, jp, j)] = 0.0 if ip == i else data_mb / max(1e-9, min(net[ip], net[i]))
    return D


# ─────────────────────────────────────────────────────────────
# CP-SAT formulation
# ─────────────────────────────────────────────────────────────

def solve(
    workflow: str,
    nodes_path: str,
    alpha: float,
    beta: float,
    time_limit: int,
    node_mem_unit: str,
    task_mem_unit: str,
    task_data_unit: str,
    mem_weight: float,
    num_workers: int,
    verbose: bool,
) -> dict:

    wname, tasks = load_workflow(workflow, task_mem_unit, task_data_unit)
    nodes = load_nodes(nodes_path, node_mem_unit)

    T = list(tasks.keys())
    N = list(nodes.keys())
    if not T or not N:
        raise ValueError("Empty tasks or nodes.")

    a   = build_a(tasks, nodes)
    p, U, sk = build_p_U(tasks, nodes, a, mem_weight)
    D   = build_D(tasks, nodes)

    # Feasibility pre-screen
    infeasible_tasks = [j for j in T if sum(a[(i, j)] for i in N) == 0]
    if infeasible_tasks:
        return {
            "status": "infeasible",
            "solver": {"name": "OR-Tools_CP-SAT", "time_limit_s": time_limit, "wall_time_s": 0.0},
            "workflow": wname,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "reason": "feature_or_resource_mismatch",
            "infeasible_tasks": infeasible_tasks,
        }

    # Feasible (i,j) pairs
    feasible: Dict[str, List[str]] = {j: [i for i in N if a[(i, j)]] for j in T}
    edges: List[Tuple[str, str]] = [(jp, j) for j in T for jp in tasks[j].deps if jp in tasks]

    # ── CP-SAT model ─────────────────────────────────────────
    model = cp_model.CpModel()

    # Integer time variables (units = TIME_SCALE^{-1} seconds, i.e. ms)
    horizon = MAX_T  # pessimistic upper bound
    s_var: Dict[str, cp_model.IntVar] = {}
    f_var: Dict[str, cp_model.IntVar] = {}
    for j in T:
        s_var[j] = model.NewIntVar(0, horizon, f"s_{j}")
        f_var[j] = model.NewIntVar(0, horizon, f"f_{j}")

    cmax = model.NewIntVar(0, horizon, "Cmax")

    # Binary assignment:  x[i][j] ∈ {0,1}
    x: Dict[Tuple[str, str], cp_model.IntVar] = {}
    for j in T:
        for i in N:
            x[(i, j)] = model.NewBoolVar(f"x_{i}_{j}")

    # (1) Each task assigned to exactly one feasible node
    for j in T:
        model.Add(sum(x[(i, j)] for i in N) == 1)

    # (2) Feature/resource feasibility
    for j in T:
        for i in N:
            if a[(i, j)] == 0:
                model.Add(x[(i, j)] == 0)

    # (3) f_j = s_j + Σ_i p_ij * x_ij  (exact; p_ij is integer in scaled units)
    for j in T:
        p_int = {i: float_to_int(p[(i, j)]) for i in N}
        model.Add(
            f_var[j] == s_var[j] + sum(p_int[i] * x[(i, j)] for i in N)
        )

    # (4) C_max ≥ f_j
    for j in T:
        model.Add(cmax >= f_var[j])

    # (5) & (6) Precedence + data transfer
    # y_{ip,i,jp,j} = x[(ip,jp)] ∧ x[(i,j)]  (reified)
    y: Dict[Tuple, cp_model.IntVar] = {}
    for (jp, j) in edges:
        for ip in N:
            for i in N:
                yvar = model.NewBoolVar(f"y_{ip}_{i}_{jp}_{j}")
                y[(ip, i, jp, j)] = yvar
                model.AddBoolAnd([x[(ip, jp)], x[(i, j)]]).OnlyEnforceIf(yvar)
                model.AddBoolOr([x[(ip, jp)].Not(), x[(i, j)].Not()]).OnlyEnforceIf(yvar.Not())

    for (jp, j) in edges:
        D_int = {(ip, i): float_to_int(D[(ip, i, jp, j)]) for ip in N for i in N}
        # s_j >= f_jp + Σ_{ip,i} D_int[ip,i] * y
        model.Add(
            s_var[j] >= f_var[jp] + sum(D_int[(ip, i)] * y[(ip, i, jp, j)]
                                        for ip in N for i in N)
        )

    # ── Objective (scaled integer) ────────────────────────────
    # Coefficients: alpha * U_ij  and  beta  (scaled by TIME_SCALE for cmax)
    # We need a single integer objective → scale by TIME_SCALE and round.
    OBJ_SCALE = TIME_SCALE  # additional scale so makespan aligns with usage

    usage_terms = []
    for j in T:
        for i in N:
            if a[(i, j)] == 0:
                continue
            coeff = int(round(alpha * U[(i, j)] * OBJ_SCALE))
            usage_terms.append(coeff * x[(i, j)])

    # cmax is already in TIME_SCALE units; multiply by beta
    beta_int = int(round(beta * OBJ_SCALE))

    model.Minimize(sum(usage_terms) + beta_int * cmax)

    # ── Solver configuration ─────────────────────────────────
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = float(time_limit)
    solver.parameters.num_search_workers = num_workers
    solver.parameters.log_search_progress = verbose

    t0 = time.time()
    status_code = solver.Solve(model)
    t1 = time.time()

    status_name = solver.StatusName(status_code).lower()  # optimal / feasible / infeasible / unknown

    # ── Extract solution ─────────────────────────────────────
    if status_code not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return {
            "status": status_name,
            "solver": {"name": "OR-Tools_CP-SAT", "time_limit_s": time_limit, "wall_time_s": round(t1 - t0, 4)},
            "workflow": wname,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        }

    makespan_val = int_to_float(solver.Value(cmax))
    usage_val = sum(
        (solver.Value(x[(i, j)]) * U[(i, j)])
        for i in N for j in T
    )
    obj_val = alpha * usage_val + beta * makespan_val

    schedule: Dict[str, Any] = {}
    node_util: Dict[str, int] = {i: 0 for i in N}
    for j in T:
        assigned = next((i for i in N if solver.Value(x[(i, j)]) > 0), None)
        if assigned:
            node_util[assigned] += 1
        schedule[j] = {
            "node": assigned,
            "start_time": int_to_float(solver.Value(s_var[j])),
            "finish_time": int_to_float(solver.Value(f_var[j])),
            "p_ij": p.get((assigned, j), float("nan")) if assigned else float("nan"),
            "speed_key": sk.get((assigned, j)) if assigned else None,
            "task_features": sorted(tasks[j].features),
            "task_cores": tasks[j].cores,
            "task_mem_mb": tasks[j].mem_mb,
            "task_data_mb": tasks[j].data_mb,
        }

    return {
        "status": status_name,
        "workflow": wname,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "solver": {
            "name": "OR-Tools_CP-SAT",
            "time_limit_s": time_limit,
            "wall_time_s": round(t1 - t0, 4),
            "num_workers": num_workers,
            "time_scale": TIME_SCALE,
            "best_bound": solver.BestObjectiveBound() / (OBJ_SCALE * TIME_SCALE),
            "objective_raw_scaled": solver.ObjectiveValue(),
        },
        "objective": {
            "alpha": alpha, "beta": beta,
            "value": round(obj_val, 6),
            "makespan": round(makespan_val, 6),
            "usage_term": round(usage_val, 6),
        },
        "counts": {"num_tasks": len(T), "num_nodes": len(N), "num_edges": len(edges)},
        "units": {"tasks_memory": "MB", "tasks_data": "MB", "nodes_memory": "MB", "time": "seconds"},
        "node_utilization": node_util,
        "schedule": schedule,
    }


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(description="Feature-aware CP-SAT workflow scheduler (OR-Tools)")
    ap.add_argument("workflow", help="Workflow JSON (STG-style)")
    ap.add_argument("--nodes", required=True, help="Nodes JSON")
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--beta",  type=float, default=1.0)
    ap.add_argument("--time-limit", type=int, default=300)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--node-mem-unit", choices=["MB", "GB"], default="MB")
    ap.add_argument("--task-mem-unit", choices=["MB", "GB"], default="MB")
    ap.add_argument("--task-data-unit", choices=["MB", "GB"], default="MB")
    ap.add_argument("--mem-weight", type=float, default=0.0)
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("-o", "--output", default="solution_cpsat.json")
    args = ap.parse_args()

    sol = solve(
        workflow=args.workflow,
        nodes_path=args.nodes,
        alpha=args.alpha,
        beta=args.beta,
        time_limit=args.time_limit,
        node_mem_unit=args.node_mem_unit,
        task_mem_unit=args.task_mem_unit,
        task_data_unit=args.task_data_unit,
        mem_weight=args.mem_weight,
        num_workers=args.num_workers,
        verbose=args.verbose,
    )

    out = Path(args.output)
    out = out.parent / f"{out.stem}_{ts_compact()}{out.suffix}"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(sol, indent=2))
    print(f"✓ Result written: {out}  (status={sol.get('status')})")
    if sol.get("status") in {"optimal", "feasible"}:
        obj = sol.get("objective", {})
        print(f"  makespan={obj.get('makespan'):.4f}  usage={obj.get('usage_term'):.4f}"
              f"  obj={obj.get('value'):.4f}")
        return 0
    if sol.get("status") == "infeasible":
        return 2
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
