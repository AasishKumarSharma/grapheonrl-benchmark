#!/usr/bin/env python3
"""
heft_solver.py — Feature-aware HEFT (Heterogeneous Earliest Finish Time) scheduler

Implements the canonical HEFT algorithm (Topcuoglu, Hariri & Wu, 2002, IEEE TPDS)
extended for:
    • Feature/resource feasibility constraints (identical to MILP/CP-SAT a_ij matrix)
    • Data transfer delays D_{i',i}^{(j',j)} matching the thesis formulation
    • Multi-objective evaluation: alpha * Σ U_ij x_ij  +  beta * C_max
    • Heterogeneous processing times p_ij = duration_j / speed_{ij}

HEFT provides an approximate (heuristic) solution in polynomial time O(e·q²)
where e = number of edges and q = number of nodes.  It serves as an *upper bound*
reference for MILP/CP-SAT optimality evaluation.

Algorithm summary:
    Phase 1: Compute upward rank rank_u(j) = w̄_j + max_{k∈succ(j)}(c̄_{jk} + rank_u(k))
             where w̄_j = (1/|feasible_nodes|) * Σ_{i feasible} p_ij
                   c̄_{jk} = (1/|N|²) * Σ_{i',i} D_{i',i}^{(j,k)}
    Phase 2: Sort tasks by rank_u (non-increasing).  Assign each task to the node
             that gives the Earliest Finish Time (EFT) subject to:
             - Feature/resource feasibility (a_ij = 1)
             - Predecessor finish times + communication delays
             - Node availability (non-preemptive serial queue per node)

References:
    [1] Topcuoglu, H., Hariri, S., & Wu, M.-Y. (2002). Performance-Effective and
        Low-Complexity Task Scheduling for Heterogeneous Computing. IEEE TPDS, 13(3).
    [2] Sharma & Kunkel (2025), COMPSAC — Workflow-Driven Modeling
    [3] Pinedo, M. (2016). Scheduling: Theory, Algorithms, and Systems (5th ed.).

Author: Generated for Aasish Kumar Sharma, PhD Thesis — University of Göttingen
"""

from __future__ import annotations

import argparse
import json
import math
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple


# ─────────────────────────────────────────────────────────────
# Constants / defaults
# ─────────────────────────────────────────────────────────────

GPU_MARKERS = {
    "gpu", "cuda", "opencl", "v100", "a100", "h100", "rtx", "tesla",
    "nvidia_v100", "nvidia_a100", "nvidia_h100",
}

DEFAULT_TASK = {
    "cores": 1, "memory_mb": 1024,
    "features": ["cpu"], "data_mb": 0, "duration": 10.0,
}


# ─────────────────────────────────────────────────────────────
# Utilities
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


def node_net_rate_mb_s(dtr: dict, default: float = 1000.0) -> float:
    if not dtr:
        return default
    vals = []
    for v in dtr.values():
        try:
            vals.append(float(v))
        except Exception:
            pass
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
# Loaders
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
# Parameter builders
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


def build_p(
    tasks: Dict[str, Task],
    nodes: Dict[str, Node],
    a: Dict[Tuple[str, str], int],
) -> Tuple[Dict[Tuple[str, str], float], Dict[Tuple[str, str], float], Dict[Tuple[str, str], str]]:
    """Returns p_ij, U_ij, and speed_key."""
    p: Dict[Tuple[str, str], float] = {}
    U: Dict[Tuple[str, str], float] = {}
    sk: Dict[Tuple[str, str], str] = {}
    for j, t in tasks.items():
        for i, n in nodes.items():
            if a[(i, j)] == 0:
                p[(i, j)] = math.inf
                U[(i, j)] = math.inf
                sk[(i, j)] = "infeasible"
                continue
            speed, key = pick_speed(t.features, n.processing_speed)
            p_ij = max(1e-9, t.duration / speed)
            p[(i, j)] = p_ij
            U[(i, j)] = t.cores * p_ij          # core-time usage
            sk[(i, j)] = str(key)
    return p, U, sk


def build_D(tasks: Dict[str, Task], nodes: Dict[str, Node]) -> Dict[Tuple[str, str, str, str], float]:
    """D(ip, i, jp, j) = data transfer delay between predecessor jp→j when jp on ip, j on i."""
    D: Dict[Tuple[str, str, str, str], float] = {}
    net = {nid: node_net_rate_mb_s(nodes[nid].data_transfer_rate) for nid in nodes}
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
# Graph utilities
# ─────────────────────────────────────────────────────────────

def topological_sort(tasks: Dict[str, Task]) -> List[str]:
    """Kahn's algorithm for topological sort of the DAG."""
    in_deg: Dict[str, int] = {j: 0 for j in tasks}
    succ: Dict[str, List[str]] = {j: [] for j in tasks}
    for j, t in tasks.items():
        for p in t.deps:
            if p in tasks:
                in_deg[j] += 1
                succ[p].append(j)

    q = sorted([j for j in tasks if in_deg[j] == 0])
    order: List[str] = []
    while q:
        v = q.pop(0)
        order.append(v)
        for w in sorted(succ[v]):
            in_deg[w] -= 1
            if in_deg[w] == 0:
                q.append(w)
    if len(order) != len(tasks):
        raise ValueError("Workflow DAG has a cycle.")
    return order


def build_succ(tasks: Dict[str, Task]) -> Dict[str, List[str]]:
    succ: Dict[str, List[str]] = {j: [] for j in tasks}
    for j, t in tasks.items():
        for p in t.deps:
            if p in tasks:
                succ[p].append(j)
    return succ


# ─────────────────────────────────────────────────────────────
# HEFT Phase 1 — upward rank computation
# ─────────────────────────────────────────────────────────────

def compute_average_computation(
    tasks: Dict[str, Task],
    nodes: Dict[str, Node],
    p: Dict[Tuple[str, str], float],
    a: Dict[Tuple[str, str], int],
) -> Dict[str, float]:
    """
    w̄_j = average processing time over feasible nodes.
    If no feasible node exists, use median of all p values.
    """
    w_bar: Dict[str, float] = {}
    N = list(nodes.keys())
    for j in tasks:
        feasible_p = [p[(i, j)] for i in N if a[(i, j)] == 1 and math.isfinite(p[(i, j)])]
        if feasible_p:
            w_bar[j] = sum(feasible_p) / len(feasible_p)
        else:
            finite_p = [p[(i, j)] for i in N if math.isfinite(p[(i, j)])]
            w_bar[j] = (sum(finite_p) / len(finite_p)) if finite_p else tasks[j].duration
    return w_bar


def compute_average_communication(
    tasks: Dict[str, Task],
    nodes: Dict[str, Node],
    D: Dict[Tuple[str, str, str, str], float],
) -> Dict[Tuple[str, str], float]:
    """
    c̄_{jp,j} = average data transfer delay for edge (jp→j).
    Zero for intra-node; average over all cross-node pairs.
    """
    N = list(nodes.keys())
    c_bar: Dict[Tuple[str, str], float] = {}
    for j, t in tasks.items():
        for jp in t.deps:
            if jp not in tasks:
                continue
            vals = [D[(ip, i, jp, j)] for ip in N for i in N if ip != i]
            c_bar[(jp, j)] = (sum(vals) / len(vals)) if vals else 0.0
    return c_bar


def upward_rank(
    tasks: Dict[str, Task],
    topo_order: List[str],
    succ: Dict[str, List[str]],
    w_bar: Dict[str, float],
    c_bar: Dict[Tuple[str, str], float],
) -> Dict[str, float]:
    """
    rank_u(exit) = w̄_exit
    rank_u(j)    = w̄_j + max_{k ∈ succ(j)} (c̄_{j,k} + rank_u(k))

    Computed in reverse topological order.
    """
    rank: Dict[str, float] = {}
    for j in reversed(topo_order):
        if not succ[j]:
            rank[j] = w_bar[j]
        else:
            rank[j] = w_bar[j] + max(c_bar.get((j, k), 0.0) + rank[k] for k in succ[j])
    return rank


# ─────────────────────────────────────────────────────────────
# HEFT Phase 2 — list scheduling
# ─────────────────────────────────────────────────────────────

class NodeQueue:
    """
    Non-preemptive task queue for a single node.
    Tracks finish times of assigned tasks to determine earliest start.
    """
    def __init__(self, nid: str):
        self.nid = nid
        self.available_at: float = 0.0  # earliest time node is free

    def earliest_start(self, ready_time: float) -> float:
        """Earliest start for a new task given precedence constraint."""
        return max(self.available_at, ready_time)

    def assign(self, start: float, duration: float) -> float:
        finish = start + duration
        self.available_at = finish
        return finish


def heft_schedule(
    tasks: Dict[str, Task],
    nodes: Dict[str, Node],
    a: Dict[Tuple[str, str], int],
    p: Dict[Tuple[str, str], float],
    D: Dict[Tuple[str, str, str, str], float],
    rank: Dict[str, float],
) -> Tuple[Dict[str, str], Dict[str, float], Dict[str, float], float]:
    """
    HEFT list scheduling (Phase 2).

    Returns:
        task_to_node: mapping task→node
        start_time:   mapping task→start time
        finish_time:  mapping task→finish time
        makespan:     C_max
    """
    N = list(nodes.keys())

    # Sort tasks by rank_u descending
    sorted_tasks = sorted(tasks.keys(), key=lambda j: -rank[j])

    queues: Dict[str, NodeQueue] = {i: NodeQueue(i) for i in N}
    task_to_node: Dict[str, str] = {}
    start_time: Dict[str, float] = {}
    finish_time: Dict[str, float] = {}

    for j in sorted_tasks:
        best_eft = math.inf
        best_node: Optional[str] = None
        best_est: float = 0.0

        for i in N:
            if a[(i, j)] == 0:
                continue
            if not math.isfinite(p[(i, j)]):
                continue

            # Earliest time j can start on node i given predecessor constraints
            pred_ready = 0.0
            for jp in tasks[j].deps:
                if jp not in finish_time:
                    continue  # predecessor not yet scheduled (shouldn't happen in DAG)
                ip = task_to_node[jp]
                comm = D.get((ip, i, jp, j), 0.0)
                pred_ready = max(pred_ready, finish_time[jp] + comm)

            est = queues[i].earliest_start(pred_ready)
            eft = est + p[(i, j)]

            if eft < best_eft:
                best_eft = eft
                best_node = i
                best_est = est

        if best_node is None:
            raise RuntimeError(f"Task {j} has no feasible node — check compatibility matrix.")

        # Commit assignment
        fin = queues[best_node].assign(best_est, p[(best_node, j)])
        task_to_node[j] = best_node
        start_time[j] = best_est
        finish_time[j] = fin

    makespan = max(finish_time.values()) if finish_time else 0.0
    return task_to_node, start_time, finish_time, makespan


# ─────────────────────────────────────────────────────────────
# Main solve function
# ─────────────────────────────────────────────────────────────

def solve(
    workflow: str,
    nodes_path: str,
    alpha: float,
    beta: float,
    node_mem_unit: str,
    task_mem_unit: str,
    task_data_unit: str,
) -> dict:

    wname, tasks = load_workflow(workflow, task_mem_unit, task_data_unit)
    nodes = load_nodes(nodes_path, node_mem_unit)

    T = list(tasks.keys())
    N = list(nodes.keys())
    if not T or not N:
        raise ValueError("Empty tasks or nodes.")

    t0 = time.time()

    a = build_a(tasks, nodes)

    # Pre-screen feasibility
    infeasible_tasks = [j for j in T if sum(a[(i, j)] for i in N) == 0]
    if infeasible_tasks:
        return {
            "status": "infeasible",
            "solver": {"name": "HEFT", "wall_time_s": 0.0},
            "workflow": wname,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "reason": "feature_or_resource_mismatch",
            "infeasible_tasks": infeasible_tasks,
        }

    p, U, sk = build_p(tasks, nodes, a)
    D        = build_D(tasks, nodes)

    # Phase 1 — rank computation
    topo_order = topological_sort(tasks)
    succ       = build_succ(tasks)
    w_bar      = compute_average_computation(tasks, nodes, p, a)
    c_bar      = compute_average_communication(tasks, nodes, D)
    rank       = upward_rank(tasks, topo_order, succ, w_bar, c_bar)

    # Phase 2 — list scheduling
    task_to_node, start_time, finish_time, makespan = heft_schedule(
        tasks, nodes, a, p, D, rank
    )

    t1 = time.time()

    # Compute multi-objective value (same formula as MILP/CP-SAT)
    usage_val = sum(
        U.get((task_to_node[j], j), 0.0) for j in T if j in task_to_node
    )
    obj_val = alpha * usage_val + beta * makespan

    # Build schedule output (compatible with MILP solver format)
    edges = [(jp, j) for j in T for jp in tasks[j].deps if jp in tasks]
    schedule: Dict[str, Any] = {}
    node_util: Dict[str, int] = {i: 0 for i in N}
    for j in T:
        assigned = task_to_node.get(j)
        if assigned:
            node_util[assigned] += 1
        schedule[j] = {
            "node": assigned,
            "start_time": round(start_time.get(j, 0.0), 6),
            "finish_time": round(finish_time.get(j, 0.0), 6),
            "p_ij": round(p.get((assigned, j), float("nan")), 6) if assigned else float("nan"),
            "speed_key": sk.get((assigned, j)) if assigned else None,
            "rank_u": round(rank.get(j, 0.0), 4),
            "task_features": sorted(tasks[j].features),
            "task_cores": tasks[j].cores,
            "task_mem_mb": tasks[j].mem_mb,
            "task_data_mb": tasks[j].data_mb,
        }

    return {
        "status": "feasible",
        "workflow": wname,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "solver": {
            "name": "HEFT",
            "wall_time_s": round(t1 - t0, 6),
            "algorithm": "Heterogeneous Earliest Finish Time (Topcuoglu et al., 2002)",
        },
        "objective": {
            "alpha": alpha, "beta": beta,
            "value": round(obj_val, 6),
            "makespan": round(makespan, 6),
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
    ap = argparse.ArgumentParser(description="Feature-aware HEFT workflow scheduler")
    ap.add_argument("workflow", help="Workflow JSON (STG-style)")
    ap.add_argument("--nodes", required=True, help="Nodes JSON")
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--beta",  type=float, default=1.0)
    ap.add_argument("--node-mem-unit", choices=["MB", "GB"], default="MB")
    ap.add_argument("--task-mem-unit", choices=["MB", "GB"], default="MB")
    ap.add_argument("--task-data-unit", choices=["MB", "GB"], default="MB")
    ap.add_argument("-o", "--output", default="solution_heft.json")
    args = ap.parse_args()

    sol = solve(
        workflow=args.workflow,
        nodes_path=args.nodes,
        alpha=args.alpha,
        beta=args.beta,
        node_mem_unit=args.node_mem_unit,
        task_mem_unit=args.task_mem_unit,
        task_data_unit=args.task_data_unit,
    )

    out = Path(args.output)
    out = out.parent / f"{out.stem}_{ts_compact()}{out.suffix}"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(sol, indent=2))
    print(f"✓ Result written: {out}  (status={sol.get('status')})")
    if sol.get("status") in {"feasible", "optimal"}:
        obj = sol.get("objective", {})
        print(f"  makespan={obj.get('makespan'):.4f}  usage={obj.get('usage_term'):.4f}"
              f"  obj={obj.get('value'):.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
