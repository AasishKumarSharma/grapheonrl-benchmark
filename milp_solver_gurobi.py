#!/usr/bin/env python3
"""
milp_solver_gurobi.py — Feature-aware MILP baseline for workflow scheduling (Gurobi)

Equivalent in intent to the user's PuLP+CBC implementation:
min  alpha * Σ_j Σ_i U_ij x_ij  +  beta * C_max

s.t.
(1)  Σ_i x_ij = 1                                    ∀ j
(2)  x_ij <= a_ij                                    ∀ i,j
(3)  f_j >= s_j + Σ_i p_ij x_ij                      ∀ j
(4)  C_max >= f_j                                    ∀ j
(5)  s_j >= f_j' + Σ_{i',i} D_{i',i}^{(j',j)} y_{i',i}^{(j',j)}     ∀ (j',j) in δ
(6)  y linearization:
     y <= x_pred,  y <= x_succ,  y >= x_pred + x_succ - 1

Notes:
- Keeps the same JSON conventions as the PuLP file.
- Returns "infeasible" instead of throwing for feature/resource mismatch.
- Uses gurobipy.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Set, Any

import gurobipy as gp
from gurobipy import GRB


DEFAULT_TASK = {
    "cores": 1,
    "memory_mb": 1024,
    "features": ["cpu"],
    "data_mb": 0,
    "duration": 10.0,
}

GPU_MARKERS = {
    "gpu", "cuda", "opencl",
    "v100", "a100", "h100", "rtx", "tesla",
    "nvidia_v100", "nvidia_a100", "nvidia_h100",
}


def ts_compact() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def norm_token(x: str) -> str:
    x = str(x).strip().lower()
    x = re.sub(r"[^a-z0-9]+", "_", x)
    x = re.sub(r"_+", "_", x).strip("_")
    return x


def norm_feature_set(lst: Any) -> Set[str]:
    if not lst:
        return set()
    if isinstance(lst, (set, tuple, list)):
        return {norm_token(v) for v in lst}
    return {norm_token(lst)}


def infer_device(task_features: Set[str]) -> str:
    return "GPU" if (task_features & GPU_MARKERS) else "CPU"


def to_mb(value: float, unit: str) -> float:
    unit = unit.strip().upper()
    if unit == "MB":
        return float(value)
    if unit == "GB":
        return float(value) * 1024.0
    raise ValueError(f"Unsupported unit: {unit} (use MB or GB)")


def node_net_rate_mb_s(node: dict, default: float = 1000.0) -> float:
    rates = node.get("data_transfer_rate", {}) or {}
    if not isinstance(rates, dict) or not rates:
        return default
    vals = []
    for v in rates.values():
        try:
            vals.append(float(v))
        except Exception:
            pass
    return max(vals) if vals else default


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
        duration = float(t.get("duration", DEFAULT_TASK["duration"]))
        cores = float(t.get("cores", DEFAULT_TASK["cores"]))
        mem_mb = to_mb(float(t.get("memory_required", DEFAULT_TASK["memory_mb"])), task_mem_unit)
        feats = norm_feature_set(t.get("features", DEFAULT_TASK["features"]))
        deps = list(t.get("dependencies", [])) if t.get("dependencies") else []
        data_mb = to_mb(float(t.get("data", DEFAULT_TASK["data_mb"])), task_data_unit)

        tasks[tid] = Task(
            tid=tid,
            duration=duration,
            cores=cores,
            mem_mb=mem_mb,
            features=feats,
            deps=deps,
            data_mb=data_mb,
        )
    return wname, tasks


def load_nodes(path: str, node_mem_unit: str) -> Dict[str, Node]:
    nd = json.loads(Path(path).read_text())
    raw_nodes = nd.get("nodes", nd)
    if not isinstance(raw_nodes, dict):
        raise ValueError("Nodes JSON must be an object or contain 'nodes' object.")

    nodes: Dict[str, Node] = {}
    for nid, n in raw_nodes.items():
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


def node_has_gpu(node: Node) -> bool:
    if "gpu" in node.features:
        return True
    ks = {norm_token(k) for k in (node.processing_speed or {}).keys()}
    return "gpu" in ks


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
    chosen_key: Dict[Tuple[str, str], str] = {}

    for j, t in tasks.items():
        for i, n in nodes.items():
            if a[(i, j)] == 0:
                p[(i, j)] = 1e12
                U[(i, j)] = 1e12
                chosen_key[(i, j)] = "infeasible"
                continue

            speed, key = pick_speed(t.features, n.processing_speed)
            chosen_key[(i, j)] = str(key)
            p_ij = max(1e-6, t.duration / speed)
            p[(i, j)] = p_ij

            mem_gb = t.mem_mb / 1024.0
            U[(i, j)] = (t.cores * p_ij) + (mem_weight * mem_gb * p_ij)

    return p, U, chosen_key


def build_D(tasks: Dict[str, Task], nodes: Dict[str, Node]) -> Dict[Tuple[str, str, str, str], float]:
    D: Dict[Tuple[str, str, str, str], float] = {}
    net = {
        nid: node_net_rate_mb_s({"data_transfer_rate": nodes[nid].data_transfer_rate})
        for nid in nodes.keys()
    }

    for j, t in tasks.items():
        for jp in t.deps:
            if jp not in tasks:
                continue
            data_mb = max(0.0, tasks[jp].data_mb)
            for ip in nodes.keys():
                for i in nodes.keys():
                    if ip == i:
                        D[(ip, i, jp, j)] = 0.0
                    else:
                        bw = max(1e-9, min(net[ip], net[i]))
                        D[(ip, i, jp, j)] = data_mb / bw
    return D


def explain_infeasible(tasks: Dict[str, Task], nodes: Dict[str, Node]) -> Dict[str, Any]:
    report: Dict[str, Any] = {}
    node_list = list(nodes.items())[:10]
    for tid, t in tasks.items():
        reasons = []
        dev = infer_device(t.features)
        for nid, n in node_list:
            feat_missing = sorted(list(t.features - n.features))
            if dev == "GPU" and not node_has_gpu(n):
                reasons.append({"node": nid, "reason": "gpu_required_but_node_has_no_gpu"})
                continue
            if feat_missing:
                reasons.append({"node": nid, "reason": "missing_features", "missing": feat_missing})
                continue
            if t.cores > n.cores:
                reasons.append({"node": nid, "reason": "insufficient_cores", "task": t.cores, "node": n.cores})
                continue
            if t.mem_mb > n.mem_mb:
                reasons.append({"node": nid, "reason": "insufficient_memory_mb", "task": t.mem_mb, "node": n.mem_mb})
                continue
        report[tid] = {
            "task_features": sorted(t.features),
            "cores": t.cores,
            "mem_mb": t.mem_mb,
            "device": dev,
            "sample_reasons": reasons[:10],
        }
    return report


def status_to_text(status_code: int) -> str:
    mapping = {
        GRB.OPTIMAL: "optimal",
        GRB.INFEASIBLE: "infeasible",
        GRB.TIME_LIMIT: "time_limit",
        GRB.INTERRUPTED: "interrupted",
        GRB.UNBOUNDED: "unbounded",
        GRB.INF_OR_UNBD: "inf_or_unbd",
    }
    return mapping.get(status_code, f"status_{status_code}")


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
    msg: bool,
    threads: int | None,
    mip_gap: float | None,
    mip_focus: int | None,
) -> dict:
    wname, tasks = load_workflow(workflow, task_mem_unit=task_mem_unit, task_data_unit=task_data_unit)
    nodes = load_nodes(nodes_path, node_mem_unit=node_mem_unit)

    T = list(tasks.keys())
    N = list(nodes.keys())
    if not T or not N:
        raise ValueError("Empty tasks or nodes.")

    a = build_a(tasks, nodes)

    infeasible_tasks = [j for j in T if sum(a[(i, j)] for i in N) == 0]
    if infeasible_tasks:
        diag = explain_infeasible({j: tasks[j] for j in infeasible_tasks}, nodes)
        return {
            "status": "infeasible",
            "workflow": wname,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "reason": "feature_or_resource_mismatch",
            "infeasible_tasks": infeasible_tasks,
            "diagnostic": diag,
            "counts": {"num_tasks": len(T), "num_nodes": len(N)},
            "solver": {"name": "Gurobi", "time_limit_s": time_limit, "wall_time_s": 0.0},
        }

    p, U, speed_key = build_p_U(tasks, nodes, a, mem_weight=mem_weight)
    D = build_D(tasks, nodes)
    edges: List[Tuple[str, str]] = [(jp, j) for j in T for jp in tasks[j].deps if jp in tasks]

    model = gp.Model("MILP_Workflow_Scheduling")

    model.Params.OutputFlag = 1 if msg else 0
    model.Params.TimeLimit = time_limit
    if threads is not None:
        model.Params.Threads = threads
    if mip_gap is not None:
        model.Params.MIPGap = mip_gap
    if mip_focus is not None:
        model.Params.MIPFocus = mip_focus

    x = model.addVars(N, T, vtype=GRB.BINARY, name="x")

    y_index = [(ip, i, jp, j) for (jp, j) in edges for ip in N for i in N]
    y = model.addVars(y_index, vtype=GRB.BINARY, name="y")

    s = model.addVars(T, lb=0.0, vtype=GRB.CONTINUOUS, name="s")
    f = model.addVars(T, lb=0.0, vtype=GRB.CONTINUOUS, name="f")
    Cmax = model.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name="Cmax")

    model.setObjective(
        alpha * gp.quicksum(U[(i, j)] * x[i, j] for i in N for j in T) + beta * Cmax,
        GRB.MINIMIZE,
    )

    for j in T:
        model.addConstr(gp.quicksum(x[i, j] for i in N) == 1, name=f"assign_{j}")

    for i in N:
        for j in T:
            if a[(i, j)] == 0:
                model.addConstr(x[i, j] == 0, name=f"feas_{i}_{j}")

    for j in T:
        model.addConstr(
            f[j] >= s[j] + gp.quicksum(p[(i, j)] * x[i, j] for i in N),
            name=f"proc_{j}",
        )

    for j in T:
        model.addConstr(Cmax >= f[j], name=f"cmax_{j}")

    for (jp, j) in edges:
        model.addConstr(
            s[j] >= f[jp] + gp.quicksum(D[(ip, i, jp, j)] * y[(ip, i, jp, j)] for ip in N for i in N),
            name=f"prec_{jp}_{j}",
        )
        for ip in N:
            for i in N:
                model.addConstr(y[(ip, i, jp, j)] <= x[ip, jp], name=f"y1_{ip}_{i}_{jp}_{j}")
                model.addConstr(y[(ip, i, jp, j)] <= x[i, j], name=f"y2_{ip}_{i}_{jp}_{j}")
                model.addConstr(y[(ip, i, jp, j)] >= x[ip, jp] + x[i, j] - 1, name=f"y3_{ip}_{i}_{jp}_{j}")

    t0 = time.time()
    model.optimize()
    t1 = time.time()

    status = status_to_text(model.Status)

    has_solution = model.SolCount > 0
    makespan_val = float(Cmax.X) if has_solution else None
    usage_val = (
        float(sum(x[i, j].X * U[(i, j)] for i in N for j in T))
        if has_solution else None
    )
    obj_val = float(model.ObjVal) if has_solution else None

    schedule = {}
    node_util = {i: 0 for i in N}

    if has_solution:
        for j in T:
            assigned = None
            for i in N:
                if x[i, j].X > 0.5:
                    assigned = i
                    node_util[i] += 1
                    break
            schedule[j] = {
                "node": assigned,
                "start_time": float(s[j].X),
                "finish_time": float(f[j].X),
                "p_ij": float(p[(assigned, j)] if assigned else float("nan")),
                "speed_key": speed_key.get((assigned, j)) if assigned else None,
                "task_features": sorted(tasks[j].features),
                "task_cores": tasks[j].cores,
                "task_mem_mb": tasks[j].mem_mb,
                "task_data_mb": tasks[j].data_mb,
            }

    return {
        "status": status,
        "workflow": wname,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "solver": {
            "name": "Gurobi",
            "time_limit_s": time_limit,
            "wall_time_s": round(t1 - t0, 6),
            "status_code": model.Status,
            "sol_count": int(model.SolCount),
        },
        "objective": {
            "alpha": alpha,
            "beta": beta,
            "value": obj_val,
            "makespan": makespan_val,
            "usage_term": usage_val,
        },
        "counts": {"num_tasks": len(T), "num_nodes": len(N), "num_edges": len(edges)},
        "units": {"tasks_memory": "MB", "tasks_data": "MB", "nodes_memory": "MB", "time": "seconds (relative)"},
        "node_utilization": node_util,
        "schedule": schedule,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Feature-aware MILP workflow scheduler (Gurobi)")
    ap.add_argument("workflow", help="Workflow JSON (STG-style)")
    ap.add_argument("--nodes", required=True, help="Nodes JSON")
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--time-limit", type=int, default=300)

    ap.add_argument("--node-mem-unit", choices=["MB", "GB"], default="MB")
    ap.add_argument("--task-mem-unit", choices=["MB", "GB"], default="MB")
    ap.add_argument("--task-data-unit", choices=["MB", "GB"], default="MB")

    ap.add_argument("--mem-weight", type=float, default=0.0)
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--mip-gap", type=float, default=None)
    ap.add_argument("--mip-focus", type=int, choices=[0, 1, 2, 3], default=None)
    ap.add_argument("--msg", action="store_true")
    ap.add_argument("-o", "--output", default="solution.json")

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
        msg=args.msg,
        threads=args.threads,
        mip_gap=args.mip_gap,
        mip_focus=args.mip_focus,
    )

    out = Path(args.output)
    out = out.parent / f"{out.stem}_{ts_compact()}{out.suffix}"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(sol, indent=2))
    print(f"✓ Result written: {out}  (status={sol.get('status')})")

    if sol.get("status") in {"optimal"}:
        return 0
    if sol.get("status") == "infeasible":
        return 2
    if sol.get("solver", {}).get("sol_count", 0) > 0:
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
