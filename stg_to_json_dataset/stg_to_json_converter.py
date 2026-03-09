#!/usr/bin/env python3
"""
stg_converter.py — Convert Standard Task Graph (.stg / STG_dt) files into your MILP workflow JSON format.

Key goals
- Parse both STG (no comm costs) and STG_dt (with comm costs), including the "pair-per-line" variant.
- Produce workflow JSON compatible with milp_solver_revised.py:
    {"meta": {...}, "tasks": {"T1": {...}, ...}}
  where each task includes:
    duration, cores, memory_required, features, dependencies, data

- Generate two workload variants depending on a given system configuration:
    * Homogeneous workload: all tasks require CPU ("cpu")
    * Heterogeneous workload: tasks require CPU by default, with optional GPU fraction ("gpu") if the system provides GPU nodes.

Usage examples
--------------
Single file:
  python stg_converter.py \
      --input /path/to/rand0000.stg \
      --system /path/to/homogeneous_nodes_scc_mb.json \
      --output-dir ./out

Batch from tarball:
  python stg_converter.py \
      --input /path/to/rnc50.tgz \
      --system /path/to/heterogeneous_nodes_scc_mb.json \
      --output-dir ./out \
      --gpu-fraction 0.2

Notes
-----
- Units:
  * STG processing times are in arbitrary "u.t." (units of time).
  * This converter maps them to "duration" via an auto-scaling heuristic unless you override --duration-scale.
  * Communication costs (u.t.) are mapped to data volume (MB) using --comm-to-data-scale.
    For STG without comm costs, data is synthesized from duration via --data-per-time.

- The converter clamps generated per-task cores/memory to fit within the provided system configuration.
  This reduces infeasibility artifacts when running your MILP baseline.

Author: (generated) — project-quality, defensive parsing, reproducible mapping.
"""
from __future__ import annotations

import argparse
import io
import json
import math
import random
import re
import sys
import tarfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union


# ----------------------------
# Utilities
# ----------------------------

_INT_RE = re.compile(r"^-?\d+$")


def _tokenize(line: str) -> List[str]:
    # STG columns are fixed width but whitespace split works.
    # Some copies contain literal "..." tokens; drop them.
    toks = [t for t in line.strip().split() if t != "..."]
    return toks


def _is_int(tok: str) -> bool:
    return bool(_INT_RE.match(tok))


def _safe_int(tok: str, where: str) -> int:
    if not _is_int(tok):
        raise ValueError(f"Expected int token at {where}, got {tok!r}")
    return int(tok)


def _now_iso() -> str:
    return datetime.now().replace(microsecond=0).isoformat()


def _load_json(path: Union[str, Path]) -> dict:
    return json.loads(Path(path).read_text())


def _write_json(path: Union[str, Path], obj: dict) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2, sort_keys=False))


# ----------------------------
# System model helpers (from nodes JSON)
# ----------------------------

@dataclass(frozen=True)
class SystemCaps:
    max_cores: int
    max_mem_mb: int
    has_gpu: bool
    compute_features: Tuple[str, ...]  # e.g., ("cpu","gpu")


def infer_system_caps(nodes_json: dict) -> SystemCaps:
    nodes = nodes_json.get("nodes", {})
    if not isinstance(nodes, dict) or not nodes:
        raise ValueError("System JSON must contain a non-empty 'nodes' object.")

    max_cores = 1
    max_mem = 1
    has_gpu = False
    for n in nodes.values():
        max_cores = max(max_cores, int(n.get("cores", 1)))
        max_mem = max(max_mem, int(n.get("memory", 1)))
        feats = set((n.get("features") or []))
        if "gpu" in feats:
            has_gpu = True

    compute_features = ("cpu", "gpu") if has_gpu else ("cpu",)
    return SystemCaps(max_cores=max_cores, max_mem_mb=max_mem, has_gpu=has_gpu, compute_features=compute_features)


# ----------------------------
# STG parsing
# ----------------------------

@dataclass
class StgTask:
    tid: int              # numeric id as in STG (0..n+1)
    proc_time: float      # computation time (u.t.)
    preds: List[int]      # predecessor ids
    comm_from_pred: Dict[int, float]  # pred -> comm cost (u.t.) if present, else empty


@dataclass
class StgGraph:
    n_tasks: int                  # number of *real* tasks (excluding dummies), as in header
    tasks: Dict[int, StgTask]     # includes dummy nodes 0 and n+1 if present
    info: Dict[str, Union[str, int, float]]  # parsed from comment part
    has_comm_costs: bool


def parse_stg_bytes(data: bytes, source_name: str = "<bytes>") -> StgGraph:
    """
    Parse an STG file content.

    Supports:
      - STG without communication costs: each task line includes [tid, proc, k, preds...]
      - STG_dt with communication costs:
          a) inline pairs: [tid, proc, k, pred1, comm1, pred2, comm2, ...]
          b) pair-per-line variant: main line ends at k; then k lines follow, each "pred comm"

    Tolerates occasional "..." tokens.
    """
    text = data.decode("utf-8", errors="replace")
    lines = text.splitlines()

    # Split task-graph part vs information part
    graph_lines: List[str] = []
    info_lines: List[str] = []
    in_info = False
    for ln in lines:
        if ln.lstrip().startswith("#"):
            in_info = True
        if in_info:
            info_lines.append(ln.rstrip("\n"))
        else:
            if ln.strip():
                graph_lines.append(ln.rstrip("\n"))

    if not graph_lines:
        raise ValueError(f"{source_name}: empty or invalid STG (no graph part).")

    # Header: number of tasks (real, excluding dummies)
    header_toks = _tokenize(graph_lines[0])
    if not header_toks:
        raise ValueError(f"{source_name}: missing header task count.")
    n = _safe_int(header_toks[0], f"{source_name}:header")
    if n <= 0:
        raise ValueError(f"{source_name}: header task count must be positive; got {n}.")

    # Parse sequentially after header
    idx = 1
    tasks: Dict[int, StgTask] = {}
    has_comm = False

    def next_nonempty_line() -> str:
        nonlocal idx
        while idx < len(graph_lines) and not graph_lines[idx].strip():
            idx += 1
        if idx >= len(graph_lines):
            raise ValueError(f"{source_name}: unexpected EOF while parsing graph part.")
        ln = graph_lines[idx]
        idx += 1
        return ln

    # We expect nodes 0..n+1 inclusive (n+2 nodes), but be defensive:
    # stop when we parsed >= n+2 unique tids OR when we run out of graph lines.
    target_nodes = n + 2

    while len(tasks) < target_nodes and idx <= len(graph_lines):
        ln = next_nonempty_line()
        toks = _tokenize(ln)
        if not toks:
            continue

        # Main line must start with tid proc k
        if len(toks) < 3:
            raise ValueError(f"{source_name}: invalid task line (need >=3 tokens): {ln!r}")
        tid = _safe_int(toks[0], f"{source_name}:tid")
        proc = float(_safe_int(toks[1], f"{source_name}:proc_time"))
        k = _safe_int(toks[2], f"{source_name}:num_pred")
        rest = toks[3:]

        preds: List[int] = []
        comm: Dict[int, float] = {}

        if k < 0:
            raise ValueError(f"{source_name}: negative predecessor count for tid={tid}.")
        if k == 0:
            # nothing to read
            pass
        else:
            # Three supported layouts:
            # 1) no-comm: rest has >=k ints => preds are first k
            # 2) inline-comm: rest has >=2k ints => interpret as pairs if it "looks like" pairs
            # 3) per-line pairs: rest has <k => read k additional lines
            #
            # Heuristic for inline-comm:
            #   if len(rest) >= 2k and after pairing, both pred ids and comm costs are ints
            #   (comm costs are ints in STG_dt examples), then treat as comm.
            if len(rest) >= 2 * k and all(_is_int(x) for x in rest[:2 * k]):
                # inline pairs
                has_comm = True
                for pi in range(k):
                    pred = int(rest[2 * pi])
                    c = float(int(rest[2 * pi + 1]))
                    preds.append(pred)
                    comm[pred] = c
            elif len(rest) >= k and all(_is_int(x) for x in rest[:k]):
                # no comm costs
                preds = [int(x) for x in rest[:k]]
            else:
                # pair-per-line variant: read k lines, each containing pred (and maybe comm)
                # allow "pred comm" or just "pred" (rare)
                for _ in range(k):
                    ln2 = next_nonempty_line()
                    toks2 = _tokenize(ln2)
                    toks2 = [t for t in toks2 if _is_int(t)]
                    if not toks2:
                        raise ValueError(f"{source_name}: missing predecessor entry for tid={tid}.")
                    pred = int(toks2[0])
                    preds.append(pred)
                    if len(toks2) >= 2:
                        has_comm = True
                        comm[pred] = float(int(toks2[1]))

        tasks[tid] = StgTask(tid=tid, proc_time=proc, preds=preds, comm_from_pred=comm)

        # guard to avoid infinite loop on malformed input
        if idx >= len(graph_lines) and len(tasks) < min(target_nodes, n):
            break

    info = parse_stg_info(info_lines)
    return StgGraph(n_tasks=n, tasks=tasks, info=info, has_comm_costs=has_comm)


def parse_stg_info(info_lines: Sequence[str]) -> Dict[str, Union[str, int, float]]:
    """
    Parse the '#' comment metadata into a dictionary.
    Keeps keys as normalized snake_case, values as str/int/float when possible.
    """
    info: Dict[str, Union[str, int, float]] = {}
    for ln in info_lines:
        s = ln.lstrip()
        if not s.startswith("#"):
            continue
        s = s[1:].strip()
        if not s:
            continue
        # Example: "CP Length           : 52"
        if ":" in s:
            k, v = s.split(":", 1)
            key = re.sub(r"[^a-zA-Z0-9]+", "_", k.strip()).strip("_").lower()
            val = v.strip()
            # pull "Real : x" pattern but keep full too
            # Try numeric conversion
            m_real = re.search(r"\(Real\s*:\s*([0-9.+-eE]+)\)", val)
            if m_real:
                try:
                    info[key + "_real"] = float(m_real.group(1))
                except Exception:
                    pass
                # also strip "(Real: ...)" for main value
                val = re.sub(r"\(Real\s*:\s*[0-9.+-eE]+\)", "", val).strip()

            # Try int/float
            if re.fullmatch(r"-?\d+", val):
                info[key] = int(val)
            elif re.fullmatch(r"-?\d+\.\d+", val) or re.fullmatch(r"-?\d+\.\d+(e[+-]?\d+)?", val, re.IGNORECASE):
                try:
                    info[key] = float(val)
                except Exception:
                    info[key] = val
            else:
                info[key] = val
        else:
            # free text
            key = "note_" + str(len(info) + 1)
            info[key] = s
    return info


# ----------------------------
# Mapping STG -> MILP workload JSON
# ----------------------------

def auto_duration_scale(proc_times: List[float], target_mean: float) -> float:
    # Robust: use trimmed mean to avoid extreme values.
    pts = sorted(proc_times)
    if not pts:
        return 1.0
    lo = int(0.05 * len(pts))
    hi = int(0.95 * len(pts))
    core = pts[lo:hi] if hi > lo else pts
    mean = sum(core) / max(1, len(core))
    if mean <= 0:
        return 1.0
    return target_mean / mean


def map_tasks(
    g: StgGraph,
    system: SystemCaps,
    *,
    variant: str,  # "homogeneous" or "heterogeneous"
    seed: int,
    gpu_fraction: float,
    duration_scale: Optional[float],
    target_mean_duration: float,
    max_task_cores: Optional[int],
    mem_per_core_mb: int,
    min_mem_mb: int,
    data_per_time_mb: float,
    comm_to_data_scale: float,
) -> Dict[str, dict]:
    """
    Produce tasks dict keyed by "T1".."Tn" excluding dummy nodes.

    Mapping policy:
      duration = proc_time * scale  (auto or overridden)
      cores    = 1..Cmax (proportional to duration, clamped by system caps)
      memory   = clamp(min_mem, mem_per_core_mb * cores, system max)
      data     = synthesized:
                 - if comm costs present: sum(comm_from_pred) * comm_to_data_scale
                 - else: duration * data_per_time_mb
      features = ["cpu"] always for homogeneous; for heterogeneous may include some ["gpu"]
    """
    rng = random.Random(seed)

    # Identify real tasks: 1..n
    real_ids = [i for i in range(1, g.n_tasks + 1) if i in g.tasks]
    if len(real_ids) != g.n_tasks:
        # be defensive: fall back to any non-dummy ids excluding 0 and max
        ids = sorted([tid for tid in g.tasks.keys() if tid not in (0, g.n_tasks + 1)])
        real_ids = ids[: g.n_tasks]

    proc_times = [float(g.tasks[i].proc_time) for i in real_ids]
    scale = duration_scale if duration_scale is not None else auto_duration_scale(proc_times, target_mean_duration)
    if scale <= 0:
        scale = 1.0

    # Core budget for tasks
    sys_max_cores = int(system.max_cores)
    task_core_cap = min(sys_max_cores, int(max_task_cores) if max_task_cores else sys_max_cores)
    task_core_cap = max(1, task_core_cap)

    # Build normalized "work" signal from duration
    durations = [pt * scale for pt in proc_times]
    dmax = max(durations) if durations else 1.0

    # Feature assignment
    can_gpu = system.has_gpu and variant == "heterogeneous" and gpu_fraction > 0.0
    gpu_fraction = max(0.0, min(1.0, gpu_fraction))
    gpu_ids = set()
    if can_gpu:
        k = int(round(gpu_fraction * len(real_ids)))
        # Prefer assigning GPU to heavier tasks (longer durations) to emphasize heterogeneity.
        ranked = sorted(real_ids, key=lambda tid: g.tasks[tid].proc_time, reverse=True)
        gpu_ids = set(ranked[:k])

    tasks_out: Dict[str, dict] = {}

    for tid_num in real_ids:
        t = g.tasks[tid_num]
        duration = float(t.proc_time) * scale
        # cores proportional to duration (log-ish to avoid too aggressive scaling)
        rel = duration / dmax if dmax > 0 else 0.0
        cores = 1 + int(round(rel * (task_core_cap - 1)))
        cores = max(1, min(task_core_cap, cores))

        mem = max(min_mem_mb, mem_per_core_mb * cores)
        mem = min(mem, int(system.max_mem_mb))

        # deps: exclude entry dummy 0
        deps = [f"T{p}" for p in t.preds if p != 0 and 1 <= p <= g.n_tasks]

        # data: MB
        if g.has_comm_costs and t.comm_from_pred:
            comm_sum = sum(float(c) for c in t.comm_from_pred.values())
            data_mb = comm_sum * comm_to_data_scale
        else:
            data_mb = duration * data_per_time_mb

        # keep within a reasonable positive range
        data_mb = max(0.0, float(data_mb))

        # features
        if variant == "homogeneous":
            feats = ["cpu"]
        else:
            feats = ["gpu"] if tid_num in gpu_ids else ["cpu"]

        tasks_out[f"T{tid_num}"] = {
            "cores": int(cores),
            "memory_required": int(mem),
            "features": feats,
            "data": float(round(data_mb, 6)),
            "duration": float(round(duration, 6)),
            "dependencies": deps,
            # tags kept for forward compatibility with your pipeline
            "tags": [],
        }

    return tasks_out


def build_workload_json(
    *,
    tasks: Dict[str, dict],
    system_path: str,
    source_name: str,
    variant: str,
    stg_info: dict,
    mapping_meta: dict,
) -> dict:
    return {
        "meta": {
            "generated_at": _now_iso(),
            "source": source_name,
            "system_config": str(system_path),
            "variant": variant,
            "unit_memory_required": "MB",
            "unit_data": "MB",
            "stg_info": stg_info,
            "mapping": mapping_meta,
        },
        "tasks": tasks,
    }


# ----------------------------
# IO: inputs may be .stg file, directory, or .tgz
# ----------------------------

def iter_stg_sources(input_path: Union[str, Path]) -> Iterable[Tuple[str, bytes]]:
    """
    Yield (source_name, bytes) for each STG file found.
    Supports:
      - single .stg file
      - directory: all *.stg under it (recursive)
      - tar/tgz: all members ending with .stg
    """
    p = Path(input_path)
    if not p.exists():
        raise FileNotFoundError(str(p))

    if p.is_file() and p.suffix.lower() in {".stg"}:
        yield (p.name, p.read_bytes())
        return

    if p.is_dir():
        for f in sorted(p.rglob("*.stg")):
            yield (str(f.relative_to(p)), f.read_bytes())
        return

    # tarball
    if p.is_file() and (p.suffix.lower() in {".tgz", ".tar", ".gz"} or p.name.lower().endswith(".tar.gz")):
        mode = "r:gz" if (p.suffix.lower() in {".tgz", ".gz"} or p.name.lower().endswith(".tar.gz")) else "r"
        with tarfile.open(p, mode) as tf:
            for m in tf.getmembers():
                if not m.isfile():
                    continue
                if not m.name.lower().endswith(".stg"):
                    continue
                f = tf.extractfile(m)
                if f is None:
                    continue
                yield (m.name, f.read())
        return

    raise ValueError(f"Unsupported input: {input_path} (expected .stg, directory, or .tgz/.tar.gz)")


# ----------------------------
# CLI
# ----------------------------

def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Convert STG (.stg / STG_dt) into MILP workflow JSON for your scheduler.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--input", required=True, help="Path to a .stg file, a directory containing .stg files, or a .tgz/.tar.gz archive.")
    ap.add_argument("--system", required=True, help="Path to system configuration JSON (homogeneous_nodes_*.json or heterogeneous_nodes_*.json).")
    ap.add_argument("--output-dir", required=True, help="Directory to write converted workload JSON files.")

    ap.add_argument("--variant", choices=["auto", "homogeneous", "heterogeneous"], default="auto",
                    help="Workload variant. 'auto' uses the system config: GPU nodes -> heterogeneous, otherwise homogeneous.")
    ap.add_argument("--seed", type=int, default=12345, help="Random seed for deterministic feature assignment and any randomized mapping.")

    # Duration mapping
    ap.add_argument("--duration-scale", type=float, default=None,
                    help="Override duration scale (duration = proc_time * scale). If omitted, scale is chosen to match --target-mean-duration.")
    ap.add_argument("--target-mean-duration", type=float, default=10.0,
                    help="Target mean duration (in your model units) used to auto-scale STG proc times when --duration-scale is not provided.")

    # Resource synthesis
    ap.add_argument("--max-task-cores", type=int, default=None, help="Upper bound for per-task cores (also capped by system max cores).")
    ap.add_argument("--mem-per-core-mb", type=int, default=2048, help="Memory required per core (MB).")
    ap.add_argument("--min-mem-mb", type=int, default=1024, help="Minimum memory required per task (MB).")

    # Data synthesis
    ap.add_argument("--data-per-time", type=float, default=1024.0,
                    help="If STG has no comm costs, data = duration * data_per_time (MB per time unit).")
    ap.add_argument("--comm-to-data-scale", type=float, default=1.0,
                    help="If STG has comm costs (u.t.), convert them to data MB as: data = sum(comm_costs) * comm_to_data_scale.")

    # Heterogeneity policy
    ap.add_argument("--gpu-fraction", type=float, default=0.2,
                    help="For heterogeneous variant on GPU-capable systems: fraction of tasks that require GPU. Applied to largest tasks by proc time.")

    ap.add_argument("--suffix", default=None,
                    help="Optional output filename suffix (e.g., '_stg'). Default uses '_homo' or '_hetero' based on variant.")
    ap.add_argument("--limit", type=int, default=None, help="Convert at most N STG files (debugging).")
    return ap


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = build_argparser()
    args = ap.parse_args(argv)

    system_json = _load_json(args.system)
    caps = infer_system_caps(system_json)

    if args.variant == "auto":
        variant = "heterogeneous" if caps.has_gpu else "homogeneous"
    else:
        variant = args.variant

    if variant == "homogeneous" and caps.has_gpu:
        # homogeneous workload is still valid on heterogeneous systems, but warn.
        pass

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    converted = 0
    for source_name, b in iter_stg_sources(args.input):
        g = parse_stg_bytes(b, source_name)

        tasks = map_tasks(
            g,
            caps,
            variant=variant,
            seed=args.seed,
            gpu_fraction=args.gpu_fraction,
            duration_scale=args.duration_scale,
            target_mean_duration=args.target_mean_duration,
            max_task_cores=args.max_task_cores,
            mem_per_core_mb=args.mem_per_core_mb,
            min_mem_mb=args.min_mem_mb,
            data_per_time_mb=args.data_per_time,
            comm_to_data_scale=args.comm_to_data_scale,
        )

        mapping_meta = {
            "seed": args.seed,
            "duration_scale": args.duration_scale if args.duration_scale is not None else "auto",
            "target_mean_duration": args.target_mean_duration,
            "max_task_cores": args.max_task_cores,
            "mem_per_core_mb": args.mem_per_core_mb,
            "min_mem_mb": args.min_mem_mb,
            "data_per_time_mb": args.data_per_time,
            "comm_to_data_scale": args.comm_to_data_scale,
            "gpu_fraction": args.gpu_fraction if (variant == "heterogeneous" and caps.has_gpu) else 0.0,
        }

        out_json = build_workload_json(
            tasks=tasks,
            system_path=args.system,
            source_name=source_name,
            variant=variant,
            stg_info=g.info,
            mapping_meta=mapping_meta,
        )

        base = Path(source_name).stem
        suffix = args.suffix if args.suffix is not None else ("_homo" if variant == "homogeneous" else "_hetero")
        out_path = out_dir / f"{base}{suffix}.json"
        _write_json(out_path, out_json)
        converted += 1

        if args.limit and converted >= args.limit:
            break

    print(f"[stg_converter] Converted {converted} STG file(s) -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
