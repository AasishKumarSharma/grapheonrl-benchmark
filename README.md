# STG-JSON Benchmark Suite for Workflow Scheduling in Heterogeneous HPC Systems

This repository provides tools, datasets, and benchmark implementations for research on **workflow mapping and scheduling in heterogeneous High Performance Computing (HPC) systems**.

The project extends the widely used **Standard Task Graph (STG) benchmark suite** developed by Hiroshi Kasahara and collaborators at Waseda University by converting STG task graphs into **structured JSON workflow representations** suitable for modern workflow scheduling frameworks.

The repository is organized into two primary components:

1. **Dataset Conversion Tools** – scripts and system configurations used to convert STG graphs into structured JSON workflow descriptions.
2. **Benchmark Implementations** – implementations and experiment scripts for evaluating scheduling algorithms.

The dataset itself is archived on Zenodo to ensure **long-term preservation, reproducibility, and citability**.

Dataset DOI:
https://doi.org/10.5281/zenodo.18927122

---

## Repository Structure

```
.
├── dataset/
│   ├── converter scripts
│   └── system configurations
│
├── benchmarks/
│   ├── MILP scheduling
│   ├── CP-SAT scheduling
│   ├── HEFT heuristic scheduler
│   └── GNN-RL learning-based scheduler
```

Each directory contains its own documentation describing the methodology, scripts, and experiment setup.

---

## Dataset Overview

The dataset provides structured JSON workflow descriptions derived from the **Standard Task Graph (STG)** benchmark suite.

Graph families include:

- rnc50
- rnc100
- rnc200
- rnc300
- larger variants planned for future releases

Each graph family contains **180 randomly generated directed acyclic graphs (DAGs)**.

The JSON representation augments the original STG graphs with:

- execution durations
- CPU core requirements
- memory requirements
- data transfer volumes
- heterogeneous processing features (CPU/GPU)
- explicit dependency lists

These additions enable reproducible benchmarking of modern scheduling algorithms.

---

## Research Applications

The benchmark suite supports research in:

- heterogeneous HPC workflow scheduling
- optimization-based scheduling algorithms
- heuristic scheduling strategies
- machine learning based schedulers
- reinforcement learning scheduling agents

---

## Citation

If you use this dataset or repository in your research, please cite:

@dataset{sharma2026stgjson,
  author       = {Aasish Kumar Sharma},
  title        = {Standard Task Graph (STG) Dataset with JSON Conversions for Workflow Scheduling in Heterogeneous HPC Systems},
  year         = {2026},
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.18927122}
}

---

## Dataset Provenance

The dataset is derived from the **Standard Task Graph (STG) benchmark suite**:

Prof. Dr. Hiroshi Kasahara, Waseda University  
https://www.kasahara.cs.waseda.ac.jp/schedule/stgarc_e.html
