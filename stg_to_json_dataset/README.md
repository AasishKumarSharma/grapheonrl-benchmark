# STG to JSON Dataset Conversion

This directory contains tools and configuration files used to convert the **Standard Task Graph (STG)** benchmark suite into structured JSON workflow representations.

The conversion enables STG workflows to be used in modern workflow scheduling experiments and machine learning based schedulers.

Dataset DOI:
https://doi.org/10.5281/zenodo.18927122

---

## Purpose

The original STG benchmark provides task graphs containing:

- task execution times
- dependency relationships
- structural DAG characteristics

However, modern scheduling research requires richer workflow representations including:

- resource requirements
- heterogeneous hardware features
- memory constraints
- data transfer volumes

The conversion scripts in this directory generate **JSON workflow descriptions** that extend the original STG graphs with these attributes.

---

## Conversion Pipeline

The dataset generation process follows these steps:

1. Parse STG graph files to extract tasks and dependencies.
2. Map processing times to workflow execution durations.
3. Assign CPU core requirements and memory demands.
4. Generate data transfer volumes between dependent tasks.
5. Assign heterogeneous execution features (CPU or GPU).
6. Generate workflow metadata and statistics.

---

## System Configurations

Two representative HPC system configurations are included:

- homogeneous HPC clusters
- heterogeneous CPU–GPU systems

These configurations provide node descriptions including:

- available CPU cores
- memory capacity
- storage
- processing features
- network characteristics

---

## Output Format

Each workflow is stored as a JSON file with the following structure:

```
{
  "meta": {...},
  "tasks": {
      "T1": {...},
      "T2": {...}
  }
}
```

Task descriptions include:

- execution duration
- CPU requirements
- memory requirements
- data volumes
- dependency relationships

---

## Reproducibility

All scripts required to regenerate the dataset are included in this directory to support reproducible workflow scheduling experiments.
