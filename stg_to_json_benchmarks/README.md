# Workflow Scheduling Benchmarks

This directory contains implementations and experiment scripts for evaluating workflow scheduling algorithms using the **STG-JSON dataset**.

The benchmarks support multiple classes of scheduling approaches commonly studied in HPC workflow research.

---

## Implemented Scheduling Methods

### Optimization-Based Methods

Mixed Integer Linear Programming (MILP)

A mathematical optimization model that computes optimal task-to-node assignments and schedules subject to resource constraints and task dependencies.

Constraint Programming (CP-SAT)

A constraint optimization approach capable of solving large combinatorial scheduling problems.

---

### Heuristic Scheduling

HEFT (Heterogeneous Earliest Finish Time)

A widely used heuristic for scheduling DAG workflows on heterogeneous computing systems.

---

### Learning-Based Scheduling

Graph Neural Network + Reinforcement Learning (GNN-RL)

A learning-based scheduling approach that models workflow graphs using graph neural networks and learns scheduling policies through reinforcement learning.

---

## Benchmark Objectives

The benchmark framework allows systematic evaluation of scheduling methods across:

- different workflow sizes
- heterogeneous system configurations
- varying workload characteristics

Performance metrics may include:

- workflow makespan
- resource utilization
- scheduling overhead
- scalability across graph sizes

---

## Reproducible Experiments

The scripts in this directory allow researchers to reproduce scheduling experiments and evaluate alternative algorithms using the same workflow dataset and system configurations.