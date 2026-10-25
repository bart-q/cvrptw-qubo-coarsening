# Graph Coarsening and Adaptive QUBO Penalties for the CVRPTW

Code for solving the Capacitated Vehicle Routing Problem with Time Windows
(CVRPTW) as a Quadratic Unconstrained Binary Optimization (QUBO) problem on a
simulated-annealing backend, with two contributions:

1. **Adaptive penalty calibration.** Uniformly scaling QUBO penalties leaves
   simulated annealing unchanged; the effective lever is the QUBO's internal
   dynamic range, which is dominated by capacity log-slack couplings that are
   non-binding on most small instances. Dropping those and scaling the rest
   reduces raw constraint violations by roughly an order of magnitude at equal
   solver budget.
2. **Learned, tuning-free coarsening.** A GraphSAGE merge scorer replaces the
   hand-tuned spatio-temporal score with a single fixed configuration for all
   instance families. It preserves feasibility better than the per-family-tuned
   heuristic across `N = 10..100`, tuning-free, and keeps the QUBO small enough to
   stay solvable and embeddable at large `N`.

All experiments run on a classical simulated-annealing sampler and OR-Tools
serves as a classical reference; no physical quantum hardware is used.

## Computational audit

An independent computational audit of the reported GNN results is available in [review/README.md](review/README.md).

## Installation

```bash
python -m pip install -r requirements.txt
```

Install a `torch` build matching your CUDA/CPU before `torch-geometric`. The
GPU is used only for training the GNN; all solving runs on CPU.

## Layout

```
src/          core library (datasets, coarsening, QUBO, solvers, GNN, metrics)
scripts/      experiment drivers and reproduction runners
configs/      YAML configs for the baseline and penalty-ablation runs
data/         Solomon (CVRPTW) and Christofides (CVRP) benchmark instances
results/      output directory; ships the trained policy gnn_model_rl.pt
```

## Reproducing the results

```bash
# baseline reproduction and penalty ablation
python scripts/run_phase1.py
python scripts/run_phase2_full.py

# GNN: generate labels, train, evaluate
python scripts/gen_labels.py
python scripts/train_gnn.py
python scripts/run_phase4_full.py

# controls and scaling
python scripts/run_confound.py         # dynamic-range vs. variable-count control
python scripts/run_scorer_ablation.py  # learned vs. fixed merge score
python scripts/run_repair_only.py       # classical repair baseline
python scripts/run_local_search.py      # shared local-search polish
python scripts/run_scaling_large.py     # QUBO size and feasibility vs. N
```

A pre-trained coarsening policy is provided at `results/gnn_model_rl.pt`, so the
evaluation and scaling scripts can be run without retraining.

## Solver backend

The default backend is D-Wave's `SimulatedAnnealingSampler` from
`dwave-samplers`. `scripts/run_qpu.py` contains a hardware path (via the D-Wave
Leap framework) with a dry-run mode that validates the pipeline on the simulator;
running it on real hardware requires Leap access.
