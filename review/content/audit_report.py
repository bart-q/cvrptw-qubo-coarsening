"""Presentation of the GNN results audit: the report's analyses, built from a Verification
(audit_verify.py). Each analysis shows views of the verified tables, then the conclusion and the
limitation they support; the sentences are assembled from the verified values and the check
statuses, so they change with them. Nothing here runs the pipeline or recomputes a statistic.
Also here: the execution paths shown in the appendix, the origin of every table, the evidence of
every check, and stage_log, which keeps the generation output in a log file.
"""
from __future__ import annotations

import contextlib
import io
import shutil
import sys
from html import escape
from pathlib import Path

import matplotlib
import matplotlib.lines
import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import HTML, Markdown, display

from audit_verify import STATUSES, pct

ARM_LABEL = {"heuristic": "heuristic (family-tuned)",
             "heuristic_fixed": "heuristic-fixed (P=0.5, radiusCoeff=0.5)",
             "gnn": "GNN (released model)"}
ARM_STYLE = {"gnn": dict(color="#2a78d6", marker="o"),
             "heuristic": dict(color="#eb6834", marker="s"),
             "heuristic_fixed": dict(color="#1baf7a", marker="^")}
plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "#e4e4e0", "grid.linewidth": 0.6,
                     "axes.edgecolor": "#8a8984", "axes.titlesize": 11, "axes.labelsize": 10,
                     "legend.frameon": False, "font.size": 9.5})
pd.set_option("display.max_columns", 40)
pd.set_option("display.width", 200)
pd.set_option("display.max_colwidth", 90)

# --------------------------------------------------------------------------- #
# Execution paths presented in the notebook appendix.
# Each entry describes the released functions called, added measurements,
# or intentional changes made by a control variant.
# Source references identify functions in audit_probe.py or in the released
# repository at the pinned commit; released source text is recorded in results.json.
# Roles:
#   call       a released function or method, called by the probe
#   inside     an operation inside that released function
#   script     a released experiment script, run unchanged
#   probe      set-up of the experiment
#   measure    an added measurement (does not feed back into the pipeline)
#   change     an intentional change in a control variant
#   substitute an input replaced because the original is not published
# --------------------------------------------------------------------------- #
_STAGED_STEPS = [
    ["call", "`src.utils.set_global_seed(seed)`", ["probe", "staged_job", ["set_global_seed"]]],
    ["call", "`src.datasets.load_solomon(instance).truncate(N)`", ["probe", "staged_job", ["load_solomon", "truncate"]]],
    ["call", "`src.solvers.SolverConfig('sa', 200, 200, seed=seed)`", ["probe", "staged_job", ["SolverConfig"]]],
    ["call", "`src.coarsening.graph_from_instance(inst)`", ["probe", "make_coarsener", ["graph_from_instance"]]],
    ["probe", "coarsener: `SpatioTemporalGraphCoarsener` with the family settings (`heuristic`) or "
              "α = β = 1, P = 0.5, radiusCoeff = 0.5 (`heuristic_fixed`); `GNNCoarsener` with P = 0.5, "
              "radiusCoeff = 0.5 and the checkpoint (`gnn`). Each is subclassed with `audit_probe.LogMixin`, "
              "whose methods call the original ones and then only record", ["probe", "make_coarsener", ["type"]]],
    ["call", "`coarsener.coarsen()`; `audit_probe.coarsen_summary` then summarises the merges (partition, "
             "ties, merges executed against their scored direction)", ["probe", "staged_job", ["coarsen", "coarsen_summary"]]],
    ["call", "`src.experiment.solve_on_graph(coarse, depot, capacity, 'FQS', cfg, penalty_mode='adaptive', coarsener=coarsener)`",
     ["probe", "staged_job", ["solve_on_graph"]]],
    ["inside", "`vrp_problem_from_graph` → `adaptive_penalties` → `solve_vrp`",
     ["upstream", "src/experiment.py::solve_on_graph", ["vrp_problem_from_graph", "adaptive_penalties", "solve_vrp"]]],
    ["inside", "`solve_vrp`: `get_qubo` → `solve_qubo` (SimulatedAnnealingSampler, reads, sweeps, seed) → "
               "`VRPSolution`: decoded `raw_solution`, then `_repair_solution` and `_repair_time_windows`",
     ["upstream", "src/solvers.py::solve_vrp", ["get_qubo", "solve_qubo", "VRPSolution"]]],
    ["inside", "`report(raw_solution)` gives the pre-repair metrics; `inflate_route` expands the repaired routes; "
               "`calculate_route_metrics` scores them on the original instance",
     ["upstream", "src/experiment.py::solve_on_graph", ["report", "inflate_route", "calculate_route_metrics"]]],
    ["measure", "raw routes: `audit_probe.route_schedule` with and without the return to the depot, alongside "
                "`VRPSolution._count_route_tw_violations`", ["probe", "staged_job", ["route_schedule", "_count_route_tw_violations"]]],
    ["measure", "raw routes expanded with `inflate_route` and scored with `calculate_route_metrics` on the original "
                "instance; `audit_probe.coverage` checks that every customer appears once",
     ["probe", "staged_job", ["inflate_route", "calculate_route_metrics", "coverage"]]],
    ["measure", "repaired routes scored on the coarse graph (`calculate_route_metrics`, `VRPSolution.check()`); "
                "coverage and distance of the expanded routes", ["probe", "staged_job", ["check"]]],
]
EXECUTION_PATHS = {
    "data": {
        "kind": "released functions, no solving",
        "call": "run.data_and_scripts()",
        "entry": "`audit_probe.data_facts`",
        "steps": [
            ["call", "`src.datasets.list_solomon`, `train_test_split_instances` and `load_solomon` on all 56 instances",
             ["probe", "data_facts", ["list_solomon", "train_test_split_instances", "load_solomon"]]],
            ["measure", "first N customers (`.truncate(N)`) of each test instance compared with every training instance",
             ["probe", "data_facts", ["truncate"]]],
            ["call", "`SpatioTemporalGraphCoarsener.merge_score` with α = 0, β = 1 on every ordered customer pair of the "
                     "test instances (the β term of Eq. (1))", ["probe", "data_facts", ["merge_score"]]],
            ["call", "a constructed one-customer route that returns late, scored by "
                     "`VRPSolution._count_route_tw_violations`, `VRPSolution.report` and `calculate_route_metrics`",
             ["probe", "data_facts", ["_count_route_tw_violations", "report", "calculate_route_metrics"]]],
        ],
        "scope": "not applicable; these are properties of the data and of the functions (test data, merge order, "
                 "pre-repair metrics).",
    },
    "scripts": {
        "kind": "experiment script, unchanged",
        "call": "run.data_and_scripts()",
        "entry": "`scripts/run_scorer_ablation.py` (and `scripts/run_scaling_100.py` on the full grid) "
                 "from the repository root",
        "steps": [
            ["script", "`src.refcache.build_cache` (OR-Tools references) → `src.runner_parallel.build_jobs` "
                       "(13 test instances, N = 10, 20, 40, seeds 0-2, three arms, adaptive penalties, 200 reads × "
                       "200 sweeps, checkpoint `results/gnn_model_rl.pt`) → `src.runner_parallel.run_grid`",
             ["upstream", "scripts/run_scorer_ablation.py::main", ["build_cache", "build_jobs", "run_grid"]]],
            ["inside", "`run_grid` calls `run_one_job` for each job in a process pool; rows go to "
                       "`results/scorer_ablation.csv`; means and a Wilcoxon test are printed", None],
            ["probe", "no tracked file of the repository may change", None],
        ],
        "scope": "this output is the reference for the instrumented runs (comparison below).",
    },
    "main_grid": {
        "kind": "instrumented pipeline",
        "call": "run.main_grid()",
        "entry": "`audit_probe.staged_job` once per test instance × N × seed × arm",
        "same_as": "follows `src.runner_parallel.run_one_job` step by step (seed, truncation, coarsener settings, "
                   "`solve_on_graph` call, metrics) without calling it",
        "steps": _STAGED_STEPS,
        "scope": "compared field by field with `run_scorer_ablation.py` on every run both contain (comparison below).",
    },
    "edge_order": {
        "kind": "control variant",
        "call": "run.edge_order()",
        "entry": "`audit_probe.staged_job` with `orient='flipped'` (steps of the main grid), and "
                 "`audit_probe.signature_job` (coarsening only) for the label pool",
        "steps": [
            ["change", "edges from `graph_from_instance` get `u_id` and `v_id` swapped before coarsening",
             ["probe", "make_coarsener", ["graph_from_instance", "u_id"]]],
            ["change", "`audit_probe.FlipMixin._reconnect` replaces `_reconnect`: edges added after a merge are "
                       "created as (neighbour, super-node) instead of (super-node, neighbour). D_ij is directional, "
                       "with i = the edge's first endpoint, and `build_features` builds the GNN's candidate-edge "
                       "features in stored order, so the change can alter which pairs score lowest in every arm",
             ["probe", "make_coarsener", ["FlipMixin"]]],
            ["probe", "label pool: the five heuristic settings `SWEEP` of `scripts/gen_labels_rl.py`, applied with "
                      "`SpatioTemporalGraphCoarsener` to every training instance at N = 10, 15, 20 in both "
                      "orientations; coarsening only", ["probe", "signature_job", ["make_coarsener", "coarsen"]]],
        ],
        "scope": "none; no released script runs this variant.",
    },
    "hash_salts": {
        "kind": "pipeline under other hash salts, and a control variant",
        "call": "run.hash_salts()",
        "entry": "a hash probe, then `signature_job` and `staged_job` under each `PYTHONHASHSEED`, and "
                 "`signature_job` with `tie='canonical'`",
        "steps": [
            ["call", "`hash('SN_12_7')` before and after `src.utils.set_global_seed(0)` in fresh processes",
             ["probe", "main", ["hash", "set_global_seed"]]],
            ["probe", "coarsening of every main-grid case recomputed under each salt; no code change",
             ["probe", "signature_job", ["coarsen"]]],
            ["probe", "cases whose coarsening differs from salt 0 solved again under that salt for every seed; "
                      "unchanged cases solved as a control", None],
            ["change", "`tie='canonical'`: `audit_probe.CanonicalMixin` sorts the edge list by node rank before each "
                       "level and reconnects neighbours in rank order, so ties are broken by rank instead of by the salt",
             ["probe", "make_coarsener", ["CanonicalMixin"]]],
        ],
        "scope": "the salt runs use the code of the main grid unchanged, and the unchanged cases check that an identical "
                 "coarsening gives an identical outcome; the canonical tie-break has no counterpart in the released code.",
    },
    "label_stability": {
        "kind": "label-generation functions, re-evaluated",
        "call": "run.label_stability()",
        "entry": "`audit_probe.label_plan`, then `audit_probe.label_case` per training instance × N",
        "steps": [
            ["probe", "`label_plan` reproduces the random stream of `gen_labels_rl.main()`: "
                      "`np.random.default_rng(0)`, consumed in the order N × training instance, four sub-seeds each",
             ["probe", "label_plan", ["default_rng", "integers"]]],
            ["call", "`src.baselines.solve_cvrptw_ortools(inst, time_limit_s=2.0)` for the reference cost",
             ["probe", "label_case", ["solve_cvrptw_ortools"]]],
            ["substitute", "checkpoint `results/gnn_model_rl.pt` instead of the script default "
                           "`results/gnn_model_multiN.pt`, which is not published", None],
            ["probe", "candidates as in `gen_labels_rl.build_pool`: the five `SWEEP` heuristics, a deterministic "
                      "`GNNCoarsener` (explore = 0) and four stochastic ones (explore = 0.35, the sub-seeds), rebuilt "
                      "for each SA seed", ["probe", "label_case", ["SpatioTemporalGraphCoarsener", "GNNCoarsener"]]],
            ["call", "`gen_labels_rl.eval_rollout(inst, coarsener, SolverConfig('sa', 200, 200, seed))` and "
                     "`gen_labels_rl.reward(metrics, ref_cost)`",
             ["probe", "label_case", ["SolverConfig", "eval_rollout", "reward"]]],
            ["inside", "`eval_rollout`: `coarsen(record_levels=True)` → `solve_on_graph(..., penalty_mode='adaptive')`",
             ["upstream", "scripts/gen_labels_rl.py::eval_rollout", ["coarsen", "solve_on_graph"]]],
            ["measure", "every candidate under every seed in `label_seeds`; seed 0 is the script's own evaluation, "
                        "and its choice (first maximum of the reward) is applied at verification", None],
        ],
        "scope": "none; the script cannot run without `gnn_model_multiN.pt`.",
    },
}




# Where the data behind a table come from. Every table a check relies on has an origin.
ORIGINS = {
    "script": "experiment script, unchanged",
    "pipeline": "instrumented pipeline (main grid)",
    "measure": "audit measurement on the main-grid runs",
    "control": "control variant (intentional change)",
    "salt": "pipeline under other hash salts",
    "data": "data loader and functions applied to the data",
    "source": "source code or data files at the pinned commit",
    "labels": "label-generation functions, several seeds",
    "stats": "statistic recomputed by the audit from the runs",
    "reference": "arithmetic on a value quoted from [1]",
}
TABLE_ORIGIN = {
    "split_by_series": "data", "shared_customer_maps": "data", "customer_R101_R104": "data",
    "series_pairs": "data", "c104_rows_not_parsed": "source", "c104_size_series": "data",
    "predecessor_first_version": "source", "hash_probe": "measure", "ties": "measure",
    "salt_variation": "salt", "feasibility_by_salt": "salt", "canonical_tie_break": "control",
    "orientation": "measure", "beta_term": "data", "orientation_counterfactual": "control",
    "orientation_feasibility": "control", "label_pool_orientation": "control",
    "predecessor_constructs": "source", "supernode_construction": "measure",
    "loss_at_expansion": "measure", "distance_gap": "measure", "source_verification": "source",
    "problem_size_by_arm_pair": "pipeline", "depot_return_example": "data", "late_returns": "measure",
    "pre_repair_measurements": "measure", "pre_repair_test": "stats", "scaling_script_tests": "source",
    "feasibility_overall": "pipeline", "feasibility_by_arm_and_size": "pipeline",
    "paired_arm_comparisons": "stats", "n10_by_instance": "script", "n10_by_type": "script",
    "problem_sizes": "pipeline", "stopped_above_target": "measure", "wilcoxon_calls": "source",
    "large_n_test": "stats", "all_sizes_base_instances": "stats", "sizes_per_script": "source",
    "per_size_two_sided": "stats", "properties_by_arm": "control", "label_script_structure": "source",
    "label_runs": "labels", "label_selection": "labels", "feasibility_by_stage": "measure",
    "large_n_per_size": "stats", "pre_repair_on_original_instance": "measure",
}
# The table that holds the evidence of each check (None: a value without a table, origin given).
CHECK_EVIDENCE = {
    "7.1": "split_by_series", "7.2": "shared_customer_maps", "7.3": "shared_customer_maps",
    "7.4": "customer_R101_R104", "7.5": "series_pairs", "7.6": "c104_rows_not_parsed",
    "7.7": "c104_size_series", "7.8": "c104_rows_not_parsed", "7.9": "c104_size_series",
    "7.10": "c104_size_series", "7.11": (None, "source"), "7.12": "predecessor_first_version",
    "8.1": "hash_probe", "8.2": (None, "pipeline"), "8.3": "ties", "8.4": "salt_variation",
    "8.5": "feasibility_by_salt", "8.6": "canonical_tie_break", "8.7": "orientation", "8.8": "orientation",
    "8.9": "beta_term", "8.10": "orientation_counterfactual", "8.11": "label_pool_orientation",
    "8.12": "predecessor_constructs",
    "9.1": "supernode_construction", "9.2": "supernode_construction", "9.3": "loss_at_expansion",
    "9.4": "loss_at_expansion", "9.5": "distance_gap", "9.6": "source_verification",
    "10.1": "problem_size_by_arm_pair", "10.2": "depot_return_example", "10.3": "pre_repair_measurements",
    "10.4": "pre_repair_test",
    "11.1": "scaling_script_tests", "11.2": "feasibility_overall", "11.3": "feasibility_by_arm_and_size",
    "11.4": "n10_by_instance", "11.5": "n10_by_type", "11.6": "split_by_series", "11.7": (None, "reference"),
    "11.8": (None, "script"), "11.9": "problem_sizes", "11.10": "stopped_above_target",
    "11.11": "scaling_script_tests", "11.12": "wilcoxon_calls", "11.13": (None, "data"),
    "11.14": "large_n_test", "11.15": "large_n_test", "11.16": "all_sizes_base_instances",
    "11.17": "sizes_per_script", "11.18": "properties_by_arm",
    "12.1": "label_script_structure", "12.2": "label_selection", "12.3": (None, "labels"),
    "12.4": "label_selection",
}


def evidence(check):
    e = CHECK_EVIDENCE[check]
    table, origin = (e if isinstance(e, tuple) else (e, TABLE_ORIGIN[e]))
    return (f"tables/{table}.csv" if table else "-"), ORIGINS[origin]


def _show(df, index=False):
    st = df.style if index else df.style.hide(axis="index")
    display(st.set_properties(**{"text-align": "left"}))


def _xpos(v):
    return {N: i for i, N in enumerate(sorted(set(v.grid) | set(v.published)))}


def _fig_expansion(v, path):
    t = v.tables["feasibility_by_stage"]
    xpos = _xpos(v)
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4), sharey=True)
    for ax, arm in zip(axes, v.arms):
        for col, ls in (("coarse_feasible", "--"), ("is_feasible", "-")):
            d = t[(t.arm == arm) & (t.stage == col)].sort_values("N")
            ax.errorbar(d.N.map(xpos), d["feasible %"], yerr=[d["feasible %"] - d["CI95 low"], d["CI95 high"] - d["feasible %"]],
                        color=ARM_STYLE[arm]["color"], marker=ARM_STYLE[arm]["marker"], linestyle=ls,
                        linewidth=2, markersize=5, capsize=3, elinewidth=1)
        ax.set_title(ARM_LABEL[arm], fontsize=9.5)
        ax.set_xlabel("N (customers, evenly spaced); n = runs")
        n_runs = d.set_index("N").runs
        ax.set_xticks([xpos[N] for N in n_runs.index], [f"{N}\nn={n}" for N, n in n_runs.items()])
    axes[0].set_ylabel("feasible runs (%)")
    axes[0].legend(handles=[matplotlib.lines.Line2D([], [], color="#52514e", linestyle="--", label="on the coarse graph (repaired)"),
                            matplotlib.lines.Line2D([], [], color="#52514e", linestyle="-", label="after expansion")],
                   loc="lower left", fontsize=8)
    fig.suptitle("Feasibility before and after expansion (95% cluster-bootstrap CI)", fontsize=10.5)
    fig.savefig(path, bbox_inches="tight")
    plt.show()


def _fig_rates(v, path):
    tab, col = v.tables["feasibility_by_arm_and_size"], "is_feasible"
    xpos = _xpos(v)
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    for arm in v.arms:
        d = tab[tab.arm == arm].sort_values("N")
        st = ARM_STYLE[arm]
        ax.errorbar(d.N.map(xpos), d[f"{col} %"], yerr=[d[f"{col} %"] - d["CI95 low"], d["CI95 high"] - d[f"{col} %"]],
                    color=st["color"], marker=st["marker"], markersize=6, linewidth=2, capsize=3, elinewidth=1,
                    label=ARM_LABEL[arm])
    ax.set_xlabel("N (customers, evenly spaced); n = runs per arm")
    ax.set_ylabel("feasible runs (%)")
    ax.set_title("Feasibility after expansion, main grid")
    n_runs = tab.groupby("N").runs.first()
    ax.set_xticks([xpos[N] for N in n_runs.index], [f"{N}\nn={n}" for N, n in n_runs.items()])
    ax.set_ylim(-3, 103)
    ax.legend(loc="lower left", fontsize=8.5)
    fig.savefig(path, bbox_inches="tight")
    plt.show()


def _fig_path(v, name):
    (v.out / "figures").mkdir(exist_ok=True)
    return v.out / "figures" / f"{name}.png"


# --------------------------------------------------------------------------- #
# the analyses of the report, in the order of the argument
# --------------------------------------------------------------------------- #
# Each analysis reads the verified facts of one group of checks (the prefix of the check ids, which
# are stable identifiers) and shows only the tables its argument needs, as views of the verified
# tables; the full tables are in the appendix (DIAGNOSTICS). Nothing here recomputes a statistic.
ANALYSES = {
    "baselines": ("GNN versus the heuristic baselines", 11),
    "statistics": ("Statistical tests", 11),
    "expansion": ("Feasibility after expansion", 9),
    "pre_repair": ("Pre-repair metrics", 10),
    "merge_order": ("Merge sensitivity to edge orientation and hash salt", 8),
    "test_data": ("Test data and the C104 instance", 7),
    "labels": ("Candidate-selection stability", 12),
}
_BASELINE_CHECKS = {"11.2", "11.3", "11.9", "11.10", "11.18"}
# verified tables each analysis shows (as views) in the main text
MAIN_TABLES = {
    "baselines": ["feasibility_by_arm_and_size", "paired_arm_comparisons"],
    "statistics": ["large_n_test", "per_size_two_sided", "n10_by_instance"],
    "expansion": ["feasibility_by_stage", "loss_at_expansion", "distance_gap"],
    "pre_repair": ["problem_size_by_arm_pair", "depot_return_example", "late_returns", "pre_repair_test"],
    "merge_order": ["ties", "salt_variation", "canonical_tie_break", "orientation", "orientation_counterfactual",
                    "properties_by_arm"],
    "test_data": ["shared_customer_maps", "c104_rows_not_parsed", "c104_size_series"],
    "labels": ["label_runs", "label_selection"],
}
# full tables of each analysis in the appendix, with a caption
DIAGNOSTICS = {
    "baselines": [("feasibility_by_arm_and_size", "Feasibility after expansion with cluster-bootstrap 95% intervals"),
                  ("feasibility_overall", "Feasibility after expansion over all sizes of the grid"),
                  ("paired_arm_comparisons", "Paired comparisons of the arms on the same instance, size and seed"),
                  ("problem_sizes", "Mean coarse problem size (QUBO variables, coarse nodes) per arm and size"),
                  ("stopped_above_target", "Coarsenings that stop above the target size P·n₀"),
                  ("properties_by_arm", "Runs whose feasibility changes because of each code property")],
    "statistics": [("n10_by_instance", "N = 10, unmodified run_scorer_ablation.py: feasible seeds per instance"),
                   ("n10_by_type", "N = 10, unmodified run_scorer_ablation.py: feasibility (%) by instance type"),
                   ("scaling_script_tests", "Tests in run_scaling_100.py"),
                   ("wilcoxon_calls", "Wilcoxon tests in the other scripts"),
                   ("large_n_per_size", "N = 80 and 100 tested separately, two-sided"),
                   ("all_sizes_base_instances", "All sizes of the grid aggregated to base instances"),
                   ("sizes_per_script", "Sizes each script covers")],
    "expansion": [("supernode_construction", "Super-node construction and the routes that are scored"),
                  ("loss_at_expansion", "Runs feasible on the coarse graph and after expansion, with Wilson 95% intervals"),
                  ("feasibility_by_stage", "Feasibility on the coarse graph and after expansion, cluster-bootstrap 95% intervals")],
    "pre_repair": [("late_returns", "Raw routes that return to the depot after its window closes"),
                   ("pre_repair_measurements", "Pre-repair feasibility (%) measured three ways"),
                   ("pre_repair_on_original_instance", "Pre-repair feasibility on the original instance, 95% intervals")],
    "merge_order": [("hash_probe", "String hashing in fresh processes, before and after set_global_seed(0)"),
                    ("ties", "Coarsenings with tied scores below the admission threshold"),
                    ("salt_variation", "Cases whose coarsening differs across hash salts"),
                    ("feasibility_by_salt", "Feasibility (%) per arm and size under each salt"),
                    ("canonical_tie_break", "The same cases with a deterministic tie-break"),
                    ("orientation", "Merges executed opposite to their scored direction"),
                    ("beta_term", "The β term of Eq. (1) on all ordered customer pairs of the test instances"),
                    ("orientation_counterfactual", "Every edge created with its endpoints in the opposite order"),
                    ("orientation_feasibility", "Feasibility with the opposite orientation, by whether the coarsening changed"),
                    ("label_pool_orientation", "Heuristic settings of the label pool: coarsenings changed by the edge order"),
                    ("predecessor_constructs", "The same constructs in the predecessor's code [3]")],
    "test_data": [("split_by_series", "Held-out instances per Solomon series"),
                  ("customer_R101_R104", "Customer 2 of R101 (training) and R104 (test)"),
                  ("series_pairs", "Series 1 and 2 of each class: shared coordinates and demands")],
    "labels": [("label_script_structure", "Structure of the selection in gen_labels_rl.py"),
               ("label_selection", "The label under one evaluation and under the average over seeds, per case")],
}
_SHORT = {"heuristic": "tuned", "heuristic_fixed": "fixed", "gnn": "GNN"}


def analysis_of(check):
    """The analysis of the report a check belongs to."""
    group = int(check.split(".")[0])
    if group == 11:
        return "baselines" if check in _BASELINE_CHECKS else "statistics"
    return next(k for k, (_, g) in ANALYSES.items() if g == group)


def _checks(v, key):
    return [c for c in v.checks if analysis_of(c["check"]) == key]


def _ok(v, *ids):
    st = {c["check"]: c["status"] for c in v.checks}
    return all(st.get(i) == "confirmed" for i in ids)


def _stages(names):
    return _join([f"`{n}`" for n in names])


def _status_line(v, key):
    cs = sorted(_checks(v, key), key=lambda c: tuple(map(int, c["check"].split("."))))
    counts = {s: sum(c["status"] == s for c in cs) for s in STATUSES}
    line = (f"Checks {', '.join(c['check'] for c in cs)}: "
            + ", ".join(f"{counts[s]} {s}" for s in STATUSES if counts[s]) + " (appendix, table of checks).")
    group = ANALYSES[key][1]
    if group in v.missing:
        return line + f" Generation stages not done: {_stages(v.missing[group])}."
    reasons = {}
    for c in cs:
        if c["status"] != "confirmed":
            why = c["note"] or ("needs the published grid" if v.quick and c["status"] == "insufficient data" else "")
            reasons.setdefault((c["status"], why), []).append(c["check"])
    if reasons:
        line += " " + "; ".join(f"{', '.join(ids)}: {st}" + (f" ({why})" if why else "")
                                for (st, why), ids in reasons.items()) + "."
    return line


def _caption(text, *names):
    origins = "; ".join(dict.fromkeys(ORIGINS[TABLE_ORIGIN[n]] for n in names))
    display(Markdown(f"**{text}** - *{origins}* ({', '.join(f'`tables/{n}.csv`' for n in names)})"))


def _net(v):
    """Net feasible runs (better − worse) of each paired comparison, summed over sizes."""
    s = v.tables["paired_arm_comparisons"].groupby("comparison")[["better", "worse", "pairs (runs)"]].sum()
    return {k: int(r.better - r.worse) for k, r in s.iterrows()}, s


def _stage_totals(v):
    """Runs feasible on the coarse graph and after expansion, per arm, summed over sizes."""
    return v.tables["loss_at_expansion"].groupby("arm")[
        ["runs", "feasible on the coarse graph", "feasible after expansion", "lost at expansion"]].sum()


def _fixed_vs_gnn_at_expansion(tot):
    """(coarse GNN, coarse fixed, expanded GNN, expanded fixed), or None without both arms."""
    if not {"gnn", "heuristic_fixed"} <= set(tot.index):
        return None
    return tuple(int(tot.loc[a, c]) for c in ("feasible on the coarse graph", "feasible after expansion")
                 for a in ("gnn", "heuristic_fixed"))


def _join(items):
    items = list(items)
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


# --------------------------------------------------------------------------- #
# the views: each shows its tables and returns (conclusion, limitation)
# --------------------------------------------------------------------------- #
def _v_baselines(v, f):
    _caption("Feasibility after expansion per arm and size", "feasibility_by_arm_and_size")
    _fig_rates(v, _fig_path(v, "feasibility_by_arm"))
    F = v.tables["feasibility_by_arm_and_size"]
    C = v.tables["paired_arm_comparisons"].set_index(["N", "comparison"])
    pairs = (("gnn vs heuristic_fixed", "GNN vs fixed"), ("heuristic_fixed vs heuristic", "fixed vs tuned"),
             ("gnn vs heuristic", "GNN vs tuned"))
    rows = []
    for N, d in F.groupby("N"):
        row = {"N": N, "runs per arm": int(d.runs.iloc[0])}
        row.update({f"{_SHORT[a]} %": d.set_index("arm").loc[a, "is_feasible %"] for a in v.arms})
        row.update({f"{lab}: better / worse": f"{C.loc[(N, c), 'better']} / {C.loc[(N, c), 'worse']}" for c, lab in pairs})
        rows.append(row)
    _caption("Paired runs (same instance, size and seed) in which one arm is feasible and the other is not",
             "feasibility_by_arm_and_size", "paired_arm_comparisons")
    _show(pd.DataFrame(rows))
    net, S = _net(v)
    gt, fh, gf = net["gnn vs heuristic"], net["heuristic_fixed vs heuristic"], net["gnn vs heuristic_fixed"]
    n = int(S.loc["gnn vs heuristic", "pairs (runs)"])
    concl = (f"Over the {n} paired runs, the GNN has {gt:+d} feasible runs relative to the tuned heuristic. Giving the "
             f"tuned heuristic the GNN's settings (heuristic-fixed) changes its count by {fh:+d}; replacing its score by "
             f"the GNN with those settings held equal changes it by a further {gf:+d}. The two parts add up to the total "
             f"because all arms run on the same instances, sizes and seeds. Against heuristic-fixed, the GNN is feasible "
             f"where heuristic-fixed is not in {f['g_better']} runs, and the reverse holds in {f['g_worse']}.")
    if gt > 0 and fh > 0 and gf > 0:
        concl += (f" On this grid the headline comparison therefore combines two changes in the same direction, and "
                  f"the settings alone account for {fh} of its {gt} net runs.")
    else:
        concl += " The headline comparison combines both changes; the two parts above separate them."
    concl += (f" Equal settings do not mean equal problems: the GNN and heuristic-fixed end at different coarse sizes in "
              f"{f['diff_size']} of {f['cases']} instance-size cases.")
    lim = ("These counts describe the quick grid (one seed at N = 80 and 100; N = 50 and 60 not run), not the published "
           "grid, and the comparisons with the published rates need the full grid. " if v.quick else "")
    lim += ("Runs of one instance are not independent, so the counts are descriptive; the tests follow below. The arms "
            "are also affected unequally by the expansion step and by the edge orientation (analysed below), which "
            "these counts do not separate from the score.")
    return concl, lim


def _v_statistics(v, f):
    _caption("The large-N test as scripts/run_scaling_100.py computes it (seed means, N = 80 and 100 pooled, "
             "GNN vs tuned heuristic), recomputed from the stored runs", "large_n_test")
    _show(v.tables["large_n_test"])
    P = v.tables["per_size_two_sided"]
    rows = []
    for N, d in P.groupby("N"):
        row = {"N": N, "base instances": int(d["base instances"].iloc[0])}
        for cmp, lab in (("gnn vs heuristic_fixed", "GNN vs fixed"), ("gnn vs heuristic", "GNN vs tuned")):
            r = d.set_index("comparison").loc[cmp]
            row[f"{lab}: instances better / worse"] = f"{int(r['GNN better'])} / {int(r['GNN worse'])}"
            row[f"{lab}: p two-sided"] = r["p two-sided"]
        rows.append(row)
    _caption("Per size, median over seeds, base instances as the unit, two-sided Wilcoxon (Section 4 of [1])",
             "per_size_two_sided")
    _show(pd.DataFrame(rows))
    T = v.tables["n10_by_instance"]
    below = T[(T < len(v.seeds)).any(axis=1)]
    _caption("N = 10: instances not feasible for every seed in every arm (unmodified run_scorer_ablation.py)",
             "n10_by_instance")
    if len(below):
        _show(below, index=True)
    else:
        print("none: every instance is feasible for every seed in every arm")
    fx, gt = P[P.comparison == "gnn vs heuristic_fixed"], P[P.comparison == "gnn vs heuristic"]

    def sig(d):
        s = d[d["p two-sided"] < 0.05]
        return (f"p < 0.05 at N = {', '.join(map(str, s.N))}" if len(s)
                else f"no p below 0.05 (smallest {d['p two-sided'].min():.4f})")
    concl = (f"Recomputed from the stored runs, the released large-N test gives one-sided p = {f['p1_large']:.4f}; "
             f"the same comparison two-sided gives p = {f['p2_large']:.4f}. It compares the GNN with the tuned heuristic "
             f"only, on seed means pooled over N = 80 and 100. Tested as Section 4 of [1] describes, per size and base "
             f"instance and two-sided, the GNN against heuristic-fixed gives {sig(fx)}, and against the tuned heuristic "
             f"{sig(gt)}.")
    k = _max_differing(v)
    if f["fx_sig"] == 0 and 0 < k <= 5:
        concl += (f" At most {k} base instances differ between the GNN and heuristic-fixed at any size, and an exact "
                  f"two-sided signed-rank test on {k} differences cannot give p below {2 / 2 ** k:.4f}; the absence of "
                  "significance means that this grid cannot resolve the difference at the 5% level, not that there is none.")
    if f["n10_below"]:
        concl += (f" At N = 10 the arms differ on {'one instance' if f['n10_below'] == 1 else str(f['n10_below']) + ' instances'} "
                  f"({f['n10_found']}); every other test instance is feasible for every seed in every arm, so the 100% "
                  f"at N = 10 rests on {'that instance' if f['n10_below'] == 1 else 'these instances'}.")
    gap = next((c["computed"] for c in v.checks if c["check"] == "11.7"), None)
    if gap and _ok(v, "11.7"):
        concl += (f" The retraining mean of Table 3 of [1], 87.2%, lies {gap} below 100% on a test set of "
                  f"{len(v.D['test'])} instances.")
    lim = ("On the quick grid N = 80 and 100 have one seed, so the median over seeds there is a single run, and the "
           "published p-values are not compared. " if v.quick else "")
    lim += "The retrained models of Table 3 of [1] are not re-run; 87.2 ± 3.6 is quoted."
    return concl, lim


def _v_expansion(v, f):
    _caption("Feasibility on the coarse graph (repaired, dashed) and after expansion (solid)", "feasibility_by_stage")
    _fig_expansion(v, _fig_path(v, "feasibility_before_after_expansion"))
    L = v.tables["loss_at_expansion"]
    T = L.assign(cell=[f"{a} / {b}" for a, b in zip(L["lost at expansion"], L["feasible on the coarse graph"])]) \
        .pivot(index="arm", columns="N", values="cell").reindex(v.arms)
    T.columns = [f"N = {N}: lost / coarse-feasible" for N in T.columns]
    tot = _stage_totals(v)
    T = T.join(tot.rename(columns={"runs": "runs, all sizes", "feasible on the coarse graph": "coarse-feasible, all sizes",
                                   "feasible after expansion": "feasible after expansion, all sizes",
                                   "lost at expansion": "lost, all sizes"}))
    _caption("Runs feasible on the coarse graph that are infeasible after expansion", "loss_at_expansion")
    _show(T.rename(index=lambda a: ARM_LABEL[a]), index=True)
    _caption("Expanded length minus (coarse length + internal legs of the merged pairs)", "distance_gap")
    _show(v.tables["distance_gap"])
    lost = f["lost"]
    concl = (f"Expansion alone turns {int(lost.sum())} of {f['coarse_feasible']} coarse-feasible runs infeasible ("
             + ", ".join(f"{_SHORT[a]} {int(lost[a])}" for a in v.arms) + f"), most at N = {f['worst_N']} "
             f"({f['worst_lost']} of {f['worst_coarse']}).")
    pair = _fixed_vs_gnn_at_expansion(tot)
    if pair:
        cg, cf, eg, ef = pair
        concl += (f" Before expansion the GNN and heuristic-fixed are feasible in {cg} and {cf} of "
                  f"{int(tot.loc['gnn', 'runs'])} runs; after expansion in {eg} and {ef}.")
        if (eg - ef) != (cg - cf):
            part = (eg - ef) - (cg - cf)
            concl += (f" Of their difference after expansion ({eg - ef:+d} runs), {part:+d} "
                      + ("- all of it on this grid - " if cg == cf else "") + "arises in the lifting step, not in the "
                      "coarse solutions.")
    if f["gap_nonzero"] and _ok(v, "9.5"):
        concl += (f" In {f['gap_nonzero']} of {f['gap_runs']} runs the expanded length differs from the coarse length "
                  "plus the internal legs of the merged pairs: distances are consistent only for the leg inside a pair.")
    big = L[L.N >= 80].runs
    lim = ("Both measurements use the evaluator of the reported feasibility on the same runs. Whether a repair after "
           "expansion, or a super-node that keeps the entry and exit of its pair, would remove the loss is not computed; "
           "both are changes of method.")
    if len(big):
        lim += (f" At N ≥ 80 there are {big.min()}-{big.max()} runs per arm and size, so the loss rates there are "
                "imprecise (Wilson intervals in the appendix).")
    return concl, lim


def _v_pre_repair(v, f):
    _caption("Runs in which two arms hand the sampler coarse problems of different size", "problem_size_by_arm_pair")
    _show(v.tables["problem_size_by_arm_pair"])
    _caption("A constructed route that returns to the depot late, scored by the pre-repair counter and the final "
             "evaluator", "depot_return_example")
    _show(v.tables["depot_return_example"])
    _caption("The pre-repair test of [1] (code of run_scorer_ablation.py) on both measurements", "pre_repair_test")
    _show(v.tables["pre_repair_test"])
    E = v.tables["depot_return_example"].iloc[0]
    printed = v.tables["pre_repair_test"]["printed by the script"].iloc[0]
    concl = (f"On the same run, the GNN and heuristic-fixed hand the sampler coarse problems of different size in "
             f"{f['size_diff']} of {f['runs']} runs, so their pre-repair counts are counts on different problems. The "
             f"pre-repair counter stops at the last customer: on the constructed route it counts "
             f"{E['solver_counter_violations']} violations where the final evaluator counts "
             f"{E['final_evaluator_violations']}, and {f['late_returns']} raw routes of the main grid return to the "
             f"depot late. Measured on the original instance with the final evaluator, pre-repair feasibility differs "
             f"from the reported measurement in {f['disagree']} of {f['n']} runs. ")
    concl += (f"The pre-repair test of [1] reproduces (one-sided p = {f['p62']:.3f}; the script prints {printed})"
              if _ok(v, "10.4") else f"The pre-repair test of [1] gives one-sided p = {f['p62']:.3f}")
    concl += (f"; on the original-instance measurement it gives p = {f['p62_orig']:.3f}. A large p here does not show "
              "equal pre-repair quality: the test compares counts on problems of different size.")
    lim = (f"The depot-return omission changes the pre-repair verdict in {f['disagree']} of {f['n']} runs of this grid "
           f"({pct(f['disagree'], f['n'])}); whether it matters for the hardware comparison of Section 6 of [1], which "
           "uses the same counter, is not tested.")
    return concl, lim


def _v_merge_order(v, f):
    salts = f["salts"]
    TI = v.tables["ties"].set_index("arm")
    SV = v.tables["salt_variation"].groupby("arm")[["cases", "cases_that_vary"]].sum()
    CV = v.tables["canonical_tie_break"].set_index("arm")
    O = v.tables["orientation"].set_index("arm")
    OF = v.tables["orientation_counterfactual"].set_index("arm")
    PR = v.tables["properties_by_arm"]
    rows = []
    for a in v.arms:
        rows.append({"arm": ARM_LABEL[a], "coarsenings": int(TI.loc[a, "coarsenings"]),
                     "with tied scores": int(TI.loc[a, "with_ties"]),
                     f"differ across salts {', '.join(map(str, salts))}": f"{int(SV.loc[a, 'cases_that_vary'])} / {int(SV.loc[a, 'cases'])}",
                     "differ, deterministic tie-break": f"{int(CV.loc[a, 'cases_that_vary'])} / {int(CV.loc[a, 'cases'])}",
                     "merges executed against the scored direction": f"{O.loc[a, 'reversed % of merges']}%",
                     "first-level merges changed, reversed orientation": f"{int(OF.loc[a, 'level1_changed'])} / {int(OF.loc[a, 'coarsenings'])}",
                     "runs whose feasibility changes: salt": int(PR.loc["hash salt", a]),
                     "runs whose feasibility changes: reversed orientation": int(PR.loc["edge order", a])})
    _caption("Coarsenings of the main grid under other hash salts, a deterministic tie-break and the reversed edge "
             "orientation", "ties", "salt_variation", "canonical_tie_break", "orientation", "orientation_counterfactual",
             "properties_by_arm")
    _show(pd.DataFrame(rows))
    heur = [a for a in v.arms if a != "gnn"]
    concl = ""
    if _ok(v, "8.1"):
        concl += ("Setting `PYTHONHASHSEED` from inside the running process leaves string hashing unchanged in every "
                  "fresh process tested, so the seed function does not fix the order in which edges are added after a "
                  "merge. ")
    concl += (f"Tied scores below the admission threshold occur in {int(TI.loc[heur, 'with_ties'].sum())} of "
              f"{int(TI.loc[heur, 'coarsenings'].sum())} heuristic coarsenings and {int(TI.loc['gnn', 'with_ties'])} of "
              f"{int(TI.loc['gnn', 'coarsenings'])} GNN coarsenings. Under the salts tried, "
              + _join([f"{int(f['salt_vary'][a])} of {int(f['salt_cases'][a])} `{a}` coarsenings" for a in v.arms])
              + " change; " + (f"{f['salt_flips']} re-solved runs change their feasibility outcome"
                               if f["salt_flips"] else "none of the re-solved runs changes its feasibility outcome"))
    concl += (", and with a deterministic tie-break no case varies. " if f["canonical_vary"] == 0
              else f"; with a deterministic tie-break {f['canonical_vary']} cases still vary. ")
    orient_larger = (int(f["OF"].level1_changed.sum()) > int(f["salt_vary"].sum())
                     and int(PR.loc["edge order"].sum()) > int(PR.loc["hash salt"].sum()))
    concl += ("The stored edge orientation has a larger effect: " if orient_larger else "As for the edge orientation, ")
    concl += (f"{f['reversed_min']:.0f}-{f['reversed_max']:.0f}% of "
              "merges are executed in the direction opposite to the one in which they were scored, and creating every "
              "edge in the opposite order changes the first-level merges in "
              + _join([f"{int(f['OF'].level1_changed[a])} of {int(f['OF'].coarsenings[a])} `{a}`" for a in v.arms])
              + f" coarsenings, and {f['pool_changed']} of {f['pool_cases']} coarsenings of the training-label pool. "
              "It changes the feasibility outcome of "
              + _join([f"{int(PR.loc['edge order', a])} `{a}`" for a in v.arms]) + " runs")
    if 11 in v.facts:
        concl += (f"; for comparison, the GNN minus heuristic-fixed on the same runs is {_net(v)[0]['gnn vs heuristic_fixed']:+d} "
                  "feasible runs (GNN versus the heuristic baselines).")
    else:
        concl += "."
    if _ok(v, "8.12"):
        concl += (" The score-only sort, the set-based reconnection and the scoring orientation are already in the "
                  "first version of the predecessor's code [3]; the in-process seeding is not.")
    lim = ("The reversed orientation is a counterfactual convention: it shows that the stored orientation decides which "
           "pairs merge, not which convention is better. Only "
           f"{len(salts)} salts are tried, and the salts of the published runs were not recorded, so salt-driven variation "
           "in the published numbers cannot be measured.")
    return concl, lim


def _v_test_data(v, f):
    _caption("Training instances with the same first N customers (coordinates and demands) as each test instance",
             "shared_customer_maps")
    _show(v.tables["shared_customer_maps"], index=True)
    _caption("Data rows of C104.csv that do not parse as seven numbers", "c104_rows_not_parsed")
    _show(v.tables["c104_rows_not_parsed"])
    _caption("Whether the loaded C104 equals the first N customers of its file", "c104_size_series")
    _show(v.tables["c104_size_series"])
    s = f["same_series"]
    concl = ""
    if _ok(v, "7.2"):
        concl += (f"Every test instance except C104 has the same customers - positions and demands - as "
                  f"{f['shared_min']}-{f['shared_max']} training instances at every evaluated size; only the time windows "
                  "differ. ")
    concl += ("In the R and RC classes the two series also share their coordinates; C1 and C2 do not. "
              if s["R"] and s["RC"] and not s["C"] else
              "Series sharing coordinates, by class: " + ", ".join(f"{k} {x}" for k, x in s.items()) + ". ")
    if f["bad_field"] is not None:
        concl += (f"C104 is loaded with {f['c104_customers']} customers: line {f['bad_line']} of its file has the ready "
                  f"time {f['bad_field']!r}, the loader skips the row without a warning, and from N = {f['first_bad']} on "
                  f"the loaded instance is no longer the first N customers of the benchmark. At N = 100 it is excluded, "
                  f"which leaves {f['pairs_large']} instance-size pairs at N = 80, 100 and {f['pairs_all']} over the seven "
                  "published sizes.")
    lim = ("How much the altered C104 changes any reported number is not measured. The split shows what the test set "
           "contains; it cannot show how the model does on unseen customer maps, since none is evaluated.")
    return concl, lim


def _v_labels(v, f):
    _caption("The rebuilt candidate pool", "label_runs")
    _show(v.tables["label_runs"])
    comp = {c["check"]: c["computed"] for c in v.checks}
    LW = v.tables["label_selection"]
    name = {"heur": "heuristic setting", "gnn": "GNN rollout"}
    tab = pd.DataFrame({"chosen by one evaluation (seed 0)": pd.Series(f["types_seed0"]),
                        f"best on average over {f['seeds']} seeds": pd.Series(f["types_best"])}).fillna(0).astype(int)
    _caption("Selection under one evaluation and under the average over seeds", "label_selection")
    _show(pd.DataFrame([{"cases (instance, N)": len(LW), "label not best on average": comp.get("12.2"),
                         "winner varies across seeds": f"{f['varies']} / {len(LW)}",
                         "label's reward at seed 0 minus its mean at the other seeds": comp.get("12.4"),
                         "reward variance explained by the feasibility terms": f"{f['share']:.0%}"}]))
    _show(tab.rename(index=lambda t: name.get(t, t)).rename_axis("type of the candidate"), index=True)
    concl = (f"In {f['k']} of {f['n']} instance-size cases ({pct(f['k'], f['n'])}) the candidate one evaluation selects is "
             f"not among those with the highest mean reward over {f['seeds']} seeds, and the winning candidate changes "
             f"between seeds in {f['varies']} cases. The selected label scores {f['curse']:.3f} higher at its selection "
             f"seed than on average at the other seeds (95% CI {f['clo']:.3f} to {f['chi']:.3f})")
    concl += (", the pattern expected when the maximum of single noisy evaluations is kept." if f["clo"] > 0 else
              "; the interval includes zero, so this grid does not show a selection bias.")
    if _ok(v, "12.3"):
        concl += (f" Two binary feasibility indicators explain {f['share']:.0%} of the reward's variance, so one draw "
                  "can decide between candidates of similar quality.")
    lim = ("The GNN rollouts use the published checkpoint `gnn_model_rl.pt` in place of the unpublished "
           "`gnn_model_multiN.pt` of the script, so these runs measure the selection procedure, not the published "
           "labels, and the oracle-win rates quoted in [1] are not recomputed.")
    if v.quick:
        lim += (f" The quick grid uses {LW.instance.nunique()} training instances and {f['seeds']} seeds; the full mode "
                "uses all training instances and more seeds.")
    return concl, lim


VIEWS = {"baselines": _v_baselines, "statistics": _v_statistics, "expansion": _v_expansion,
         "pre_repair": _v_pre_repair, "merge_order": _v_merge_order, "test_data": _v_test_data, "labels": _v_labels}


# one sentence per analysis for the summary at the top, from the same facts
def _h_baselines(v, f):
    net, S = _net(v)
    return (f"Over {int(S.loc['gnn vs heuristic', 'pairs (runs)'])} paired runs the GNN has "
            f"{net['gnn vs heuristic']:+d} feasible runs relative to the tuned heuristic: "
            f"{net['heuristic_fixed vs heuristic']:+d} from the settings alone (heuristic-fixed) and "
            f"{net['gnn vs heuristic_fixed']:+d} from the score with the settings held equal.")


def _max_differing(v):
    P = v.tables["per_size_two_sided"]
    fx = P[P.comparison == "gnn vs heuristic_fixed"]
    return int((fx["GNN better"] + fx["GNN worse"]).max())


def _h_statistics(v, f):
    k = _max_differing(v)
    return (f"The released large-N test (one-sided, seed means, N = 80 and 100 pooled, GNN vs tuned) gives "
            f"p = {f['p1_large']:.4f}; per size and base instance, two-sided, GNN vs heuristic-fixed gives "
            + (f"p < 0.05 at {f['fx_sig']} of {f['fx_sizes']} sizes." if f["fx_sig"]
               else f"no p below 0.05 (smallest {f['fx_min_p']:.4f})"
               + (f"; with at most {k} differing base instances per size no exact two-sided test can reach p < 0.05."
                  if 0 < k <= 5 else "."))
            + (f" At N = 10 the arms differ only on {f['n10_found'].split(':')[0]}." if f["n10_below"] == 1 else ""))


def _h_expansion(v, f):
    out = (f"Expansion turns {int(f['lost'].sum())} of {f['coarse_feasible']} coarse-feasible runs infeasible ("
           + ", ".join(f"{_SHORT[a]} {int(f['lost'][a])}" for a in v.arms) + ").")
    pair = _fixed_vs_gnn_at_expansion(_stage_totals(v))
    if pair and pair[0] == pair[1] and pair[2] != pair[3]:
        out += (f" Before expansion the GNN and heuristic-fixed are feasible equally often ({pair[0]} runs each); the "
                f"difference between them after expansion ({pair[2] - pair[3]:+d} runs) arises in the lifting step.")
    return out


def _h_pre_repair(v, f):
    return (f"Pre-repair counts compare problems of different size (GNN vs heuristic-fixed: {f['size_diff']} of "
            f"{f['runs']} runs) with a counter that ignores the return to the depot; the pre-repair test gives "
            f"p = {f['p62']:.3f}.")


def _h_merge_order(v, f):
    PR = v.tables["properties_by_arm"]
    return (f"Reversing the stored edge orientation changes the first-level merges in "
            f"{int(f['OF'].level1_changed.sum())} of {int(f['OF'].coarsenings.sum())} coarsenings and the feasibility "
            f"of {int(PR.loc['edge order'].sum())} runs; the hash salts tried change "
            f"{int(f['salt_vary'].sum())} coarsenings and {f['salt_flips']} outcomes.")


def _h_test_data(v, f):
    return (f"Test instances other than C104 share their customer maps with {f['shared_min']}-{f['shared_max']} training "
            f"instances; C104 loads with {f['c104_customers']} customers because line {f['bad_line']} of its file is "
            "skipped.")


def _h_labels(v, f):
    return (f"In {f['k']} of {f['n']} cases the label chosen by one evaluation is not the best candidate on average over "
            f"{f['seeds']} seeds.")


HEADLINES = {"baselines": _h_baselines, "statistics": _h_statistics, "expansion": _h_expansion,
             "pre_repair": _h_pre_repair, "merge_order": _h_merge_order, "test_data": _h_test_data,
             "labels": _h_labels}

# What follows for the manuscript, per analysis: (checks that must be confirmed, optional condition on the
# facts, text). A statement whose checks are not all confirmed is not drawn, and the report says why.
IMPLICATIONS = [
    ("baselines", ["11.1"], None,
     "The headline compares the GNN with a baseline that differs in its settings as well as its score; on its own it "
     "does not show a benefit of the learned score. The comparison with heuristic-fixed should be reported per size, "
     "with the coarse problem sizes beside it."),
    ("baselines", ["11.9"], lambda v, f: f["h_better"] > f["h_worse"],
     "The tuning-free result should be credited to the fixed settings unless the learned score is shown to add to it."),
    ("statistics", ["11.11", "11.15"], None,
     "Every p-value should state the test, its sidedness and its unit (run, seed mean or base instance; pooled or per "
     "size)."),
    ("statistics", ["11.4", "11.7"], None,
     "The 100% at N = 10 should name the instance it rests on and be reported with the retraining mean of Table 3."),
    ("expansion", ["9.2", "9.3"], None,
     "Reported feasibility combines the coarse solution with the error of the lifting step; the loss at expansion "
     "should be reported per arm and size, and any repair after expansion introduced as a named change of method."),
    ("expansion", ["9.5"], None,
     "The statement on distance consistency should be limited to the leg inside a merged pair."),
    ("pre_repair", ["10.1", "10.2"], None,
     "Pre-repair numbers are coarse-space quantities with a counter that ignores the return to the depot; they should be "
     "labelled as such or measured on the original instance with the definition of the final metric. The p = 0.62 "
     "comparison is then a comparison of counts on different problems, not evidence about the learned score."),
    ("merge_order", ["8.1", "8.7", "8.10"], None,
     "Algorithm 1 should state which endpoint is i, that the merge order is chosen after scoring, and how ties are "
     "broken; the seeding should fix the hash salt before the interpreter starts, and the comparisons should be re-run "
     "with deterministic coarsening."),
    ("test_data", ["7.2"], None,
     "The family claim should be worded as evaluation on new time windows over customer maps seen in training."),
    ("test_data", ["7.6", "7.7"], None,
     "The C104 row should be restored, the loader should fail on rows it cannot parse, and the C104 results and the "
     "pair counts that depend on it should be recomputed."),
    ("labels", ["12.1", "12.2"], None,
     "The self-improvement statements should be qualified: the oracle-win rate counts which candidate won one draw. "
     "Labels averaged over several seeds would separate a better merge from a lucky sample."),
]


# --------------------------------------------------------------------------- #
# public: the report
# --------------------------------------------------------------------------- #
@contextlib.contextmanager
def stage_log(run):
    """Run the generation stages with their full output (commands, job summaries) written to
    output/generation_log.txt and only the stage results shown. In re-analysis mode the
    stages only report that they are skipped, which the constructor already said."""
    if run.verify_only:
        with contextlib.redirect_stdout(io.StringIO()):
            yield
        return
    path = run.out / "generation_log.txt"
    with open(path, "w") as fh:
        fh.write(f"# generation log of the results created {run.R['created_utc']}\n")
        tee = _StageTee(fh, sys.stdout, str(run.results_path))
        try:
            with contextlib.redirect_stdout(tee):
                yield
        except BaseException:
            tee.flush()
            fh.flush()
            print("".join(path.read_text().splitlines(keepends=True)[-40:]))
            raise
        finally:
            tee.flush()
    print(f"full log of this generation: {path} (appendix)")


class _StageTee:
    """Writes every line to the log file and passes the stage results to the notebook. A plain object,
    not an io stream: nothing flushes it after stage_log has closed the log file."""
    KEEP = ("stage '", "WARNING")

    def __init__(self, fh, out, results_path):
        self.fh, self.out, self.results_path, self.buf = fh, out, results_path, ""

    def write(self, s):
        self.buf += s
        *lines, self.buf = self.buf.split("\n")
        for ln in lines:
            self.fh.write(ln + "\n")
            if ln.startswith(self.KEEP + (self.results_path,)) or " commands; stage time " in ln:
                self.out.write(ln + "\n")
        return len(s)

    def flush(self):
        if self.fh.closed:
            return
        if self.buf:
            self.write("\n")
        self.fh.flush()


def key_findings(v):
    """What the results file is, and one sentence per analysis, from the verified facts."""
    grid = "; ".join(f"N = {N}: seed{'s' if len(s) > 1 else ''} {', '.join(map(str, s))}" for N, s in sorted(v.grid.items()))
    counts = v.status_counts
    head = (f"**{'Quick grid (QUICK)' if v.quick else 'Published grid (FULL)'}** - {grid}; "
            f"{len(v.R['config']['arms'])} arms on {len(v.D.get('test', []))} test instances. ")
    if v.quick:
        head += ("These results describe the quick grid. They are not a reproduction of the published grid "
                 "(N = 10-100 with three seeds per size); comparisons with published aggregates are reported as "
                 "insufficient data. ")
    head += (f"Results file: status *{v.R['status']}*, created {v.R['created_utc']}. {len(v.checks)} checks: "
             + ", ".join(f"{counts.get(s, 0)} {s}" for s in STATUSES) + ".")
    e = getattr(v, "equiv", None)
    if e:
        head += (f" The instrumented runs reproduce `{e['script']}` on {e['shared_runs']} of {e['script_rows']} shared "
                 f"runs ({len(e['fields'])} fields, {e['mismatches']} mismatches).")
    items = []
    for key, (title, group) in ANALYSES.items():
        if group in v.missing:
            items.append(f"* **{title}.** Not assessed: generation stages not done: {_stages(v.missing[group])}.")
        else:
            items.append(f"* **{title}.** {HEADLINES[key](v, v.facts[group])}")
    display(Markdown(head + "\n\n" + "\n".join(items)))


def analysis(v, key):
    """One analysis of the report: its tables and figure, then the conclusion they support and its
    limitation, then the status of its checks."""
    title, group = ANALYSES[key]
    if group in v.missing:
        display(Markdown(f"No data for this analysis: generation stages not done: {_stages(v.missing[group])}. Its checks are "
                         "reported as insufficient data, and no conclusion is drawn.\n\n*" + _status_line(v, key) + "*"))
        return
    concl, lim = VIEWS[key](v, v.facts[group])
    display(Markdown(concl + "\n\n" + lim + "\n\n*" + _status_line(v, key) + "*"))


def implications(v):
    """What follows for the manuscript, drawn only where its checks are confirmed (summary_findings.md)."""
    lines = []
    for key, (title, group) in ANALYSES.items():
        own = [x for x in IMPLICATIONS if x[0] == key]
        if group in v.missing:
            lines.append(f"* **{title}.** Not drawn: generation stages not done: {_stages(v.missing[group])}.")
            continue
        for _, ids, cond, text in own:
            if not _ok(v, *ids):
                st = {c["check"]: c["status"] for c in v.checks}
                lines.append(f"* **{title}.** Not drawn: " + ", ".join(f"{i} {st.get(i)}" for i in ids
                                                                        if st.get(i) != "confirmed") + ".")
            elif cond is None or cond(v, v.facts[group]):
                lines.append(f"* **{title}.** {text}")
    text = "\n".join(lines)
    (v.out / "summary_findings.md").write_text(f"# Implications ({v.R['mode']} mode)\n\n{text}\n")
    display(Markdown(text))


def checks(v):
    """Every check, grouped by the analysis of the report it belongs to."""
    S = v.summary_table()
    S["evidence"], S["origin of the data"] = zip(*S.check.map(evidence))
    S["analysis"] = S.check.map(analysis_of)
    display(S.pivot_table(index="kind of finding", columns="status", values="check", aggfunc="count", fill_value=0)
            .reindex(columns=list(STATUSES), fill_value=0))
    for key, (title, _) in ANALYSES.items():
        d = S[S.analysis == key]
        display(Markdown(f"**{title}**"))
        _show(d[["check", "stated", "test", "computed", "status", "kind", "evidence", "origin of the data"]]
              .rename(columns={"stated": "statement tested", "test": "measurement"}))


def diagnostics(v):
    """The full tables of every analysis, with the origin of their data."""
    for key, (title, group) in ANALYSES.items():
        display(Markdown(f"**{title}**"))
        if group in v.missing:
            print("no data: generation stages not done: " + ", ".join(v.missing[group]))
            continue
        for name, caption in DIAGNOSTICS[key]:
            _caption(caption, name)
            df = v.tables[name]
            _show(df.drop(columns=[c for c in ("call",) if c in df]), index=not isinstance(df.index, pd.RangeIndex))


def integrity(v):
    """How far the instrumented runs reproduce the released script, and what that does not cover."""
    e = getattr(v, "equiv", None)
    if e is None:
        print("No comparison with the released script: the main grid or the script output is missing.")
        return
    display(Markdown(
        f"{e['shared_runs']} of the {e['script_rows']} runs of `{e['script']}` also appear in the main grid "
        f"(N = {', '.join(map(str, e['sizes']))}; seeds {', '.join(map(str, e['seeds']))}; all three arms; "
        f"PYTHONHASHSEED {', '.join(e['salt'])} in the probe, {', '.join(e['script_salt']) or 'not logged'} for the script). "
        f"On these runs {len(e['fields'])} output fields agree within {e['tolerance']:g}: "
        f"**{e['mismatches']} mismatches**"
        + (f", {e['unmatched_script_rows']} script runs without a counterpart" if e["unmatched_script_rows"] else "")
        + "."))
    _show(v.tables["probe_vs_script"])
    rec = v.R["records"]
    other = [label for key, label in (("flipped", "the reversed-orientation variant"),
                                      ("canonical", "the deterministic tie-break"),
                                      ("salt_runs", "the runs under other hash salts"),
                                      ("labels", "the label re-evaluation (its script needs the unpublished "
                                                 "gnn_model_multiN.pt)"),
                                      ("data_facts", "the data facts")) if rec.get(key)]
    display(Markdown(
        "Not covered by this comparison: "
        + (f"the {e['uncovered_runs']} main-grid runs at N = {', '.join(map(str, e['uncovered_sizes']))}, "
           if e["uncovered_runs"] else "")
        + ", ".join(other)
        + ". It does not test the solver, repair or evaluator themselves, which both sides share."))
    _show(v.tables["accounting_shared_runs"])
    print(f"every solution covers each customer exactly once: {v.coverage_ok}")
    if hasattr(v, "source_summary"):
        ss = v.source_summary
        print(f"cited passages and lines found at the pinned sources: {ss['verified']} of {ss['distinct']}")


def appendix(v, bundle_dir):
    """Review-point map, source verification, the command record and generation log, and a zip of the
    output directory."""
    display(Markdown("**Review points → checks**"))
    _show(v.MAP)
    if hasattr(v, "QCHECK"):
        display(Markdown("**Source verification** (every passage of [1] and line of [2], [3]; "
                         "`tables/source_verification.csv`)"))
        display(v.QCHECK.groupby("source").verified.agg(["sum", "count"]).rename(columns={"sum": "verified", "count": "total"}))
        bad = v.source_summary["not_verified"]
        print("not verified:", bad if bad else "none")
        _show(v.tables["predecessor_first_version"])
    C = pd.DataFrame(v.R["commands"])
    display(Markdown("**Commands of the generation**, as recorded in the results file (`tables/commands.csv`)"))
    v.table("commands", C)
    print(f"{len(C)} commands, {C.seconds.sum() / 60:.1f} min in subprocesses; non-zero exits: {int((C.exit != 0).sum())}")
    display(C.groupby("PYTHONHASHSEED").seconds.agg(["count", "sum"]).round(1))
    log = v.out / "generation_log.txt"
    if log.is_file() and v.R["created_utc"] in log.read_text().split("\n", 1)[0]:
        display(HTML(f"<details><summary>generation log ({escape(str(log))})</summary>"
                     f"<pre>{escape(log.read_text())}</pre></details>"))
    else:
        print("no generation log of this results file in the output directory; "
              "the command record above is the one stored in the results file")
    bundle = shutil.make_archive(str(Path(bundle_dir) / "gnn_audit_output"), "zip", root_dir=v.out)
    print("results bundle:", bundle)
