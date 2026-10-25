"""Measurement probe for the audit notebook.

Runs inside the pinned virtualenv, from the root of the upstream checkout, and
writes one JSON record per job. Upstream code is imported and called, never
edited. Three small mixins add what the audit needs:

* LogMixin      records each coarsening level and each merge; it calls the
                upstream method first and changes nothing.
* FlipMixin     creates every edge with its endpoints in the opposite order
                (the counterfactual of section 8).
* CanonicalMixin puts the edge list in a fixed order before each level's stable
                sort and reconnects neighbours in that order, i.e. ties are broken
                by (score, rank(u), rank(v)) instead of by the hash salt (section 8).

Commands: hashprobe, data, signatures, staged, labels.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT))

from src.coarsening import (SpatioTemporalGraphCoarsener, euclidean,  # noqa: E402
                            graph_from_instance)
from src.datasets import (SOLOMON_FAMILY_HYPERPARAMS, list_solomon,  # noqa: E402
                          load_solomon, train_test_split_instances)
from src.utils import set_global_seed  # noqa: E402

HEUR_FIXED = {"alpha": 1.0, "beta": 1.0, "P": 0.5, "radiusCoeff": 0.5}
SALT = os.environ.get("PYTHONHASHSEED")   # read before any upstream code can overwrite it
_MODEL = {}


# --------------------------------------------------------------------------- #
# coarsener variants
# --------------------------------------------------------------------------- #
def _is_orig(nid):
    return nid.isdigit()


class LogMixin:
    def _log_init(self):
        if not hasattr(self, "levels"):
            self.levels, self.merge_log, self.creation = [], [], {}

    def score_edges(self, graph):
        self._log_init()
        super().score_edges(graph)
        edges = sorted(graph.edges, key=lambda e: e.D_ij)
        rec = {"n_nodes": len(graph.nodes), "n_edges": len(edges),
               "merges_before": len(self.merge_layers)}
        if edges:
            k = math.floor(0.1 * len(edges) * self.radiusCoeff)
            rho = edges[min(k, len(edges) - 1)].D_ij
            region = [e.D_ij for e in edges if e.D_ij <= rho
                      and self.depot_id not in (e.u_id, e.v_id)]
            counts = Counter(region)
            rec.update(n_region=len(region),
                       n_tied=sum(c for c in counts.values() if c > 1))
        # Endpoint rule: build edges run from lower to higher index; an edge added
        # by _reconnect starts at the newer super-node.
        bad = 0
        for e in graph.edges:
            u, v = e.u_id, e.v_id
            if _is_orig(u) and _is_orig(v):
                bad += not (int(u) < int(v))
            elif _is_orig(u):
                bad += 1
            elif not _is_orig(v):
                bad += not (self.creation[u] > self.creation[v])
        rec["endpoint_rule_violations"] = bad
        self.levels.append(rec)

    def _reconnect(self, g, super_node, ni, nj):
        self._log_init()
        self.creation[super_node.id] = len(self.creation)
        tau = euclidean(ni, nj)
        self.merge_log.append({
            "sid": super_node.id, "i": ni.id, "j": nj.id, "tau": tau,
            "midpoint_ok": (abs(super_node.x - (ni.x + nj.x) / 2) < 1e-9
                            and abs(super_node.y - (ni.y + nj.y) / 2) < 1e-9),
            "service_ok": abs(super_node.s - (ni.s + tau + nj.s)) < 1e-9,
        })
        super()._reconnect(g, super_node, ni, nj)


class FlipMixin:
    # upstream src/coarsening.py L209-L214 with the edge created as (nb, super_node)
    def _reconnect(self, g, super_node, ni, nj):
        nbrs = (g.get_neighbors(ni.id) | g.get_neighbors(nj.id)) - {
            ni.id, nj.id, super_node.id, self.depot_id}
        for nb in nbrs:
            if nb in g.nodes:
                g.add_edge(nb, super_node.id, euclidean(super_node, g.nodes[nb]))


class CanonicalMixin:
    def _rank(self, nid):
        return (0, int(nid)) if _is_orig(nid) else (1, self._sn_rank[nid])

    def score_edges(self, graph):
        if not hasattr(self, "_sn_rank"):
            self._sn_rank = {}
        graph.edges.sort(key=lambda e: (self._rank(e.u_id), self._rank(e.v_id)))
        super().score_edges(graph)

    def _reconnect(self, g, super_node, ni, nj):
        if not hasattr(self, "_sn_rank"):
            self._sn_rank = {}
        self._sn_rank[super_node.id] = len(self._sn_rank)
        nbrs = (g.get_neighbors(ni.id) | g.get_neighbors(nj.id)) - {
            ni.id, nj.id, super_node.id, self.depot_id}
        for nb in sorted(nbrs, key=self._rank):
            if nb in g.nodes:
                g.add_edge(super_node.id, nb, euclidean(super_node, g.nodes[nb]))


def _model(path):
    if path not in _MODEL:
        from src.gnn_coarsening import load_model
        _MODEL[path] = load_model(path, device="cpu")
    return _MODEL[path]


def make_coarsener(spec, inst):
    """Graph + coarsener for one job, exactly as src/runner_parallel.py:run_one_job."""
    graph, depot = graph_from_instance(inst)
    orient, tie = spec.get("orient", "upstream"), spec.get("tie", "upstream")
    if orient == "flipped":
        for e in graph.edges:
            e.u_id, e.v_id = e.v_id, e.u_id
    arm = spec["arm"]
    if arm == "gnn":
        from src.gnn_coarsening import GNNCoarsener as base
    else:
        base = SpatioTemporalGraphCoarsener
    mixins = [LogMixin]
    if orient == "flipped":
        mixins.append(FlipMixin)
    if tie == "canonical":
        mixins.append(CanonicalMixin)
    cls = type("Probe" + base.__name__, (*mixins, base), {})
    if arm == "heuristic":
        co = cls(graph=graph, depot_id=depot, **SOLOMON_FAMILY_HYPERPARAMS[inst.family])
    elif arm == "heuristic_fixed":
        co = cls(graph=graph, depot_id=depot, **HEUR_FIXED)
    elif arm == "params":
        co = cls(graph=graph, depot_id=depot, **spec["params"])
    elif arm == "gnn":
        co = cls(graph, P=0.5, radiusCoeff=0.5, depot_id=depot,
                 model=_model(spec["model_path"]), device="cpu")
    else:
        raise ValueError(arm)
    return graph, depot, co


def _h(obj):
    return hashlib.sha1(json.dumps(obj, sort_keys=True).encode()).hexdigest()[:16]


def coarsen_summary(co, coarse, n0):
    layers = [list(x) for x in co.merge_layers]
    parts = sorted(tuple(co.inflate_route([[nid]])[0])
                   for nid in coarse.nodes if nid != co.depot_id)
    lv = co.levels
    m1 = (co.merge_layers[:lv[1]["merges_before"]] if len(lv) > 1 else co.merge_layers)
    level1 = sorted(sorted((i, j)) for _, i, j, _ in m1)
    executed_ne_scored = sum(1 for _, i, _, pi in co.merge_layers if pi.split(" ")[0] != i)
    return {
        "n0": n0, "n_coarse": len(coarse.nodes), "P": co.P,
        "ended_above_target": len(coarse.nodes) > co.P * n0,
        "n_levels_scored": len(lv), "n_merges": len(layers),
        "merges_executed_reversed": executed_ne_scored,
        "sig_layers": _h(sorted(layers)), "sig_partition": _h(parts),
        "sig_level1": _h(level1), "level1_merges": len(level1),
        "levels_with_ties": sum(1 for r in lv if r.get("n_tied", 0) > 0),
        "tied_edges_in_region": sum(r.get("n_tied", 0) for r in lv),
        "region_edges": sum(r.get("n_region", 0) for r in lv),
        "endpoint_rule_violations": sum(r["endpoint_rule_violations"] for r in lv),
        "midpoint_ok": all(m["midpoint_ok"] for m in co.merge_log),
        "service_ok": all(m["service_ok"] for m in co.merge_log),
        "tau_internal_sum": sum(m["tau"] for m in co.merge_log),
    }


# --------------------------------------------------------------------------- #
# evaluation helpers
# --------------------------------------------------------------------------- #
def route_schedule(problem, route, with_return):
    """Time-window violations of one int route; mirrors VRPSolution._count_route_tw_violations
    (src/solvers.py L110-L126) and optionally adds the arrival back at the depot."""
    if not route:
        return 0, False
    p, dep = problem, problem.source_depot
    v = 0
    t = max(0.0, p.time_windows[dep][0]) + p.time_costs[dep][route[0]]
    ready, due = p.time_windows[route[0]]
    v += t > due
    t = max(t, ready) + p.service_times[route[0]]
    for a, b in zip(route[:-1], route[1:]):
        t += p.time_costs[a][b]
        ready, due = p.time_windows[b]
        v += t > due
        t = max(t, ready) + p.service_times[b]
    late_return = (t + p.time_costs[route[-1]][dep]) > p.time_windows[dep][1]
    if with_return:
        v += late_return
    return int(v), bool(late_return)


def coverage(routes, customers, depot):
    visits = [c for r in routes for c in r if c != depot]
    seen = set(visits)
    return {"missing": len(set(customers) - seen), "duplicates": len(visits) - len(seen),
            "unknown": len(seen - set(customers))}


def staged_job(spec):
    from src.experiment import _routes_to_strings, solve_on_graph
    from src.metrics import calculate_route_metrics
    from src.qubo import vrp_problem_from_graph
    from src.solvers import SolverConfig

    t0 = time.perf_counter()
    name, N, seed = spec["instance"], spec["N"], spec["seed"]
    set_global_seed(seed)
    inst = load_solomon(name).truncate(N)
    cfg = SolverConfig("sa", spec.get("num_reads", 200), spec.get("num_sweeps", 200), seed=seed)
    graph, depot, co = make_coarsener(spec, inst)
    n0 = len(graph.nodes)
    coarse, _ = co.coarsen()
    rec = {k: spec[k] for k in ("instance", "N", "seed", "arm")}
    rec.update(instance_cell=inst.name, orient=spec.get("orient", "upstream"), tie=spec.get("tie", "upstream"),
               family=inst.family, family_group=inst.family_group,
               salt=SALT)
    rec.update(coarsen_summary(co, coarse, n0))

    routes, m, sol = solve_on_graph(coarse, depot, inst.capacity, spec.get("solver", "FQS"),
                                    cfg, penalty_mode=spec.get("penalty_mode", "adaptive"),
                                    coarsener=co)
    problem, int_to_id = vrp_problem_from_graph(coarse, depot, inst.capacity)
    customers = [nid for nid in graph.nodes if nid != depot]
    # upstream's own numbers (same fields as run_one_job)
    rec.update({
        "is_feasible": bool(m["is_feasible"]), "is_valid": bool(m["is_valid"]),
        "pre_feasible": bool(m["pre_feasible"]), "pre_total_viol": int(m["pre_total_viol"]),
        "pre_tw_viol": int(m["pre_tw_viol"]), "pre_cap_viol": int(m["pre_cap_viol"]),
        "total_distance": float(m["total_distance"]), "num_vehicles": int(m["num_vehicles"]),
        "tw_violations": int(m["time_window_violations"]),
        "cap_violations": int(m["capacity_violations"]),
        "coarsen_ratio": round(len(coarse.nodes) / max(1, n0), 3),
        "num_qubo_vars": int(m["num_qubo_vars"]),
    })
    # stage 1: raw decode on the coarse problem
    raw = sol.raw_solution
    own = [route_schedule(problem, r, False)[0] for r in raw]
    ups = [sol._count_route_tw_violations(r) for r in raw]
    with_ret = [route_schedule(problem, r, True) for r in raw]
    rep = sol.report(raw)
    rec.update({
        "counter_matches_upstream": own == ups,
        "raw_late_returns": sum(lr for _, lr in with_ret),
        "pre_tw_viol_with_return": sum(v for v, _ in with_ret),
        "pre_feasible_with_return": bool(rep["feasible"] and sum(lr for _, lr in with_ret) == 0),
    })
    # stage 2: raw decode expanded and scored on the original instance
    raw_lifted = co.inflate_route(_routes_to_strings(raw, int_to_id, depot))
    cm_raw = calculate_route_metrics(graph, raw_lifted, depot, inst.capacity)
    cov_raw = coverage(raw_lifted, customers, depot)
    rec.update({
        "pre_orig_tw_viol": int(cm_raw["time_window_violations"]),
        "pre_orig_cap_viol": int(cm_raw["capacity_violations"]),
        "pre_orig_missing": cov_raw["missing"], "pre_orig_duplicates": cov_raw["duplicates"],
        "pre_orig_total_viol": int(cm_raw["time_window_violations"] + cm_raw["capacity_violations"]
                                   + cov_raw["missing"] + cov_raw["duplicates"]),
        "pre_orig_feasible": bool(cm_raw["is_feasible"] and cov_raw["missing"] == 0),
    })
    # stage 3: repaired solution on the coarse graph
    coarse_routes = _routes_to_strings(sol.solution, int_to_id, depot)
    cm_coarse = calculate_route_metrics(coarse, coarse_routes, depot, inst.capacity)
    cov_coarse = coverage(coarse_routes, [n for n in coarse.nodes if n != depot], depot)
    rec.update({
        "coarse_feasible": bool(cm_coarse["is_feasible"] and cov_coarse["missing"] == 0),
        "coarse_feasible_solver_def": bool(sol.check()),
        "coarse_tw_violations": int(cm_coarse["time_window_violations"]),
        "coarse_distance": float(cm_coarse["total_distance"]),
    })
    # stage 4: expanded (lifted) routes as scored by upstream
    cov = coverage(routes, customers, depot)
    rec.update({
        "lifted_is_inflated_repaired": co.inflate_route(coarse_routes) == routes,
        "final_missing": cov["missing"], "final_duplicates": cov["duplicates"],
        "final_unknown": cov["unknown"],
        "coverage_ok": cov["missing"] == 0 and cov["duplicates"] == 0 and cov["unknown"] == 0,
        "distance_gap": float(m["total_distance"] - cm_coarse["total_distance"]
                              - rec["tau_internal_sum"]),
        "lift_lost": bool(rec["coarse_feasible"] and not m["is_feasible"]),
        "seconds": round(time.perf_counter() - t0, 3),
    })
    return rec


def signature_job(spec):
    set_global_seed(spec.get("seed", 0))
    inst = load_solomon(spec["instance"]).truncate(spec["N"])
    graph, _, co = make_coarsener(spec, inst)
    n0 = len(graph.nodes)
    coarse, _ = co.coarsen()
    rec = {k: spec.get(k) for k in ("instance", "N", "seed", "arm", "params", "tag")}
    rec.update(instance_cell=inst.name, orient=spec.get("orient", "upstream"), tie=spec.get("tie", "upstream"),
               salt=SALT)
    rec.update(coarsen_summary(co, coarse, n0))
    return rec


# --------------------------------------------------------------------------- #
# label noise (section 12): scripts/gen_labels_rl.py functions, several SA seeds
# --------------------------------------------------------------------------- #
def _gen_labels_module():
    spec = importlib.util.spec_from_file_location("gen_labels_rl", ROOT / "scripts" / "gen_labels_rl.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def label_case(case):
    import numpy as np
    from src.baselines import solve_cvrptw_ortools
    from src.gnn_coarsening import GNNCoarsener
    from src.solvers import SolverConfig
    mod = _gen_labels_module()
    inst = load_solomon(case["instance"]).truncate(case["N"])
    ref = solve_cvrptw_ortools(inst, time_limit_s=case["ortools_time"])
    ref_cost = ref.cost if ref.feasible else None
    model = _model(case["model_path"])

    def candidates():
        # the pool of scripts/gen_labels_rl.py:build_pool (L67-L81), rebuilt per seed
        for p in mod.SWEEP:
            g, d = graph_from_instance(inst)
            yield "heur", SpatioTemporalGraphCoarsener(graph=g, depot_id=d, **p)
        g, d = graph_from_instance(inst)
        yield "gnn", GNNCoarsener(g, P=mod.GNN_P, radiusCoeff=mod.GNN_RC, depot_id=d,
                                  model=model, device="cpu", explore=0.0)
        for sub_seed in case["sub_seeds"]:
            g, d = graph_from_instance(inst)
            yield "gnn", GNNCoarsener(g, P=mod.GNN_P, radiusCoeff=mod.GNN_RC, depot_id=d,
                                      model=model, device="cpu", explore=0.35,
                                      rng=np.random.default_rng(sub_seed))

    out = []
    for seed in case["seeds"]:
        cfg = SolverConfig("sa", case["num_reads"], case["num_sweeps"], seed=seed)
        for idx, (tag, co) in enumerate(candidates()):
            m, _ = mod.eval_rollout(inst, co, cfg)
            out.append({"instance": case["instance"], "N": case["N"], "seed": seed,
                        "cand": idx, "tag": tag, "reward": float(mod.reward(m, ref_cost)),
                        "is_feasible": bool(m["is_feasible"]),
                        "pre_feasible": bool(m["pre_feasible"]),
                        "pre_total_viol": int(m["pre_total_viol"]),
                        "total_distance": float(m["total_distance"]),
                        "ref_cost": ref_cost,
                        "sig_layers": _h(sorted(map(list, co.merge_layers)))})
    return out


def label_plan(spec):
    """Reproduce the random stream of gen_labels_rl.py main(): one generator seeded with
    --seed, consumed by build_pool in the order N x training instance."""
    import numpy as np
    mod = _gen_labels_module()
    train, _ = train_test_split_instances()
    rng = np.random.default_rng(spec["rng_seed"])
    cases = []
    for N in spec["Ns"]:
        for name in train:
            subs = [int(rng.integers(1 << 30)) for _ in range(spec["n_gnn"])]
            if name in spec["instances"]:
                cases.append({"instance": name, "N": N, "sub_seeds": subs,
                              **{k: spec[k] for k in ("seeds", "num_reads", "num_sweeps",
                                                      "ortools_time", "model_path")}})
    assert len(mod.SWEEP) == 5
    return cases


# --------------------------------------------------------------------------- #
# data facts (section 7)
# --------------------------------------------------------------------------- #
def data_facts():
    import numpy as np
    names = list_solomon()
    train, test = train_test_split_instances()
    insts = {n: load_solomon(n) for n in names}
    out = {"names": names, "train": train, "test": test,
           "n_customers": {n: int(i.n_customers) for n, i in insts.items()},
           "family": {n: i.family for n, i in insts.items()},
           "all_coords_integer": bool(all(np.all(np.mod(i.coords, 1) == 0) for i in insts.values()))}
    sizes = [10, 20, 40, 50, 60, 80, 100]
    share = {}
    for t in test:
        share[t] = {}
        for N in sizes:
            if insts[t].n_customers < N:
                continue
            a = insts[t].truncate(N)
            geo = np.c_[a.coords, a.demand]
            same = []
            for s in train:
                if insts[s].n_customers < N:
                    continue
                b = insts[s].truncate(N)
                if np.array_equal(geo, np.c_[b.coords, b.demand]):
                    same.append(s)
            tw_diff = [s for s in same if not np.array_equal(
                np.c_[insts[s].truncate(N).ready_time, insts[s].truncate(N).due_time],
                np.c_[a.ready_time, a.due_time])]
            share[t][N] = {"same_geometry": same, "with_different_windows": tw_diff}
    out["shared_geometry"] = share
    series = {}
    for cls in ("C", "R", "RC"):
        a, b = insts[f"{cls}101"], insts[f"{cls}201"]
        series[cls] = {"coords_equal": bool(np.array_equal(a.coords, b.coords)),
                       "demand_equal": bool(np.array_equal(a.demand, b.demand))}
    out["series_pairs"] = series
    r101, r104 = insts["R101"], insts["R104"]
    out["R101_vs_R104_customer1"] = {
        k: [float(getattr(r101, k)[1]), float(getattr(r104, k)[1])]
        for k in ("ready_time", "due_time", "service_time", "demand")}
    out["R101_vs_R104_customer1"]["xy"] = [r101.coords[1].tolist(), r104.coords[1].tolist()]
    # beta term of Eq. (1) on every ordered customer pair of the test instances
    from src.coarsening import Edge
    pos = late = pos_and_late = 0
    for t in test:
        g, depot = graph_from_instance(insts[t])
        co = SpatioTemporalGraphCoarsener(g, alpha=0.0, beta=1.0, P=0.5, radiusCoeff=1.0,
                                          depot_id=depot)
        ids = [n for n in g.nodes if n != depot]
        for u in ids:
            for v in ids:
                if u == v:
                    continue
                b = co.merge_score(g, Edge(u, v, 0.0))
                ni, nj = g.nodes[u], g.nodes[v]
                is_late = ni.t + ni.s + euclidean(ni, nj) > nj.l
                pos += b > 0
                late += is_late
                pos_and_late += (b > 0) and is_late
    out["beta_term"] = {"pairs_positive": pos, "pairs_late": late,
                        "pairs_positive_and_late": pos_and_late}
    # the time-window counter and the final evaluator on a one-customer route whose
    # return to the depot is late (depot closes at 10, back at 4 + 5 + 4 = 13)
    from src.coarsening import Graph, Node
    from src.metrics import calculate_route_metrics
    from src.qubo import vrp_problem_from_graph
    from src.solvers import VRPSolution
    g = Graph()
    g.add_node(Node("0", 0.0, 0.0, 0.0, 0.0, 10.0, 0.0))
    g.add_node(Node("1", 4.0, 0.0, 5.0, 0.0, 100.0, 1.0))
    g.build_complete_edges()
    prob, _ = vrp_problem_from_graph(g, "0", 10.0)
    sol = VRPSolution(prob, {}, [1], solution=[[1]])
    out["depot_return_example"] = {
        "solver_counter_violations": int(sol._count_route_tw_violations([1])),
        "solver_report_feasible": bool(sol.report([[1]])["feasible"]),
        "final_evaluator_violations": int(calculate_route_metrics(
            g, [["0", "1", "0"]], "0", 10.0)["time_window_violations"])}
    c = insts["C104"]
    out["C104_loaded"] = {"coords": c.coords.tolist(), "ready": c.ready_time.tolist(),
                          "due": c.due_time.tolist(), "demand": c.demand.tolist(),
                          "service": c.service_time.tolist()}
    return out


# --------------------------------------------------------------------------- #
def _map(fn, items, workers):
    if workers <= 1 or len(items) <= 1:
        return [fn(x) for x in items]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(fn, items, chunksize=1))


def _write(path, rows):
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["hashprobe", "data", "signatures", "staged", "labels"])
    ap.add_argument("--jobs")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=1)
    a = ap.parse_args()
    if a.command == "hashprobe":
        probe = "SN_12_7"
        before = hash(probe)
        set_global_seed(0)
        res = {"salt_at_start": SALT, "hash_before": before,
               "hash_after_set_global_seed": hash(probe),
               "env_after_set_global_seed": os.environ.get("PYTHONHASHSEED"),
               "set_iteration": list({str(i) for i in range(12)})}
        Path(a.out).write_text(json.dumps(res))
        return
    if a.command == "data":
        Path(a.out).write_text(json.dumps(data_facts()))
        return
    jobs = json.loads(Path(a.jobs).read_text())
    if a.command == "labels":
        cases = label_plan(jobs)
        rows = [r for part in _map(label_case, cases, a.workers) for r in part]
    else:
        fn = staged_job if a.command == "staged" else signature_job
        rows = _map(fn, jobs, a.workers)
    _write(a.out, rows)


if __name__ == "__main__":
    main()
