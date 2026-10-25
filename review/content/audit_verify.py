"""Verification stage of the GNN results audit.

Reads output/results.json (written by audit_runner.py) and nothing produced in the
kernel. It checks that the file is complete and belongs to the current configuration
and inputs, recomputes every analysis and statistic from the stored observations,
compares the results with the reference values, assigns each check a status, and writes
verification.json, summary_checks.csv/.md and the analysis tables to output/.

It never runs a solver, the coarsening, OR-Tools or the pipeline environment.
"""
from __future__ import annotations

import ast
import json
import math
import re
import unicodedata
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from audit_runner import (GENERATION_INPUTS, SCHEMA_VERSION, STAGES, config_digest,
                          sha256_file, write_json_atomic)

STATUSES = ("confirmed", "not confirmed", "insufficient data")
KINDS = {"implementation": "property or defect of the code or data",
         "evidence": "what the evidence establishes about a claim",
         "scope": "limitation of the experiment",
         "reproduction": "reported result recomputed"}
TOPICS = {7: "Data and the scope of the generalisation test",
          8: "Reproducibility and the choice of merges",
          9: "Super-node geometry and route expansion",
          10: "Metric definitions and the role of repair",
          11: "GNN versus the baselines, and the statistics",
          12: "Stability of the training labels"}
# stages whose observations each section reads
NEEDS = {7: ["sources", "data_and_scripts"],
         8: ["sources", "data_and_scripts", "main_grid", "edge_order", "hash_salts"],
         9: ["sources", "main_grid"],
         10: ["data_and_scripts", "main_grid"],
         11: ["sources", "data_and_scripts", "main_grid", "edge_order", "hash_salts"],
         12: ["sources", "label_stability"]}

# Every check: what is tested, the statement it is tested against ({name} = a reference
# value), the kind of finding, and the review points it bears on.
CHECKS = {
    "7.1": ("every fourth instance of each series is held out",
            "family-stratified split of {n_train} training and {n_test} held-out test instances", "reproduction", [9]),
    "7.2": ("test instances other than C104 share their customers with training instances at every size",
            "a test instance has the same customers as training instances; only time windows are new", "scope", [9, 18]),
    "7.3": ("sizes at which C104 shares its customers with training instances",
            "C104 is the exception from {c104_shared_below} customers on", "implementation", [9, 19]),
    "7.4": ("customer 2 of R101 and R104", "only the ready time and due date differ", "scope", [9, 18]),
    "7.5": ("series 1 and 2 share coordinates, by class", "R and RC: shared; C1 and C2: not shared", "scope", [9]),
    "7.6": ("data rows of C104.csv that do not parse as numbers",
            "row {c104_malformed_line} malformed; its ready-time field is not a number", "implementation", [19]),
    "7.7": ("customers the loader returns",
            "C104 loads with {c104_customers} customers, the others with {other_customers}", "implementation", [7, 19]),
    "7.8": ("position of the skipped row among the customers", "the {c104_skipped_customer}th customer",
            "implementation", [19]),
    "7.9": ("smallest N at which the loaded C104 differs from the file's first N customers",
            "every N ≥ 36 (evaluated: 40, 50, 60, 80); excluded at 100", "implementation", [19]),
    "7.10": ("instance-size pairs at N = 80, 100 and over the seven sizes",
             "{pairs_large_n} rather than 26; {pairs_all_sizes} rather than 91", "implementation", [7, 19]),
    "7.11": ("the loader's self-check", "checks six sample instances, not C104", "implementation", [19]),
    "7.12": ("C104 line 38 in the predecessor's data, first version", "the malformed row is inherited",
             "implementation", [19]),
    "8.1": ("set_global_seed changes string hashing in the running process",
            "the salt set after start-up has no effect", "implementation", [1, 3, 12]),
    "8.2": ("coarsening identical across SA seeds (main grid)",
            "the coarsening is a function of instance, settings and salt", "reproduction", [3]),
    "8.3": ("integer coordinates; coarsenings with tied scores below the admission threshold", "ties occur",
            "implementation", [3, 15]),
    "8.4": ("cases whose coarsening differs across salts {salts}",
            "identical inputs and seeds do not give identical coarsenings", "implementation", [1, 3, 12, 15]),
    "8.5": ("runs and arm-size rates whose feasibility changes with the salt",
            "a reported comparison can change without any change of method", "implementation", [1, 3]),
    "8.6": ("cases with more than one coarsening across salts {salts}, deterministic tie-break",
            "a total sort key and sorted neighbour iteration remove the dependence on the salt",
            "implementation", [12]),
    "8.7": ("edge endpoint rule", "the score is evaluated with i = the edge's stored first endpoint",
            "implementation", [2, 15]),
    "8.8": ("merges executed opposite to their scored direction",
            "a pair can be ranked by a direction in which it is not merged", "implementation", [1, 2, 15]),
    "8.9": ("β term positive while the arrival is after the window closes (all ordered customer pairs, test instances)",
            "the β term charges waiting (earliness), never lateness", "implementation", [2]),
    "8.10": ("first-level admitted merges change when edges are created in the opposite order",
             "which pairs are admitted depends on the order of edge creation, in every arm", "implementation", [1, 2]),
    "8.11": ("label-pool coarsenings (heuristic settings, training instances) that change with edge order",
             "the training labels are built from coarsenings with the same dependence", "implementation", [2]),
    "8.12": ("score-only sort, set-based reconnection and scoring orientation in [3]; in-process seed only in [2]",
             "the constructs are inherited; the ineffective seeding is new", "implementation", [1, 2, 3]),
    "9.1": ("super-node at the midpoint of its pair, service time sᵢ + τᵢⱼ + sⱼ", "midpoint location (Eq. 2)",
            "reproduction", [4]),
    "9.2": ("the routes scored after expansion are the expanded coarse routes, without repair",
            "nothing repairs the expanded route", "implementation", [1, 4, 13]),
    "9.3": ("runs feasible on the coarse graph and infeasible after expansion",
            "a coarse-feasible route can be infeasible after expansion", "implementation", [1, 4, 13]),
    "9.4": ("loss at expansion in every arm", "the lifting step affects all arms", "implementation", [1, 4, 13]),
    "9.5": ("expanded length differs from coarse length plus internal legs",
            "distance consistency holds only for the leg inside a pair (p. 4, section 3.1)", "evidence", [4, 15]),
    "9.6": ("midpoint super-node in the predecessor's first version", "the construction is inherited",
            "implementation", [4]),
    "10.1": ("coarse problem size differs between GNN and heuristic-fixed on the same run",
             "pre-repair counts of different arms refer to different problems", "evidence", [5, 14]),
    "10.2": ("pre-repair counter and final evaluator on a late return to the depot",
             "the pre-repair counter does not check the return to the depot", "implementation", [5, 14]),
    "10.3": ("runs where pre-repair feasibility on the coarse problem and on the original instance disagree",
             "pre-repair metrics do not measure the original problem", "evidence", [5, 14]),
    "10.4": ("pre-repair violations, GNN vs heuristic-fixed, pooled over N (script method)",
             "p={p_pre_repair_pooled} pooled over N (p. 21, section 8)", "reproduction", [5]),
    "11.1": ("run_scaling_100.py: arms run, arms tested, sidedness",
             "all three arms run; only GNN vs the tuned heuristic tested, one-sided", "evidence", [1, 6]),
    "11.2": ("overall feasibility, GNN vs tuned heuristic (all published sizes)",
             "{gnn_overall_pct}% vs. {heuristic_overall_pct}% overall (p. 11, section 5.4)", "reproduction", [1, 6]),
    "11.3": ("N = 80, 100: tuned heuristic, and heuristic-fixed above it",
             "tuned {heuristic_large_n_pct[0]}-{heuristic_large_n_pct[1]}%, below the untuned fixed configuration (p. 13, section 5.4)",
             "reproduction", [1, 6, 16]),
    "11.4": ("N = 10: instances below 100% in any arm (unmodified run_scorer_ablation.py, salt 0)",
             "the margin at N = 10 rests on {n10_margin_instance} alone: heuristic arms 0/3, GNN 3/3", "evidence", [1, 8, 16]),
    "11.5": ("N = 10: GNN overall and tuned heuristic on R-type (unmodified script, salt 0)",
             "{gnn_n10_pct:.0f}% at N = 10; {heuristic_n10_r_pct:.0f}% on R-type for the tuned heuristic (p. 11, section 5.3)",
             "reproduction", [8]),
    "11.6": ("test instances, and R-type among them",
             "{n_test} instances, {n_test_r_type} R-type: 80% on R is four of five, and one instance moves a rate by 7.7 points",
             "scope", [8]),
    "11.7": ("distance between 100% and the retraining mean {retraining_mean_pct}%, in instances of the test set",
             "one to two instances out of 13", "evidence", [8]),
    "11.8": ("what run_scorer_ablation.py saves and prints", "every run is saved; only means are printed",
             "evidence", [1, 8]),
    "11.9": ("GNN and heuristic-fixed end at different coarse sizes on the same case",
             "equal settings do not guarantee equal problem sizes", "evidence", [6, 17]),
    "11.10": ("coarsenings that stop above the target P·n₀", "a level ends early when no admissible pair is found",
              "implementation", [6]),
    "11.11": ("run_scaling_100.py: seed aggregation, pooling of sizes, sidedness",
              "mean over seeds where Section 4 states a median; N = 80 and 100 pooled into one test; one-sided",
              "evidence", [7, 11]),
    "11.12": ("the other scaling and ablation tests", "one-sided as well", "evidence", [7]),
    "11.13": ("pairs available at N = 80 and N = 100", "{pairs_n80} at N = 80 and {pairs_n100} at N = 100",
              "reproduction", [7]),
    "11.14": ("N = 80, 100 pairs where the GNN is better or tied (script computation)",
              "better or tied on all {large_n_better_or_tied} available pairs (p. 13, section 5.4)", "reproduction", [7]),
    "11.15": ("one-sided p is half the two-sided p", "halving a two-sided p-value is what a one-sided test does",
              "evidence", [7]),
    "11.16": ("all sizes aggregated to base instances: which test gives the published p",
              "p = {p_all_sizes} (p. 13, section 5.4), consistent with a one-sided test", "evidence", [7]),
    "11.17": ("a script whose size lists cover N = 10, 20, 40, 50, 60, 80, 100",
              "no released script produces the seven-size headline", "evidence", [7, 11]),
    "11.18": ("each code property changes feasibility outcomes in every arm",
              "the code properties act on every arm regardless of the scorer", "implementation", [1]),
    "12.1": ("one solver configuration and seed for all candidates; the best kept by '>'",
             "each candidate solved once with the same seed; the best is kept", "implementation", [10, 17]),
    "12.2": ("cases where the label from one evaluation is not the best candidate on average",
             "the labels partly record selection noise", "evidence", [10, 17]),
    "12.3": ("share of reward variance explained by 2·feasible + pre-feasible (squared correlation)",
             "the reward is dominated by binary feasibility indicators", "implementation", [10]),
    "12.4": ("label's reward at seed 0 minus its mean reward at the other seeds",
             "the maximum over single evaluations is biased toward a favourable sample", "evidence", [10]),
}


class ResultsError(RuntimeError):
    """The results file is missing, incomplete, or does not belong to this configuration."""


# --------------------------------------------------------------------------- #
# statistics
# --------------------------------------------------------------------------- #
def wilson(k, n, z=1.959964):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (100 * (c - h), 100 * (c + h))


def sign_test(better, worse):
    n = better + worse
    return float("nan") if n == 0 else stats.binomtest(better, n, 0.5).pvalue


def wilcoxon_p(x, y, alternative):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) == 0 or np.all(x == y):
        return float("nan")
    return float(stats.wilcoxon(x, y, alternative=alternative)[1])


def paired(df, a, b, col, keys=("instance", "N", "seed")):
    """Rows present for both arms a and b, with their `col` side by side."""
    piv = df[df.arm.isin([a, b])].pivot_table(index=list(keys), columns="arm", values=col, aggfunc="first")
    return piv.dropna(subset=[a, b]).reset_index()


def pct(k, n):
    if not n:
        return "n/a"
    v = 100 * k / n
    return f"{v:.1f}%" if 0 < v < 10 else f"{v:.0f}%"


# --------------------------------------------------------------------------- #
# source normalisation (passages keep the Markdown/LaTeX of the text they were taken
# from; the PDF has typeset glyphs; both are reduced to the same plain form)
# --------------------------------------------------------------------------- #
TEX = [("\\%", "%"), ("\\pm", "±"), ("\\in", "∈"), ("\\dots", "…"), ("\\ldots", "…"),
       ("\\cdot", "·"), ("\\to", "→"), ("\\times", "×"), ("\\leq", "≤"), ("\\geq", "≥"),
       ("\\{", "{"), ("\\}", "}")]
TEX_DROP = re.compile(r"\\(?:!|,|;|:|mathbf|mathrm|text|textit|tfrac|left|right|big|Big)")


def norm_text(s):
    s = unicodedata.normalize("NFKC", s)
    for a, b in TEX:
        s = s.replace(a, b)
    s = TEX_DROP.sub("", s)
    for a, b in (("–", "-"), ("—", "-"), ("−", "-"), ("’", "'"), ("‘", "'"), ("“", '"'), ("”", '"')):
        s = s.replace(a, b)
    s = re.sub(r"-\s*\n\s*", "", s)
    s = re.sub(r"[${}*\\]", "", s)
    return re.sub(r"\s+", "", s).lower()


# --------------------------------------------------------------------------- #
# static analysis of the release's scripts
# --------------------------------------------------------------------------- #
def script_facts(rel, src, arms):
    """Wilcoxon calls (with the arms their arguments are built from and the alternative),
    seed aggregations in groupby expressions, and the sizes written into a script."""
    tree = ast.parse(src)
    assigns = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            names = {n.id for t in node.targets for n in ast.walk(t) if isinstance(n, ast.Name)}
            strs = {c.value for c in ast.walk(node.value) if isinstance(c, ast.Constant) and isinstance(c.value, str)}
            assigns.append((node.lineno, names, strs))
    tests, aggs, sizes = [], [], set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            fname = getattr(node.func, "id", None) or getattr(node.func, "attr", None)
            if fname == "wilcoxon":
                alt = next((k.value.value for k in node.keywords if k.arg == "alternative"), "two-sided")
                used = {n.id for a in node.args for n in ast.walk(a) if isinstance(n, ast.Name)}
                found = {c.value for a in node.args for c in ast.walk(a) if isinstance(c, ast.Constant) and c.value in arms}
                for ln, names, strs in assigns:
                    if ln < node.lineno and names & used:
                        found |= strs & set(arms)
                tests.append({"script": rel, "line": node.lineno, "alternative": alt,
                              "arms compared": ", ".join(sorted(found)) or "-", "call": ast.get_source_segment(src, node)})
            elif fname in ("mean", "median"):
                if "groupby" in (ast.get_source_segment(src, node) or ""):
                    aggs.append({"script": rel, "line": node.lineno, "aggregation": fname})
            elif fname == "add_argument" and node.args and isinstance(node.args[0], ast.Constant) \
                    and node.args[0].value in ("--Ns", "--N"):
                for k in node.keywords:
                    if k.arg == "default":
                        v = ast.literal_eval(k.value)
                        sizes |= set(v if isinstance(v, list) else [v])
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.List) and \
                any(isinstance(t, ast.Name) and t.id == "Ns" for t in node.targets):
            sizes |= {c.value for c in node.value.elts if isinstance(c, ast.Constant)}
        if isinstance(node, ast.For) and isinstance(node.target, ast.Name) and node.target.id == "N" \
                and isinstance(node.iter, ast.List):
            sizes |= {c.value for c in node.iter.elts if isinstance(c, ast.Constant)}
    return {"tests": tests, "aggregations": aggs, "sizes": sorted(sizes), "src": src}


# --------------------------------------------------------------------------- #
# loading
# --------------------------------------------------------------------------- #
def load_results(path, audit_dir, expected_config=None, allow_incomplete=False):
    """Read results.json and refuse it unless it is complete (or allow_incomplete), has the
    current schema, was made with the expected configuration and the same generation
    inputs, and is pinned to the configured commits."""
    path = Path(path)
    if not path.is_file():
        raise ResultsError(f"{path} does not exist: run the generation stages first "
                           "(or upload the results file of an earlier run to output/)")
    try:
        R = json.loads(path.read_text())
    except json.JSONDecodeError as e:
        raise ResultsError(f"{path} is not valid JSON ({e}); it may be from an interrupted write") from None
    if R.get("schema_version") != SCHEMA_VERSION:
        raise ResultsError(f"schema_version {R.get('schema_version')!r}, expected {SCHEMA_VERSION}")
    problems = []
    if expected_config is not None and R.get("config_sha256") != config_digest(expected_config):
        diff = sorted(k for k in set(expected_config) | set(R.get("config", {}))
                      if k != "workers" and json.dumps(expected_config.get(k), sort_keys=True, default=str)
                      != json.dumps(R.get("config", {}).get(k), sort_keys=True, default=str))
        problems.append(f"configuration differs from the current one (mode {R.get('mode')}; keys: {diff})")
    for f in GENERATION_INPUTS:
        p = Path(audit_dir) / f
        if p.is_file() and R.get("inputs", {}).get(f) != sha256_file(p):
            problems.append(f"{f} differs from the version that produced the results")
    prov, src = R.get("provenance", {}), R.get("config", {}).get("sources", {})
    if R.get("stages", {}).get("prepare", {}).get("status") == "done":
        if prov.get("upstream", {}).get("commit") != src.get("upstream_ref"):
            problems.append("upstream commit differs from the pinned one")
        if prov.get("predecessor", {}).get("commit") != src.get("predecessor_ref"):
            problems.append("predecessor commit differs from the pinned one")
    if problems:
        raise ResultsError("results do not belong to this configuration: " + "; ".join(problems))
    not_done = [s for s in STAGES if R.get("stages", {}).get(s, {}).get("status") != "done"]
    if R.get("status") != "complete" or not_done:
        msg = f"results are incomplete: stages not done {not_done}"
        if not allow_incomplete:
            raise ResultsError(msg + " (pass allow_incomplete=True to verify what is there; "
                               "sections without their data are reported as insufficient data)")
        print("WARNING:", msg)
    return R


# --------------------------------------------------------------------------- #
# the verification
# --------------------------------------------------------------------------- #
class Verification:
    def __init__(self, R, audit_dir, out):
        self.R, self.out = R, Path(out)
        self.audit_dir = Path(audit_dir)
        c = self.cfg = R["config"]
        self.arms, self.seeds = c["arms"], c["sa_seeds"]
        self.grid = {int(k): v for k, v in c["grid"].items()}
        self.published = c["published_sizes"]
        self.quick = R["mode"] == "QUICK"
        self.ref = R["references"]["values"]
        self.checks, self.tables, self.facts, self.missing = [], {}, {}, {}
        (self.out / "tables").mkdir(parents=True, exist_ok=True)
        rec = R["records"]
        df = lambda k: pd.DataFrame(rec.get(k) or [])
        self.ST, self.FL, self.SALT_RUNS = df("main"), df("flipped"), df("salt_runs")
        self.CTRL, self.SIG, self.CAN = df("salt_control"), df("signatures"), df("canonical")
        self.LP, self.LAB = df("label_pool"), df("labels")
        for d in (self.SIG, self.CAN):
            if len(d):
                d["salt"] = d.salt.astype(int)
        self.D = rec.get("data_facts", {})
        self.scripts = rec.get("scripts", {})
        self.ABL = pd.DataFrame(self.scripts.get("scripts/run_scorer_ablation.py", {}).get("rows", []))
        sc = self.scripts.get("scripts/run_scaling_100.py")
        self.SC100 = pd.DataFrame(sc["rows"]) if sc else None
        self.src = R.get("sources", {})
        self.files = self.src.get("files", {})
        rem = json.loads((self.audit_dir / "audit_remarks.json").read_text())
        self.remarks = rem["remarks"]
        self.pointers = rem["predecessor_pointers"]
        self.fragments = json.loads((self.audit_dir / "audit_fragments.json").read_text())["fragments"]

    # ---- recording ----------------------------------------------------------
    def record(self, check, computed, status, note="", **fmt):
        test, stated, kind, remarks = CHECKS[check]
        assert status in STATUSES, status
        self.checks[:] = [c for c in self.checks if c["check"] != check]
        self.checks.append({"check": check, "section": int(check.split(".")[0]),
                            "test": test.format(**{**self.ref, **fmt}), "computed": str(computed),
                            "stated": stated.format(**self.ref), "status": status, "kind": kind,
                            "remarks": list(remarks), "note": note})

    def table(self, name, df, index=False):
        df.to_csv(self.out / "tables" / f"{name}.csv", index=index)
        self.tables[name] = df
        return df

    def published_grid(self, sizes):
        """True when every size in `sizes` was run with the published seeds."""
        return all(self.grid.get(N) == self.seeds for N in sizes)

    def scored(self, ok, sizes=None):
        """Status of a comparison with a published aggregate over `sizes`."""
        if sizes is not None and not self.published_grid(sizes):
            return "insufficient data"
        return "confirmed" if ok else "not confirmed"

    def or_quick(self, ok):
        """A check whose effect may not appear on the quick grid."""
        return "confirmed" if ok else ("insufficient data" if self.quick else "not confirmed")

    def lines(self, src, path):
        return self.files[f"{src}:{path}"].split("\n")

    def boot_rate(self, df, col, cluster="instance"):
        """Mean of `col` (in %) with a cluster-bootstrap 95% CI over `cluster`."""
        s = self.cfg["statistics"]
        g = df.groupby(cluster)[col].agg(["sum", "count"])
        su, n = g["sum"].to_numpy(float), g["count"].to_numpy(float)
        est = 100 * su.sum() / n.sum()
        if len(g) < 2:
            return est, float("nan"), float("nan")
        rng = np.random.default_rng(s["boot_seed"])
        idx = rng.integers(0, len(g), size=(s["boot_reps"], len(g)))
        reps = 100 * su[idx].sum(1) / n[idx].sum(1)
        lo, hi = np.percentile(reps, [2.5, 97.5])
        return est, lo, hi

    def rate_table(self, df, col, by=("arm", "N")):
        rows = []
        for key, d in df.groupby(list(by)):
            est, lo, hi = self.boot_rate(d, col)
            rows.append({**dict(zip(by, key if isinstance(key, tuple) else (key,))),
                         "runs": len(d), "instances": d.instance.nunique(),
                         f"{col} %": round(est, 1), "CI95 low": round(lo, 1), "CI95 high": round(hi, 1)})
        return pd.DataFrame(rows)

    @staticmethod
    def accounting(df, label, keys=("instance", "N", "seed"), arm_col="arm"):
        arms = sorted(df[arm_col].unique())
        per_key = df.groupby(list(keys))[arm_col].nunique()
        row = {"runs": label, "records": len(df), "base instances": df.instance.nunique(),
               "sizes": ",".join(map(str, sorted(df.N.unique()))),
               "seeds": ",".join(map(str, sorted(df.seed.unique()))) if "seed" in df else "-",
               "arms": ", ".join(arms),
               "paired across arms": f"{int((per_key == len(arms)).sum())}/{len(per_key)} keys"}
        if "coverage_ok" in df:
            row["every customer exactly once"] = f"{int(df.coverage_ok.sum())}/{len(df)}"
        return pd.DataFrame([row])

    # ---- run ----------------------------------------------------------------
    def run(self):
        done = {s for s in STAGES if self.R["stages"].get(s, {}).get("status") == "done"}
        if {"sources", "prepare"} <= done:
            self.source_check()
        if {"main_grid", "data_and_scripts"} <= done:
            self.integrity()
        for sec in TOPICS:
            lacking = [s for s in NEEDS[sec] if s not in done]
            if lacking:
                self.missing[sec] = lacking
                for cid in [c for c in CHECKS if int(c.split(".")[0]) == sec]:
                    self.record(cid, "not computed", "insufficient data", note=f"stages not done: {lacking}",
                                salts=self.cfg["alt_salts"])
                continue
            self.facts[sec] = getattr(self, f"s{sec}")()
        self.review_map()
        self.write()
        return self

    # ---- source check -------------------------------------------------------
    def source_check(self):
        pages = [norm_text(p) for p in self.src["pdf_pages"]]
        rows = []
        for r in self.remarks:
            for q in r["quotes"]:
                if q["kind"] == "article":
                    found = [i + 1 for i, p in enumerate(pages) if norm_text(q["text"]) in p]
                    rows.append({"remark": r["number"], "source": "[1]",
                                 "location": f"p. {q['page']}" + (f", {q['section']}" if q.get("section") else ""),
                                 "quotation": q["text"], "found on page(s)": ",".join(map(str, found)) or "-",
                                 "verified": q["page"] in found})
                else:
                    line = self.lines(q["src"], q["path"])[q["line"] - 1]
                    rows.append({"remark": r["number"], "source": f"[{q['src']}]",
                                 "location": f"{q['path']}:L{q['line']}", "quotation": q["text"],
                                 "found on page(s)": "", "verified": q["text"].strip() in line})
        for p in self.pointers:
            lines = self.lines(3, p["path"])
            frags = [f for f in self.fragments if f["path"] == p["path"] and p["lo"] <= f["line"] <= p["hi"]]
            ok = bool(frags) and all(f["fragment"] in lines[f["line"] - 1] for f in frags)
            loc = f"{p['path']}:L{p['lo']}" + (f"-L{p['hi']}" if p["hi"] != p["lo"] else "")
            rows.append({"remark": p["remark"], "source": "[3]", "location": loc,
                         "quotation": "; ".join(f["construct"] for f in frags) + " (cited, not quoted)",
                         "found on page(s)": "", "verified": ok})
        self.QCHECK = self.table("source_verification", pd.DataFrame(rows))
        uq = self.QCHECK.drop_duplicates(["source", "location", "quotation"])
        self.source_summary = {"verified": int(uq.verified.sum()), "distinct": len(uq),
                               "by_source": {s: int((uq.source == s).sum()) for s in ("[1]", "[2]", "[3]")},
                               "not_verified": uq[~uq.verified][["source", "location", "quotation"]].to_dict("records")}
        pf = self.src["predecessor_first_commit"]
        self.table("predecessor_first_version", pd.DataFrame(
            [{"file": f, "commit that added it": c, "is the pinned commit": c == self.cfg["sources"]["predecessor_ref"]}
             for f, c in pf.items()]))

    # ---- integrity of the observations ---------------------------------------
    def integrity(self):
        ST, ABL = self.ST, self.ABL
        fields = ["is_feasible", "pre_feasible", "pre_total_viol", "num_qubo_vars", "coarsen_ratio",
                  "total_distance", "tw_violations", "cap_violations", "num_vehicles"]
        m = ST.merge(ABL.rename(columns={"instance": "instance_cell", "condition": "arm"}),
                     on=["instance_cell", "N", "seed", "arm"], suffixes=("", "_script"))
        tol = 1e-9
        diff = {f: int((~np.isclose(m[f].astype(float), m[f + "_script"].astype(float), rtol=0, atol=tol)).sum())
                for f in fields}
        self.table("probe_vs_script", pd.DataFrame(
            [{"field": f, "runs compared": len(m), "tolerance": f"|probe − script| ≤ {tol:g}", "mismatches": d}
             for f, d in diff.items()]))
        self.equiv_ok = len(m) == len(ABL) and sum(diff.values()) == 0
        key = ["instance_cell", "N", "seed", "arm"]
        shared = set(map(tuple, m[key].to_numpy().tolist()))
        rest = ST[[tuple(r) not in shared for r in ST[key].to_numpy().tolist()]]
        self.equiv = {
            "script": "scripts/run_scorer_ablation.py", "script_rows": len(ABL), "shared_runs": len(m),
            "main_grid_runs": len(ST), "fields": fields, "tolerance": tol,
            "mismatches": int(sum(diff.values())), "unmatched_script_rows": len(ABL) - len(m),
            "sizes": sorted(int(x) for x in m.N.unique()), "seeds": sorted(int(x) for x in m.seed.unique()),
            "arms": sorted(m.arm.unique()), "salt": sorted(str(x) for x in m.salt.unique()),
            "script_salt": sorted({c.get("PYTHONHASHSEED") for c in self.R.get("commands", [])
                                   if c.get("command", "").split()[1:] == ["scripts/run_scorer_ablation.py"]}),
            "uncovered_runs": len(rest), "uncovered_sizes": sorted(int(x) for x in rest.N.unique())}
        parts = [self.accounting(ST, "main grid (salt 0)")]
        if len(self.FL):
            parts.append(self.accounting(self.FL, "opposite orientation (salt 0)"))
        if len(self.SALT_RUNS):
            parts.append(self.accounting(self.SALT_RUNS, "re-solved under other salts"))
        if len(self.CTRL):
            parts.append(self.accounting(self.CTRL, "salt control sample"))
        self.table("accounting_shared_runs", pd.concat(parts, ignore_index=True))
        self.coverage_ok = bool(ST.coverage_ok.all() and (self.FL.coverage_ok.all() if len(self.FL) else True))

    # ---- 7. data and the scope of the generalisation test --------------------
    def s7(self):
        D, ref = self.D, self.ref
        TRAIN, TEST, NCUST = D["train"], D["test"], D["n_customers"]
        by_fam = {}
        for n in D["names"]:
            by_fam.setdefault(D["family"][n], []).append(n)
        rule = sorted(n for f in sorted(by_fam) for i, n in enumerate(sorted(by_fam[f])) if i % 4 == 3)
        self.table("split_by_series", pd.DataFrame(
            [{"series": f, "instances": len(v), "held out": ", ".join(n for n in sorted(v) if n in TEST)}
             for f, v in sorted(by_fam.items())]))
        self.record("7.1", f"{len(TRAIN)} training / {len(TEST)} test; the rule reproduces the split: {rule == sorted(TEST)}",
                    "confirmed" if rule == sorted(TEST) and len(TRAIN) == ref["n_train"] else "not confirmed")

        rows = []
        for t in TEST:
            for N, v in D["shared_geometry"][t].items():
                rows.append({"test instance": t, "N": int(N), "training instances, same customers": len(v["same_geometry"]),
                             "of which different windows": len(v["with_different_windows"])})
        SG = pd.DataFrame(rows)
        self.table("shared_customer_maps",
                   SG.pivot(index="test instance", columns="N", values="training instances, same customers"), index=True)
        others = SG[SG["test instance"] != "C104"]
        c104 = SG[SG["test instance"] == "C104"]
        windows_differ = bool((others["of which different windows"] == others["training instances, same customers"]).all())
        self.record("7.2", f"{int((others['training instances, same customers'] > 0).sum())}/{len(others)} instance-size cells; "
                           f"time windows differ in all of them: {windows_differ}",
                    "confirmed" if (others["training instances, same customers"] > 0).all() and windows_differ else "not confirmed")
        shared_at = sorted(c104[c104["training instances, same customers"] > 0].N)
        lim = ref["c104_shared_below"]
        self.record("7.3", "N = " + ", ".join(map(str, shared_at)) + f" of {sorted(c104.N)}",
                    "confirmed" if shared_at and max(shared_at) < lim
                    and set(c104.N) - set(shared_at) == {n for n in c104.N if n >= lim} else "not confirmed")

        c = D["R101_vs_R104_customer1"]
        self.table("customer_R101_R104", pd.DataFrame(c, index=["R101 (training)", "R104 (test)"]).T, index=True)
        same_rest = c["xy"][0] == c["xy"][1] and c["demand"][0] == c["demand"][1] and c["service_time"][0] == c["service_time"][1]
        differ_tw = c["ready_time"][0] != c["ready_time"][1] and c["due_time"][0] != c["due_time"][1]
        self.record("7.4", f"same position, demand, service time: {same_rest}; ready time and due date differ: {differ_tw}",
                    "confirmed" if same_rest and differ_tw else "not confirmed")
        SP = pd.DataFrame(D["series_pairs"]).T.rename(columns={"coords_equal": "series 1 and 2: same coordinates",
                                                               "demand_equal": "same demands"})
        self.table("series_pairs", SP, index=True)
        same_series = SP["series 1 and 2: same coordinates"]
        self.record("7.5", "; ".join(f"{k}: {v}" for k, v in same_series.items()),
                    "confirmed" if bool(same_series["R"]) and bool(same_series["RC"]) and not bool(same_series["C"])
                    else "not confirmed")

        RAW = self.lines(2, "data/solomon/C1/C104.csv")
        hdr = next(i for i, l in enumerate(RAW) if "CUST NO." in l)
        data_rows = [(n, l.split(",")) for n, l in enumerate(RAW, start=1) if n > hdr + 1 and l.strip()]

        def is_number(s):
            try:
                float(s)
                return True
            except ValueError:
                return False

        bad_rows = [(n, f) for n, f in data_rows if not (len(f) == 7 and all(is_number(x) for x in f))]
        self.table("c104_rows_not_parsed", pd.DataFrame(
            [{"file line": n, "fields": len(f), "content": ",".join(f)} for n, f in bad_rows]))
        ln, f = bad_rows[0] if bad_rows else (None, [])
        self.record("7.6", f"line {ln}: ready time {f[4]!r}" if bad_rows else "none",
                    "confirmed" if [b[0] for b in bad_rows] == [ref["c104_malformed_line"]] and not is_number(f[4])
                    else "not confirmed")
        others_n = {k: v for k, v in NCUST.items() if k != "C104"}
        self.record("7.7", f"C104: {NCUST['C104']}; the other {len(others_n)} instances: {sorted(set(others_n.values()))}",
                    "confirmed" if NCUST["C104"] == ref["c104_customers"] and set(others_n.values()) == {ref["other_customers"]}
                    else "not confirmed")
        customers = data_rows[1:]                                  # the first data row is the depot
        pos = next((k for k, (l, _) in enumerate(customers, start=1) if l == ref["c104_malformed_line"]), None)
        self.record("7.8", f"customer {pos} (CUST NO. {customers[pos - 1][1][0]})" if pos else "line not found",
                    "confirmed" if pos == ref["c104_skipped_customer"] else "not confirmed")

        L = D["C104_loaded"]
        file_xy = [(float(f_[1]), float(f_[2]), float(f_[3])) for _, f_ in customers]
        load_xy = [(x, y, d) for (x, y), d in zip(L["coords"][1:], L["demand"][1:])]
        TR = pd.DataFrame([{"N": N, "loaded C104 = first N customers of the file":
                            (load_xy[:N] == file_xy[:N]) if N <= len(load_xy) else None,
                            "used in the experiments": "yes" if N <= len(load_xy) else "excluded (fewer than N customers)"}
                           for N in self.published])
        self.table("c104_size_series", TR)
        first_bad = next((N for N in range(1, len(load_xy) + 1) if load_xy[:N] != file_xy[:N]), None)
        affected = [r.N for r in TR.itertuples() if r[2] is not None and not r[2]]
        self.record("7.9", f"N = {first_bad}; evaluated sizes affected: {affected}; N = 100 excluded",
                    "confirmed" if first_bad == ref["c104_skipped_customer"] and affected == ref["c104_affected_sizes"]
                    else "not confirmed")
        pairs_large = sum(NCUST[i] >= N for i in TEST for N in (80, 100))
        pairs_all = sum(NCUST[i] >= N for i in TEST for N in self.published)
        self.record("7.10", f"{pairs_large} and {pairs_all}",
                    "confirmed" if (pairs_large, pairs_all) == (ref["pairs_large_n"], ref["pairs_all_sizes"]) else "not confirmed")

        sc = self.src["loader_selfcheck"]
        sample = ast.literal_eval(next(l.split("=", 1)[1].strip() for l in self.lines(2, "src/datasets.py")
                                       if l.strip().startswith("sample =")))
        self.record("7.11", f"exit {sc['exit']}; instances checked: {sample}",
                    "confirmed" if sc["exit"] == 0 and "C104" not in sample else "not confirmed")
        pred_ref = self.cfg["sources"]["predecessor_ref"]
        same_line = self.lines(3, "solomon_dataset/C1/C104.csv")[37] == RAW[37]
        first = self.src["predecessor_first_commit"].get("solomon_dataset/C1/C104.csv") == pred_ref
        self.record("7.12", f"identical: {same_line}; added in the pinned commit: {first}",
                    "confirmed" if same_line and first else "not confirmed")
        return {"shared_min": int(others["training instances, same customers"].min()),
                "shared_max": int(others["training instances, same customers"].max()),
                "same_series": {k: bool(v) for k, v in same_series.items()},
                "c104_customers": NCUST["C104"], "bad_line": ln, "bad_field": f[4] if bad_rows else None,
                "first_bad": first_bad, "pairs_large": pairs_large, "pairs_all": pairs_all}

    # ---- 8. reproducibility and the choice of merges -------------------------
    def s8(self):
        ST, SIG, CAN, FL, LP = self.ST, self.SIG, self.CAN, self.FL, self.LP
        ARMS, c = self.arms, self.cfg
        salts = [c["main_salt"]] + c["alt_salts"]
        K = ["instance", "N", "seed", "arm"]
        H = self.table("hash_probe", pd.DataFrame(
            [{"PYTHONHASHSEED at start": h["label"], "hash before": h["hash_before"],
              "hash after set_global_seed(0)": h["hash_after_set_global_seed"],
              "variable after set_global_seed": h["env_after_set_global_seed"],
              "set iteration order": " ".join(h["set_iteration"])} for h in self.R["records"]["hashprobe"]]))
        unchanged = int((H["hash before"] == H["hash after set_global_seed(0)"]).sum())
        same0 = H.iloc[0]["hash before"] == H.iloc[1]["hash before"]
        differ = len({H.iloc[k]["hash before"] for k in (0, 2, 3, 4)}) == 4
        self.record("8.1", f"hash unchanged by the call in {unchanged}/5 processes; salt-0 processes agree: {same0}; "
                           f"other salts differ: {differ}",
                    "confirmed" if unchanged == 5 and same0 and differ else "not confirmed")

        nseed = ST.groupby(["instance", "N", "arm"]).sig_layers.nunique()
        self.record("8.2", f"{int((nseed == 1).sum())}/{len(nseed)} cases", "confirmed" if (nseed == 1).all() else "not confirmed")
        CO = ST.drop_duplicates(["instance", "N", "arm"])
        TI = CO.groupby("arm").agg(coarsenings=("instance", "size"),
                                   with_ties=("levels_with_ties", lambda s: int((s > 0).sum())),
                                   tied_edges=("tied_edges_in_region", "sum"),
                                   edges_below_threshold=("region_edges", "sum")).reindex(ARMS)
        self.table("ties", TI.reset_index())
        integer = self.D["all_coords_integer"]
        self.record("8.3", f"integer: {integer}; " + "; ".join(f"{a}: {r.with_ties}/{r.coarsenings}" for a, r in TI.iterrows()),
                    "confirmed" if integer and TI.with_ties.sum() > 0 else "not confirmed")

        base_sig = SIG[SIG.salt == c["main_salt"]].set_index(["instance", "N", "arm"]).sig_layers
        nd = SIG.groupby(["arm", "N", "instance"]).sig_layers.nunique().rename("distinct").reset_index()
        SV = (nd.assign(varies=nd.distinct > 1).groupby(["arm", "N"])
              .agg(cases=("instance", "size"), cases_that_vary=("varies", "sum"), max_distinct=("distinct", "max"))
              .reset_index())
        self.table("salt_variation", SV)
        salt_vary = SV.groupby("arm").cases_that_vary.sum().reindex(ARMS)
        salt_cases = SV.groupby("arm").cases.sum().reindex(ARMS)
        self.record("8.4", "; ".join(f"{a}: {int(v)}/{int(salt_cases[a])}" for a, v in salt_vary.items()),
                    self.or_quick(salt_vary.sum() > 0), salts=salts)

        ctrl = self.CTRL.merge(ST, on=K, suffixes=("", "_0"))
        ctrl_ok = all((ctrl[f] == ctrl[f + "_0"]).all() for f in ("is_feasible", "pre_total_viol", "total_distance"))
        SR_ = self.SALT_RUNS
        if len(SR_):
            sm = SR_.merge(ST[K + ["is_feasible"]], on=K, suffixes=("", "_salt0"))
            salt_flips = (sm.is_feasible != sm.is_feasible_salt0).groupby(sm.arm).sum().reindex(ARMS).fillna(0).astype(int)
        else:
            salt_flips = pd.Series(0, index=ARMS)
        rates = []
        for s in salts:
            d = ST.copy()
            if s != c["main_salt"] and len(SR_):
                rs = SR_[SR_.salt.astype(int) == s].set_index(K).is_feasible
                d["is_feasible"] = [rs.get(k, v) for k, v in zip(d.set_index(K).index, d.is_feasible)]
            rates.append(d.groupby(["arm", "N"]).is_feasible.mean().mul(100).rename(f"salt {s}"))
        SR = pd.concat(rates, axis=1)
        SR["range (pp)"] = SR.max(axis=1) - SR.min(axis=1)
        self.table("feasibility_by_salt", SR.round(1).reset_index())
        self.record("8.5", f"{int(salt_flips.sum())} runs; {int((SR['range (pp)'] > 0).sum())} rates move "
                           f"(largest {SR['range (pp)'].max():.1f} pp)",
                    self.or_quick(salt_flips.sum() > 0 and ctrl_ok))

        cv = CAN.groupby(["arm", "instance", "N"]).sig_layers.nunique()
        CV = cv.groupby("arm").agg(cases="size", cases_that_vary=lambda s: int((s > 1).sum())).reindex(ARMS)
        self.table("canonical_tie_break", CV.reset_index())
        self.record("8.6", f"{int(CV.cases_that_vary.sum())}/{int(CV.cases.sum())}",
                    "confirmed" if CV.cases_that_vary.sum() == 0 else "not confirmed", salts=salts)

        O = CO.groupby("arm").agg(coarsenings=("instance", "size"), merges=("n_merges", "sum"),
                                  executed_reversed=("merges_executed_reversed", "sum"),
                                  endpoint_rule_violations=("endpoint_rule_violations", "sum")).reindex(ARMS)
        O["reversed % of merges"] = (100 * O.executed_reversed / O.merges).round(1)
        O["Wilson CI95"] = [tuple(round(v, 1) for v in wilson(k, n)) for k, n in zip(O.executed_reversed, O.merges)]
        self.table("orientation", O.reset_index())
        self.record("8.7", f"{int(O.endpoint_rule_violations.sum())} violations in {int(O.coarsenings.sum())} coarsenings",
                    "confirmed" if O.endpoint_rule_violations.sum() == 0 else "not confirmed")
        self.record("8.8", "; ".join(f"{a}: {r.executed_reversed}/{r.merges} ({r['reversed % of merges']}%)" for a, r in O.iterrows()),
                    "confirmed" if (O.executed_reversed > 0).all() else "not confirmed")
        B = self.D["beta_term"]
        self.table("beta_term", pd.DataFrame([B]))
        self.record("8.9", f"{B['pairs_positive_and_late']} of {B['pairs_positive']} pairs with a positive term",
                    "confirmed" if B["pairs_positive_and_late"] == 0 and B["pairs_positive"] > 0 else "not confirmed")

        KC = ["instance", "N", "arm"]
        oc = ST.drop_duplicates(KC).merge(FL.drop_duplicates(KC), on=KC, suffixes=("", "_flip"))
        oc["level1_changed"] = oc.sig_level1 != oc.sig_level1_flip
        oc["coarsening_changed"] = oc.sig_partition != oc.sig_partition_flip
        OF = oc.groupby("arm").agg(coarsenings=("instance", "size"), level1_changed=("level1_changed", "sum"),
                                   coarsening_changed=("coarsening_changed", "sum")).reindex(ARMS)
        self.table("orientation_counterfactual", OF.reset_index())
        fm2 = FL.merge(ST[K + ["is_feasible", "sig_partition"]], on=K, suffixes=("_flip", ""))
        fm2["coarsening changed"] = fm2.sig_partition != fm2.sig_partition_flip
        Dd = (fm2.groupby(["arm", "coarsening changed"])
              .apply(lambda d: pd.Series({"runs": len(d), "feasible %": round(100 * d.is_feasible.mean(), 1),
                                          "feasible %, opposite order": round(100 * d.is_feasible_flip.mean(), 1),
                                          "better": int((d.is_feasible_flip > d.is_feasible).sum()),
                                          "worse": int((d.is_feasible_flip < d.is_feasible).sum())}),
                     include_groups=False).reset_index())
        Dd[["runs", "better", "worse"]] = Dd[["runs", "better", "worse"]].astype(int)
        Dd["sign test p"] = [round(sign_test(b, w), 4) for b, w in zip(Dd.better, Dd.worse)]
        self.table("orientation_feasibility", Dd)
        orient_flips = (fm2[fm2["coarsening changed"]].assign(x=lambda d: d.is_feasible != d.is_feasible_flip)
                        .groupby("arm").x.sum().reindex(ARMS))
        self.record("8.10", "; ".join(f"{a}: {int(r.level1_changed)}/{int(r.coarsenings)}" for a, r in OF.iterrows()),
                    "confirmed" if (OF.level1_changed > 0).all() else "not confirmed")

        sweep = self.R["stages"]["edge_order"]["meta"]["label_pool_settings"]
        lp = LP.pivot_table(index=["instance", "N", "tag"], columns="orient", values="sig_partition", aggfunc="first")
        pool_changed = int((lp.upstream != lp.flipped).sum())
        self.table("label_pool_orientation", pd.DataFrame(
            {"setting": range(len(sweep)), "params": [json.dumps(p) for p in sweep],
             "coarsenings changed": (lp.upstream != lp.flipped).groupby(level="tag").sum().values,
             "of": lp.groupby(level="tag").size().values}))
        self.record("8.11", f"{pool_changed}/{len(lp)}", "confirmed" if pool_changed > 0 else "not confirmed")

        g = self.src["grep_pythonhashseed"]
        p8 = self.QCHECK[(self.QCHECK.source == "[3]") & self.QCHECK.location.str.contains("L272|L362|L405|L436")]
        p8u = p8.drop_duplicates("location")
        self.table("predecessor_constructs", p8u.drop(columns="remark"))
        inherited = bool(p8.verified.all()) and not g["[3]"] and "src/utils.py" in g["[2] src/"]
        self.record("8.12", f"{int(p8u.verified.sum())}/{p8.location.nunique()} lines hold the construct; "
                            f"[3] {'has no' if not g['[3]'] else 'has'} PYTHONHASHSEED",
                    "confirmed" if inherited else "not confirmed")
        self._flips = {"salt": salt_flips, "orient": orient_flips}
        return {"salts": salts, "TI": TI, "salt_vary": salt_vary, "salt_cases": salt_cases,
                "salt_flips": int(salt_flips.sum()), "canonical_vary": int(CV.cases_that_vary.sum()),
                "reversed_min": O["reversed % of merges"].min(), "reversed_max": O["reversed % of merges"].max(),
                "OF": OF, "pool_changed": pool_changed, "pool_cases": len(lp), "ctrl_runs": len(ctrl), "ctrl_ok": ctrl_ok}

    # ---- 9. super-node geometry and route expansion ---------------------------
    def s9(self):
        ST, ARMS = self.ST, self.arms
        self.table("supernode_construction", pd.DataFrame([{
            "runs": len(ST), "super-node at the midpoint (every merge)": int(ST.midpoint_ok.sum()),
            "service time sᵢ+τᵢⱼ+sⱼ (every merge)": int(ST.service_ok.sum()),
            "scored routes = expanded repaired coarse routes": int(ST.lifted_is_inflated_repaired.sum())}]))
        self.record("9.1", f"{int(ST.midpoint_ok.sum())}/{len(ST)} runs",
                    "confirmed" if ST.midpoint_ok.all() and ST.service_ok.all() else "not confirmed")
        self.record("9.2", f"{int(ST.lifted_is_inflated_repaired.sum())}/{len(ST)} runs",
                    "confirmed" if ST.lifted_is_inflated_repaired.all() else "not confirmed")
        rows = []
        for (arm, N), d in ST.groupby(["arm", "N"]):
            k, n = int(d.lift_lost.sum()), int(d.coarse_feasible.sum())
            rows.append({"arm": arm, "N": N, "runs": len(d), "feasible on the coarse graph": n,
                         "feasible after expansion": int(d.is_feasible.sum()), "lost at expansion": k,
                         "lost % of coarse-feasible": round(100 * k / n, 1) if n else float("nan"),
                         "Wilson CI95": tuple(round(v, 1) for v in wilson(k, n))})
        LIFT = self.table("loss_at_expansion", pd.DataFrame(rows))
        self.table("feasibility_by_stage", pd.concat(
            [self.rate_table(ST[ST.arm == arm], col, by=("N",)).assign(arm=arm, stage=col)
             .rename(columns={f"{col} %": "feasible %"}) for arm in ARMS for col in ("coarse_feasible", "is_feasible")],
            ignore_index=True))
        lost = LIFT.groupby("arm")["lost at expansion"].sum().reindex(ARMS)
        self.record("9.3", f"{int(lost.sum())} of {int(ST.coarse_feasible.sum())} coarse-feasible runs",
                    "confirmed" if lost.sum() > 0 else "not confirmed")
        self.record("9.4", "; ".join(f"{a}: {int(v)}" for a, v in lost.items()), self.or_quick((lost > 0).all()))
        GAP = ST.assign(gap_nonzero=ST.distance_gap.abs() > 1e-6).groupby("arm").agg(
            runs=("instance", "size"), gap_nonzero=("gap_nonzero", "sum"), median_gap=("distance_gap", "median"),
            min_gap=("distance_gap", "min"), max_gap=("distance_gap", "max")).reindex(ARMS).round(2)
        self.table("distance_gap", GAP.reset_index())
        self.record("9.5", f"{int(GAP.gap_nonzero.sum())}/{int(GAP.runs.sum())} runs",
                    "confirmed" if GAP.gap_nonzero.sum() > 0 else "not confirmed")
        p9 = self.QCHECK[(self.QCHECK.source == "[3]") & self.QCHECK.location.str.contains("L462")].drop_duplicates("location")
        self.record("9.6", f"{int(p9.verified.sum())}/{len(p9)} lines", "confirmed" if p9.verified.all() else "not confirmed")
        by_n = LIFT.groupby("N")[["lost at expansion", "feasible on the coarse graph"]].sum()
        worst = (by_n["lost at expansion"] / by_n["feasible on the coarse graph"]).idxmax()
        self._lost = lost
        return {"lost": lost, "coarse_feasible": int(ST.coarse_feasible.sum()), "worst_N": worst,
                "worst_lost": int(by_n.loc[worst, "lost at expansion"]),
                "worst_coarse": int(by_n.loc[worst, "feasible on the coarse graph"]),
                "gap_nonzero": int(GAP.gap_nonzero.sum()), "gap_runs": int(GAP.runs.sum())}

    # ---- 10. metric definitions and the role of repair ------------------------
    def s10(self):
        ST, ARMS = self.ST, self.arms
        K = ["instance", "N", "seed", "arm"]
        nc = ST.pivot_table(index=K[:3], columns="arm", values="n_coarse", aggfunc="first")
        PS = self.table("problem_size_by_arm_pair", pd.DataFrame(
            [{"pair of arms": f"{a} vs {b}", "runs": len(nc), "different coarse problem size": int((nc[a] != nc[b]).sum())}
             for a, b in (("gnn", "heuristic_fixed"), ("gnn", "heuristic"), ("heuristic", "heuristic_fixed"))]))
        size_diff = int(PS.iloc[0]["different coarse problem size"])
        self.record("10.1", f"{size_diff}/{len(nc)} runs", "confirmed" if size_diff > 0 else "not confirmed")
        E = self.D["depot_return_example"]
        self.table("depot_return_example", pd.DataFrame([E]))
        self.table("late_returns", ST.groupby("arm").agg(
            runs=("instance", "size"), raw_routes_returning_late=("raw_late_returns", "sum"),
            runs_with_a_late_return=("raw_late_returns", lambda s: int((s > 0).sum())),
            counter_matches_authors=("counter_matches_upstream", "sum")).reindex(ARMS).reset_index())
        ok = E["solver_counter_violations"] == 0 and E["final_evaluator_violations"] == 1 and ST.counter_matches_upstream.all()
        self.record("10.2", f"counter {E['solver_counter_violations']}, final evaluator {E['final_evaluator_violations']}; "
                            f"{int(ST.raw_late_returns.sum())} late returns among the raw routes of the main grid",
                    "confirmed" if ok else "not confirmed")
        cols = ["pre_feasible", "pre_feasible_with_return", "pre_orig_feasible"]
        PR = ST.groupby(["arm", "N"])[cols].mean().mul(100).round(1)
        PR["runs"] = ST.groupby(["arm", "N"]).size()
        PR["coarse ≠ original"] = ST.assign(x=ST.pre_feasible != ST.pre_orig_feasible).groupby(["arm", "N"]).x.sum()
        self.table("pre_repair_measurements", PR.reset_index())
        dis = int((ST.pre_feasible != ST.pre_orig_feasible).sum())
        self.record("10.3", f"{dis}/{len(ST)}", "confirmed" if dis > 0 else "not confirmed")

        def ablation_test(df, col):
            # scripts/run_scorer_ablation.py L123-L134, N = "all"
            d = df[df.N.isin([10, 20, 40])].assign(instance=lambda x: x.instance_cell, condition=lambda x: x.arm)
            agg = (d.groupby(["instance", "condition"])[col].mean().unstack("condition")
                   .dropna(subset=["gnn", "heuristic_fixed"]))
            gi, hfi = agg["gnn"].to_numpy(), agg["heuristic_fixed"].to_numpy()
            return {"cells": len(gi), "one-sided p (script)": wilcoxon_p(hfi, gi, "greater"),
                    "two-sided p": wilcoxon_p(hfi, gi, "two-sided")}

        m_script = re.search(r"N=all:.*p\(inst\)=([0-9.]+)", self.scripts["scripts/run_scorer_ablation.py"]["stdout"])
        P62 = pd.DataFrame([{"measurement": "coarse problem (authors)", **ablation_test(ST, "pre_total_viol")},
                            {"measurement": "original instance", **ablation_test(ST, "pre_orig_total_viol")}])
        P62["printed by the script"] = [float(m_script.group(1)) if m_script else None, None]
        self.table("pre_repair_test", P62.round(4))
        p62, p62o = P62.iloc[0]["one-sided p (script)"], P62.iloc[1]["one-sided p (script)"]
        self.record("10.4", f"p = {p62:.3f} (script prints {m_script.group(1) if m_script else 'n/a'}); "
                            f"original instance p = {p62o:.3f}",
                    self.scored(round(p62, 2) == self.ref["p_pre_repair_pooled"], [10, 20, 40]))
        self.table("pre_repair_on_original_instance", self.rate_table(ST, "pre_orig_feasible"))
        return {"size_diff": size_diff, "runs": len(nc), "late_returns": int(ST.raw_late_returns.sum()),
                "disagree": dis, "n": len(ST), "p62": p62, "p62_orig": p62o}

    # ---- 11. GNN versus the baselines, and the statistics ----------------------
    def s11(self):
        ST, ARMS, ref = self.ST, self.arms, self.ref
        TEST, NCUST, FAM = self.D["test"], self.D["n_customers"], self.D["family"]
        scripts = {k[2:]: v for k, v in self.files.items() if k.startswith("2:scripts/") and k.endswith(".py")}
        FACTS = {k: script_facts(k, v, ARMS) for k, v in sorted(scripts.items())}
        F100 = FACTS["scripts/run_scaling_100.py"]
        self.table("scaling_script_tests", pd.DataFrame(F100["tests"]))
        arms_run = next(ast.literal_eval(l.split("=", 1)[1].strip()) for l in F100["src"].splitlines()
                        if l.strip().startswith("conditions ="))
        tested = sorted({t["arms compared"] for t in F100["tests"]})
        alts = sorted({t["alternative"] for t in F100["tests"]})
        self.record("11.1", f"runs {arms_run}; tests {tested}; alternative {alts}",
                    "confirmed" if set(arms_run) == set(ARMS) and tested == ["gnn, heuristic"] and alts == ["greater"]
                    else "not confirmed")

        FEAS = self.table("feasibility_by_arm_and_size", self.rate_table(ST, "is_feasible"))

        def compare_arms(df, a, b, col="is_feasible"):
            rows = []
            for N, d in df.groupby("N"):
                p = paired(d, a, b, col)
                better, worse = int((p[a] > p[b]).sum()), int((p[a] < p[b]).sum())
                per_inst = p.groupby("instance")[[a, b]].median()
                rows.append({"N": N, "comparison": f"{a} vs {b}", "pairs (runs)": len(p),
                             "first arm %": round(100 * p[a].mean(), 1), "second arm %": round(100 * p[b].mean(), 1),
                             "better": better, "worse": worse, "sign test p (two-sided)": round(sign_test(better, worse), 4),
                             "instances": len(per_inst),
                             "Wilcoxon p (two-sided, median over seeds)": round(wilcoxon_p(per_inst[a], per_inst[b], "two-sided"), 4)})
            return pd.DataFrame(rows)

        CMP = self.table("paired_arm_comparisons", pd.concat(
            [compare_arms(ST, "gnn", "heuristic_fixed"), compare_arms(ST, "gnn", "heuristic"),
             compare_arms(ST, "heuristic_fixed", "heuristic")], ignore_index=True))
        OV = self.table("feasibility_overall", self.rate_table(ST, "is_feasible", by=("arm",)))
        ov = OV.set_index("arm")
        inside = lambda arm, v: ov.loc[arm, "CI95 low"] <= v <= ov.loc[arm, "CI95 high"]
        self.record("11.2", f"GNN {ov.loc['gnn', 'is_feasible %']}% [{ov.loc['gnn', 'CI95 low']}, {ov.loc['gnn', 'CI95 high']}]; "
                            f"tuned {ov.loc['heuristic', 'is_feasible %']}% [{ov.loc['heuristic', 'CI95 low']}, "
                            f"{ov.loc['heuristic', 'CI95 high']}]",
                    self.scored(inside("gnn", ref["gnn_overall_pct"]) and inside("heuristic", ref["heuristic_overall_pct"]),
                                self.published))
        fb = FEAS[FEAS.N.isin([80, 100])].set_index(["arm", "N"])
        ok = (len(fb) == 6 and all(any(fb.loc[("heuristic", N), "CI95 low"] <= v <= fb.loc[("heuristic", N), "CI95 high"]
                                       for v in ref["heuristic_large_n_pct"]) for N in (80, 100))
              and all(fb.loc[("heuristic_fixed", N), "is_feasible %"] > fb.loc[("heuristic", N), "is_feasible %"]
                      for N in (80, 100)))
        self.record("11.3", "; ".join(f"N={N}: tuned {fb.loc[('heuristic', N), 'is_feasible %']}%, fixed "
                                      f"{fb.loc[('heuristic_fixed', N), 'is_feasible %']}%"
                                      for N in (80, 100) if ("heuristic", N) in fb.index),
                    self.scored(ok, [80, 100]))

        base_name = lambda cell: re.sub(r"\[:\d+\]$", "", cell)
        ABL = self.ABL
        n10 = ABL[ABL.N == 10].assign(instance=lambda d: d.instance.map(base_name))
        T10 = (n10.pivot_table(index="instance", columns="condition", values="is_feasible", aggfunc="sum")
               .reindex(columns=ARMS).astype(int))
        self.table("n10_by_instance", T10.rename(columns=lambda a: f"{a}: feasible seeds of {len(self.seeds)}"), index=True)
        below = T10[(T10 < len(self.seeds)).any(axis=1)]
        found = "; ".join(f"{i}: " + ", ".join(f"{a} {below.loc[i, a]}/3" for a in ARMS) for i in below.index)
        self.n10_same_as_main = (ST[ST.N == 10].pivot_table(index="instance", columns="arm", values="is_feasible", aggfunc="sum")
                                 .reindex(columns=ARMS).astype(int).equals(T10))
        mi = ref["n10_margin_instance"]
        self.record("11.4", found or "none",
                    "confirmed" if list(below.index) == [mi] and tuple(below.loc[mi]) == (0, 0, len(self.seeds))
                    else "not confirmed")
        a10 = n10.assign(type=lambda d: d.instance.map(lambda i: re.sub(r"\d", "", FAM[i])))
        R10 = a10.groupby(["condition", "type"]).is_feasible.mean().mul(100).round(1).unstack("type").reindex(ARMS)
        R10["all"] = a10.groupby("condition").is_feasible.mean().mul(100).round(1).reindex(ARMS)
        self.table("n10_by_type", R10, index=True)
        self.record("11.5", f"GNN {R10.loc['gnn', 'all']}%; tuned heuristic on R {R10.loc['heuristic', 'R']}%",
                    "confirmed" if R10.loc["gnn", "all"] == ref["gnn_n10_pct"] and R10.loc["heuristic", "R"] == ref["heuristic_n10_r_pct"]
                    else "not confirmed")
        n_r = sum(1 for i in TEST if re.sub(r"\d", "", FAM[i]) == "R")
        self.record("11.6", f"{len(TEST)} test, {n_r} R-type; one instance = {100 / len(TEST):.1f} points",
                    "confirmed" if len(TEST) == ref["n_test"] and n_r == ref["n_test_r_type"] else "not confirmed")
        gap = (100 - ref["retraining_mean_pct"]) / (100 / len(TEST))
        self.record("11.7", f"{gap:.2f} instances", "confirmed" if 1 <= gap <= 2 else "not confirmed",
                    note="87.2 is quoted from Table 3 of [1]; the retraining is not run")
        out_abl = self.scripts["scripts/run_scorer_ablation.py"]["stdout"]
        named = [i for i in TEST if re.search(rf"\b{i}\b", out_abl)]
        n_expected = 3 * len(TEST) * len(self.seeds) * 3
        self.record("11.8", f"{len(ABL)} rows saved (expected {n_expected}); instances printed: {named or 'none'}",
                    "confirmed" if len(ABL) == n_expected and not named else "not confirmed")

        CO = ST.drop_duplicates(["instance", "N", "arm"])
        SZ = (ST.groupby(["N", "arm"]).agg(mean_qubo_vars=("num_qubo_vars", "mean"), mean_coarse_nodes=("n_coarse", "mean"))
              .round(1).unstack("arm"))
        SZ.columns = [f"{a} {b}" for a, b in SZ.columns]
        self.table("problem_sizes", SZ, index=True)
        cases = CO.pivot_table(index=["instance", "N"], columns="arm", values="n_coarse", aggfunc="first")
        diff_size = int((cases.gnn != cases.heuristic_fixed).sum())
        EARLY = CO.groupby("arm").agg(coarsenings=("instance", "size"),
                                      stopped_above_target=("ended_above_target", "sum")).reindex(ARMS)
        self.table("stopped_above_target", EARLY.reset_index())
        self.record("11.9", f"{diff_size}/{len(cases)} cases", "confirmed" if diff_size > 0 else "not confirmed")
        self.record("11.10", "; ".join(f"{a}: {int(r.stopped_above_target)}/{int(r.coarsenings)}" for a, r in EARLY.iterrows()),
                    "confirmed" if EARLY.stopped_above_target.sum() > 0 else "not confirmed")

        TESTS = self.table("wilcoxon_calls", pd.DataFrame([t for f in FACTS.values() for t in f["tests"]]))
        aggs = pd.DataFrame(F100["aggregations"])
        loops = [l.strip() for l in F100["src"].splitlines() if l.strip().startswith("for N in")]
        ok = (list(aggs.aggregation) == ["mean"] and len(F100["tests"]) == 1 and F100["tests"][0]["alternative"] == "greater"
              and loops and "[80, 100]" in loops[0])
        self.record("11.11", f"aggregation {list(aggs.aggregation)} at L{list(aggs.line)}; one test over the jobs of "
                             f"`{loops[0] if loops else '?'}`; alternative {F100['tests'][0]['alternative']}",
                    "confirmed" if ok else "not confirmed")
        other = TESTS[TESTS.script.isin(["scripts/run_scaling_large.py", "scripts/run_scorer_ablation.py"])]
        self.record("11.12", "; ".join(f"{r.script}:L{r.line} {r.alternative}" for r in other.itertuples()),
                    "confirmed" if len(other) and (other.alternative == "greater").all() else "not confirmed")

        n80, n100 = [i for i in TEST if NCUST[i] >= 80], [i for i in TEST if NCUST[i] >= 100]
        self.record("11.13", f"{len(n80)} + {len(n100)} = {len(n80) + len(n100)}",
                    "confirmed" if (len(n80), len(n100)) == (ref["pairs_n80"], ref["pairs_n100"]) else "not confirmed")

        def scaling_test(df):
            # scripts/run_scaling_100.py L53-L57: pairs are (instance, N) cells, seeds averaged
            d = df[df.N.isin([80, 100])]
            piv = d.groupby(["instance_cell", "N", "arm"]).is_feasible.mean().unstack("arm").dropna(subset=["gnn", "heuristic"])
            x, y = piv.gnn.to_numpy(), piv.heuristic.to_numpy()
            return {"pairs": len(x), "better or tied": int((x >= y).sum()), "strictly better": int((x > y).sum()),
                    "worse": int((x < y).sum()), "GNN %": round(100 * x.mean(), 1), "tuned %": round(100 * y.mean(), 1),
                    "p one-sided": wilcoxon_p(x, y, "greater"), "p two-sided": wilcoxon_p(x, y, "two-sided")}

        STT = [{"data": "main grid", **scaling_test(ST)}]
        if self.SC100 is not None:
            STT.append({"data": "unmodified run_scaling_100.py",
                        **scaling_test(self.SC100.rename(columns={"instance": "instance_cell", "condition": "arm"}))})
            self.scaling_printed = self.scripts["scripts/run_scaling_100.py"]["stdout"].strip().splitlines()[-1]
        STT = self.table("large_n_test", pd.DataFrame(STT).round(4))
        per = []
        for N in (80, 100):
            d = ST[ST.N == N]
            for b in ("heuristic_fixed", "heuristic"):
                pm = paired(d, "gnn", b, "is_feasible").groupby("instance")[["gnn", b]].median()
                per.append({"N": N, "comparison": f"gnn vs {b}", "base instances": len(pm),
                            "p two-sided (median over seeds)": wilcoxon_p(pm.gnn, pm[b], "two-sided")})
        self.table("large_n_per_size", pd.DataFrame(per).round(4))
        s = STT.iloc[0]
        self.record("11.14", f"{s['better or tied']}/{s.pairs}",
                    self.scored(s.pairs == ref["large_n_better_or_tied"] and s["better or tied"] == ref["large_n_better_or_tied"],
                                [80, 100]))
        s_unr = scaling_test(ST)       # unrounded p-values for the halving check
        if np.isnan(s_unr["p one-sided"]):
            self.record("11.15", "test undefined (no differences)", "insufficient data")
        else:
            half = math.isclose(s_unr["p one-sided"], s_unr["p two-sided"] / 2, rel_tol=1e-9) or \
                math.isclose(s_unr["p one-sided"], 1 - s_unr["p two-sided"] / 2, rel_tol=1e-9)
            self.record("11.15", f"one-sided {s_unr['p one-sided']:.4f}, two-sided {s_unr['p two-sided']:.4f}",
                        "confirmed" if half else "not confirmed")

        bi = ST.groupby(["instance", "arm"]).is_feasible.mean().unstack("arm")
        p1, p2 = wilcoxon_p(bi.gnn, bi.heuristic, "greater"), wilcoxon_p(bi.gnn, bi.heuristic, "two-sided")
        self.table("all_sizes_base_instances", pd.DataFrame([{"base instances": len(bi), "sizes": sorted(ST.N.unique()),
                                                               "p one-sided": round(p1, 4), "p two-sided": round(p2, 4)}]))
        pa = ref["p_all_sizes"]
        self.record("11.16", f"one-sided {p1:.3f}, two-sided {p2:.3f}", self.scored(abs(p1 - pa) < abs(p2 - pa), self.published))
        SIZES = self.table("sizes_per_script", pd.DataFrame(
            [{"script": k, "sizes": f["sizes"], "covers the seven published sizes": set(self.published) <= set(f["sizes"])}
             for k, f in FACTS.items() if f["sizes"]]))
        self.record("11.17", ", ".join(SIZES[SIZES["covers the seven published sizes"]].script) or "none",
                    "confirmed" if not SIZES["covers the seven published sizes"].any() else "not confirmed",
                    note="lists named Ns, loops over N, and --N/--Ns defaults; sizes given on a command line are not covered")

        rows = []
        for N, d in ST.groupby("N"):
            for b in ("heuristic_fixed", "heuristic"):
                pm = paired(d, "gnn", b, "is_feasible").groupby("instance")[["gnn", b]].median()
                rows.append({"N": N, "comparison": f"gnn vs {b}", "base instances": len(pm),
                             "GNN better": int((pm.gnn > pm[b]).sum()), "GNN worse": int((pm.gnn < pm[b]).sum()),
                             "p two-sided": round(wilcoxon_p(pm.gnn, pm[b], "two-sided"), 4)})
        PERSIZE = self.table("per_size_two_sided", pd.DataFrame(rows))

        PROPS = pd.DataFrame({"loss at expansion": self._lost, "hash salt": self._flips["salt"],
                              "edge order": self._flips["orient"]}).T.reindex(columns=ARMS).fillna(0).astype(int)
        self.table("properties_by_arm", PROPS, index=True)
        zero = [(p, a) for p in PROPS.index for a in ARMS if PROPS.loc[p, a] == 0]
        self.record("11.18", "all nine counts positive" if not zero else "zero for: " + "; ".join(f"{p} / {a}" for p, a in zero),
                    self.or_quick(not zero))
        g = CMP[CMP.comparison == "gnn vs heuristic_fixed"]
        h = CMP[CMP.comparison == "heuristic_fixed vs heuristic"]
        fx = PERSIZE[PERSIZE.comparison == "gnn vs heuristic_fixed"]
        return {"g_better": int(g.better.sum()), "g_worse": int(g.worse.sum()),
                "h_better": int(h.better.sum()), "h_worse": int(h.worse.sum()),
                "fx_sizes": len(fx), "fx_sig": int((fx["p two-sided"] < 0.05).sum()), "fx_min_p": fx["p two-sided"].min(),
                "n10_below": len(below), "n10_found": found, "p1_large": s_unr["p one-sided"], "p2_large": s_unr["p two-sided"],
                "diff_size": diff_size, "cases": len(cases), "published_grid": self.published_grid(self.published)}

    # ---- 12. stability of the training labels ----------------------------------
    def s12(self):
        LAB = self.LAB
        GL = self.files["2:scripts/gen_labels_rl.py"]
        tree = ast.parse(GL)
        main_fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
        in_loop = {id(x) for lp in [n for n in ast.walk(main_fn) if isinstance(n, ast.For)] for x in ast.walk(lp)}
        cfgs = [n for n in ast.walk(main_fn) if isinstance(n, ast.Call) and getattr(n.func, "id", "") == "SolverConfig"]
        cmp_ops = [type(o).__name__ for n in ast.walk(main_fn) if isinstance(n, ast.Compare)
                   for o in n.ops if "best" in (ast.get_source_segment(GL, n) or "")]
        facts = {"SolverConfig calls in main": len(cfgs), "inside a loop": sum(id(c) in in_loop for c in cfgs),
                 "line": [c.lineno for c in cfgs], "selection comparison": cmp_ops}
        self.table("label_script_structure", pd.DataFrame([{k: str(v) for k, v in facts.items()}]))
        self.record("12.1", f"{len(cfgs)} SolverConfig call(s) at L{facts['line']}, inside a loop: {facts['inside a loop']}; "
                            f"comparison {cmp_ops}",
                    "confirmed" if len(cfgs) == 1 and facts["inside a loop"] == 0 and "Gt" in cmp_ops else "not confirmed")
        self.table("label_runs", pd.DataFrame([{"cases (instance, N)": LAB.groupby(["instance", "N"]).ngroups,
                                                "candidates per case": LAB.cand.nunique(),
                                                "seeds": len(self.cfg["label_seeds"]), "evaluations": len(LAB)}]))
        rows = []
        for (inst, N), d in LAB.groupby(["instance", "N"]):
            r = d.pivot_table(index="cand", columns="seed", values="reward")
            tags = d.drop_duplicates("cand").set_index("cand").tag
            w0 = int(r[0].idxmax())
            mean = r.mean(axis=1)
            best = set(mean.index[np.isclose(mean, mean.max(), atol=1e-12)])
            others_s = [c for c in r.columns if c != 0]
            rows.append({"instance": inst, "N": N, "label at seed 0": w0, "its type": tags[w0],
                         "best on average": sorted(best), "type of the best": tags[min(best)],
                         "label is best on average": w0 in best,
                         "distinct winners over seeds": len({int(r[c].idxmax()) for c in r.columns}),
                         "reward at seed 0 minus mean elsewhere": r.loc[w0, 0] - r.loc[w0, others_s].mean()})
        LW = self.table("label_selection", pd.DataFrame(rows))
        n_c, k_c = len(LW), int((~LW["label is best on average"]).sum())
        lo, hi = wilson(k_c, n_c)
        self.record("12.2", f"{k_c}/{n_c} ({pct(k_c, n_c)}, Wilson CI95 {lo:.0f}-{hi:.0f}%)", self.or_quick(k_c > 0))
        ind = 2.0 * LAB.is_feasible + 1.0 * LAB.pre_feasible
        share = float(np.corrcoef(ind, LAB.reward)[0, 1] ** 2) if np.var(LAB.reward) > 0 else float("nan")
        self.record("12.3", f"{share:.2f}", "confirmed" if share > 0.5 else "not confirmed")
        curse = LW["reward at seed 0 minus mean elsewhere"].to_numpy()
        s = self.cfg["statistics"]
        rng = np.random.default_rng(s["boot_seed"])
        bs = [rng.choice(curse, len(curse)).mean() for _ in range(s["boot_reps"])]
        clo, chi = np.percentile(bs, [2.5, 97.5])
        self.record("12.4", f"mean {curse.mean():.3f} (bootstrap CI95 {clo:.3f} to {chi:.3f})", self.or_quick(clo > 0))
        return {"k": k_c, "n": n_c, "varies": int((LW["distinct winners over seeds"] > 1).sum()),
                "seeds": len(self.cfg["label_seeds"]), "curse": float(curse.mean()), "clo": clo, "chi": chi, "share": share,
                "types_seed0": dict(Counter(LW["its type"])), "types_best": dict(Counter(LW["type of the best"]))}

    # ---- review map and output ---------------------------------------------------
    def review_map(self):
        rows = []
        for r in self.remarks:
            cs = sorted((c for c in self.checks if r["number"] in c["remarks"]),
                        key=lambda c: tuple(map(int, c["check"].split("."))))
            rows.append({"point": r["number"], "type": r["kind"], "title": r["title"],
                         "sections": ", ".join(sorted({str(c["section"]) for c in cs}, key=int)),
                         "checks": ", ".join(c["check"] for c in cs),
                         **{s: sum(c["status"] == s for c in cs) for s in STATUSES}})
        self.MAP = self.table("review_point_map", pd.DataFrame(rows))
        unmapped = [r["point"] for r in rows if not r["checks"]]
        if unmapped:
            raise AssertionError(f"review points without a check: {unmapped}")

    def summary_table(self):
        S = pd.DataFrame(self.checks)
        S["order"] = S.check.map(lambda c: tuple(int(x) for x in c.split(".")))
        S = S.sort_values("order").drop(columns="order").reset_index(drop=True)
        S["kind of finding"] = S.kind.map(KINDS)
        return S

    def write(self):
        S = self.summary_table()
        S.drop(columns=["kind"]).to_csv(self.out / "summary_checks.csv", index=False)
        with open(self.out / "summary_checks.md", "w") as fh:
            fh.write(f"# Checks ({self.R['mode']} mode)\n\n| check | kind | test | computed | stated | status |\n"
                     "|---|---|---|---|---|---|\n")
            for r in S.itertuples():
                fh.write("| " + " | ".join(str(x).replace("|", "\\|") for x in
                                            (r.check, r.kind, r.test, r.computed, r.stated, r.status)) + " |\n")
        counts = dict(Counter(c["status"] for c in self.checks))
        write_json_atomic(self.out / "verification.json", {
            "schema_version": SCHEMA_VERSION, "results_status": self.R["status"], "mode": self.R["mode"],
            "config_sha256": self.R["config_sha256"], "results_created_utc": self.R["created_utc"],
            "status_counts": counts, "sections_without_data": {str(k): v for k, v in self.missing.items()},
            "integrity": {"probe_reproduces_script": getattr(self, "equiv_ok", None),
                          "probe_vs_script": getattr(self, "equiv", None),
                          "every_customer_once": getattr(self, "coverage_ok", None),
                          "n10_main_grid_equals_script": getattr(self, "n10_same_as_main", None),
                          "sources": getattr(self, "source_summary", None)},
            "checks": self.checks})
        self.status_counts = counts


def verify(results_path, audit_dir, expected_config=None, allow_incomplete=False):
    """Load results.json, check that it belongs to `expected_config`, and verify it."""
    R = load_results(results_path, audit_dir, expected_config, allow_incomplete)
    v = Verification(R, audit_dir, Path(results_path).parent).run()
    n = len(v.checks)
    print(f"{Path(results_path).name}: {R['status']}, {R['mode']} mode, created {R['created_utc']}, "
          f"config {R['config_sha256'][:12]}…")
    if hasattr(v, "source_summary"):
        ss = v.source_summary
        print(f"sources: {ss['verified']} of {ss['distinct']} distinct passages and lines verified "
              f"({ss['by_source']['[1]']} from [1], {ss['by_source']['[2]']} from [2], {ss['by_source']['[3]']} from [3])")
    if hasattr(v, "equiv_ok"):
        print(f"probe reproduces the unmodified script: {v.equiv_ok}; every customer exactly once: {v.coverage_ok}")
    print(f"{n} checks: " + ", ".join(f"{v.status_counts.get(s, 0)} {s}" for s in STATUSES))
    if v.missing:
        print("sections without their data (insufficient data):", v.missing)
    return v
