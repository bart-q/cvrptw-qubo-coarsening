"""Generation stage of the GNN results audit.

Prepares the pinned sources and the pipeline environment, runs the authors' scripts and
the measurement probe (audit_probe.py), and writes every observation to one versioned
results file, ``output/results.json``. Nothing here decides whether a statement holds;
that is audit_verify.py, which reads only the results file.

Layout (Colab paths; locally, run from review/: inputs in ./content, the rest in ./):
  /content               the inputs of review/content (this module, the probe, JSON)
  /content/audit_work    checkouts of the two repositories, the manuscript PDF, the virtualenv
  /content/output        results.json, per-stage run tables, logs and reports

Results file contract (schema_version 1):
  schema_version, status ("incomplete" until every stage is done, then "complete"),
  created_utc / finished_utc, mode, config, config_sha256, inputs (sha256 of every input
  file), provenance (commits, PDF and checkpoint digests, interpreters, package versions),
  references (values from the manuscript or the review, never results), stages (status,
  seconds, error and metadata per stage), commands, records (observations per stage),
  sources (text of the pinned files and PDF pages the verification reads), artifacts.
"""
from __future__ import annotations

import ast
import contextlib
import datetime as _dt
import hashlib
import json
import math
import os
import platform
import shlex
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import pandas as pd

SCHEMA_VERSION = 1
HERE = Path(__file__).resolve().parent
INPUT_FILES = ["audit_runner.py", "audit_probe.py", "audit_verify.py", "audit_report.py",
               "audit_config.json", "audit_remarks.json", "audit_fragments.json",
               "requirements-pipeline.txt"]
# inputs whose content determines the observations; verify-only refuses results made with others
GENERATION_INPUTS = ["audit_runner.py", "audit_probe.py", "audit_config.json",
                     "requirements-pipeline.txt"]
STAGES = ["prepare", "sources", "data_and_scripts", "main_grid", "edge_order", "hash_salts",
          "label_stability"]
PIPELINE_ENV = {"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
                "CUDA_VISIBLE_DEVICES": "", "PYTHONUNBUFFERED": "1"}


# --------------------------------------------------------------------------- #
# configuration and the results contract
# --------------------------------------------------------------------------- #
def missing_inputs(audit_dir):
    return [f for f in INPUT_FILES if not (Path(audit_dir) / f).is_file()]


def load_config(audit_dir, run_full, workers=None):
    """Full configuration of one run: fixed settings plus the QUICK or FULL preset."""
    raw = json.loads((Path(audit_dir) / "audit_config.json").read_text())
    mode = "FULL" if run_full else "QUICK"
    preset = raw["modes"][mode]
    exp = raw["experiment"]
    return {
        "mode": mode,
        "sources": raw["sources"],
        **exp,
        "grid": {int(N): list(s) for N, s in preset["grid"].items()},
        "alt_salts": preset["alt_salts"],
        "label_ns": preset["label_ns"],
        "label_instances": preset["label_instances"],
        "label_seeds": preset["label_seeds"],
        "authors_scripts": preset["authors_scripts"],
        "statistics": raw["statistics"],
        "cmd_timeout_s": raw["resources"]["cmd_timeout_s"],
        "workers": int(workers or max(1, os.cpu_count() or 1)),
        "reference_values": {k: v["value"] for k, v in raw["reference_values"]["values"].items()},
        "reference_sources": {k: v["source"] for k, v in raw["reference_values"]["values"].items()},
    }


def config_digest(cfg):
    """Digest of everything that determines the observations (worker count excluded)."""
    c = {k: v for k, v in cfg.items() if k != "workers"}
    c["grid"] = {str(k): v for k, v in c["grid"].items()}
    return hashlib.sha256(json.dumps(c, sort_keys=True).encode()).hexdigest()


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _clean(x):
    """JSON-safe copy: NaN/inf become null, numpy scalars become Python numbers."""
    if isinstance(x, dict):
        return {str(k): _clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_clean(v) for v in x]
    if hasattr(x, "item") and not isinstance(x, (str, bytes)):
        x = x.item()
    if isinstance(x, float) and not math.isfinite(x):
        return None
    return x


def write_json_atomic(path, obj):
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(_clean(obj), indent=1, ensure_ascii=False))
    os.replace(tmp, path)


def _now():
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")


class StageError(RuntimeError):
    pass


def summarize_jobs(command, jobs, up=None):
    """One line describing the jobs handed to a probe process (read from the jobs themselves)."""
    if command in ("hashprobe", "data"):
        return f"audit_probe.py {command}: no jobs"
    if command == "labels":
        j = jobs
        model = Path(j["model_path"]).relative_to(up) if up else j["model_path"]
        return (f"audit_probe.py labels: {len(j['instances'])} training instances × N {j['Ns']}, "
                f"pool of 5 heuristic + {j['n_gnn'] + 1} GNN candidates, SA seeds {j['seeds']}, "
                f"{j['num_reads']} reads × {j['num_sweeps']} sweeps, OR-Tools {j['ortools_time']} s, model {model}")
    by_n = {}
    for x in jobs:
        by_n.setdefault(x["N"], set()).add(x.get("seed"))
    arms = sorted({x["arm"] for x in jobs})
    variant = sorted({(x.get("orient", "upstream"), x.get("tie", "upstream")) for x in jobs})
    first = jobs[0] if jobs else {}
    model = first.get("model_path", "")
    if up and model:
        model = Path(model).relative_to(up)
    parts = [f"audit_probe.py {command}: {len(jobs)} jobs", "arms " + ", ".join(arms),
             "; ".join(f"N={N} seeds {','.join(map(str, sorted(s)))}" for N, s in by_n.items())]
    if command == "staged":
        parts.append(f"SA {first.get('num_reads')} reads × {first.get('num_sweeps')} sweeps, "
                     f"penalty {first.get('penalty_mode')}")
    if "gnn" in arms:
        parts.append(f"model {model}")
    parts.append("variant " + ", ".join(f"orient={o} tie={t}" for o, t in variant))
    return "; ".join(parts)


# --------------------------------------------------------------------------- #
# the run
# --------------------------------------------------------------------------- #
class AuditRun:
    """One generation run. Each public method is one stage; after every stage the results
    file is rewritten (atomically) with status "incomplete" until all stages are done."""

    def __init__(self, cfg, base, audit_dir=HERE, verify_only=False):
        self.cfg = cfg
        self.verify_only = verify_only
        self.base = Path(base)
        self.audit_dir = Path(audit_dir).resolve()
        self.out = self.base / "output"
        self.work = self.base / "audit_work"
        self.results_path = self.out / "results.json"
        self.probe = self.audit_dir / "audit_probe.py"
        self.up = self.work / "cvrptw-qubo-coarsening"
        self.pred = self.work / "graph-coarsening"
        self.pdf = self.work / "arXiv-2609.04593v2.pdf"
        self.venv_py = self.work / "venv" / "bin" / "python"
        self.model = self.up / cfg["sources"]["checkpoint"]
        if verify_only:
            print(f"verify-only: no generation; the stages below are skipped and "
                  f"{self.results_path} is read by the verification")
            return
        for d in (self.out, self.out / "runs", self.out / "scripts", self.work):
            d.mkdir(parents=True, exist_ok=True)
        self.R = {
            "schema_version": SCHEMA_VERSION, "status": "incomplete", "created_utc": _now(),
            "finished_utc": None, "mode": cfg["mode"], "config": cfg,
            "config_sha256": config_digest(cfg),
            "inputs": {f: sha256_file(self.audit_dir / f) for f in INPUT_FILES},
            "provenance": {}, "references": {"values": cfg["reference_values"],
                                             "sources": cfg["reference_sources"]},
            "stages": {s: {"status": "pending"} for s in STAGES},
            "commands": [], "records": {}, "sources": {}, "artifacts": {},
        }
        self.save()       # a stale complete file never survives the start of a new run

    # ---- infrastructure ---------------------------------------------------
    def save(self):
        write_json_atomic(self.results_path, self.R)

    def command_spec(self, argv, cwd=None, env_extra=None, unset=()):
        """argv, working directory and environment of one process, exactly as run_cmd passes
        them to subprocess.run; `set`/`unset` are the variables that differ from the kernel's."""
        env = {**os.environ, **PIPELINE_ENV, **(env_extra or {})}
        for k in unset:
            env.pop(k, None)
        return {"argv": [str(a) for a in argv], "cwd": str(cwd) if cwd else None, "env": env,
                "set": {**PIPELINE_ENV, **(env_extra or {})}, "unset": list(unset)}

    @staticmethod
    def describe(spec):
        """Shell form of a command_spec: cd <cwd> && env [-u VAR] VAR=value ... argv."""
        env = ["env"] + [f"-u {k}" for k in spec["unset"]] + \
              [shlex.quote(f"{k}={v}") for k, v in spec["set"].items()]
        cd = f"cd {shlex.quote(spec['cwd'])} && " if spec["cwd"] else ""
        return cd + " ".join(env) + " " + shlex.join(spec["argv"])

    def run_cmd(self, argv, cwd=None, env_extra=None, unset=(), check=False, quiet=False, show=False):
        """Run one process. With show=True the command is printed as 'planned' before it starts and
        as 'executed' with its exit code after it ends, both from the spec handed to subprocess.run."""
        spec = self.command_spec(argv, cwd, env_extra, unset)
        entry = {"command": shlex.join(spec["argv"]), "cwd": spec["cwd"] or "", "status": "started",
                 "env_set": {k: v for k, v in spec["set"].items() if k not in PIPELINE_ENV},
                 "env_unset": spec["unset"], "exit": None, "seconds": None,
                 "PYTHONHASHSEED": spec["env"].get("PYTHONHASHSEED", "(unset)")}
        self.R["commands"].append(entry)
        if show:
            print("planned : " + self.describe(spec))
        t0 = time.perf_counter()
        p = subprocess.run(spec["argv"], cwd=spec["cwd"], env=spec["env"], capture_output=True,
                           text=True, timeout=self.cfg["cmd_timeout_s"])
        dt = time.perf_counter() - t0
        entry.update(status="executed", exit=p.returncode, seconds=round(dt, 1))
        if show:
            print(f"executed: exit {p.returncode} in {dt:.1f} s")
        elif not quiet:
            print(f"$ {' '.join(spec['argv'])[:160]}\n  -> exit {p.returncode} in {dt:.1f} s")
        if check and p.returncode != 0:
            print(p.stdout[-3000:])
            print(p.stderr[-3000:])
            raise StageError(f"command failed with exit {p.returncode}: {' '.join(spec['argv'])[:200]}")
        return p

    @contextlib.contextmanager
    def stage(self, name, needs=()):
        st = self.R["stages"][name]
        for n in needs:
            if self.R["stages"][n]["status"] != "done":
                raise StageError(f"stage '{name}' needs stage '{n}', which has status "
                                 f"'{self.R['stages'][n]['status']}'")
        st.update(status="running", started_utc=_now(), meta={})
        t0 = time.perf_counter()
        try:
            yield st["meta"]
        except BaseException as e:
            st.update(status="failed", seconds=round(time.perf_counter() - t0, 1),
                      error=f"{type(e).__name__}: {e}")
            self.save()
            print(f"stage '{name}' FAILED: {type(e).__name__}: {e}\n"
                  f"results.json stays 'incomplete'; stages that need '{name}' will refuse to run")
            raise
        st.update(status="done", seconds=round(time.perf_counter() - t0, 1), error=None)
        self.save()
        print(f"stage '{name}' done in {st['seconds']:.0f} s")

    def _skip(self, name):
        if self.verify_only:
            print(f"verify-only: stage '{name}' not run; its observations are read from {self.results_path}")
        return self.verify_only

    def run_probe(self, command, jobs=None, salt=None, name=None):
        """One probe process (in the pipeline environment, from the authors' checkout) with
        PYTHONHASHSEED set before it starts; salt=None uses the main salt. Prints the planned
        command and the jobs it is given, then the executed result."""
        salt = self.cfg["main_salt"] if salt is None else salt
        name = name or command
        jf, of = self.work / f"jobs_{name}.json", self.work / f"out_{name}.jsonl"
        jf.write_text(json.dumps(jobs if jobs is not None else []))
        env = {} if salt == "unset" else {"PYTHONHASHSEED": str(salt)}
        unset = ("PYTHONHASHSEED",) if salt == "unset" else ()
        print(f"[{name}] " + summarize_jobs(command, jobs, self.up))
        self.run_cmd([self.venv_py, self.probe, command, "--jobs", jf, "--out", of,
                      "--workers", self.cfg["workers"]],
                     cwd=self.up, env_extra=env, unset=unset, check=True, show=True)
        if command in ("hashprobe", "data"):
            return json.loads(of.read_text())
        rows = [json.loads(x) for x in of.read_text().splitlines() if x]
        if command != "labels" and len(rows) != len(jobs):
            raise StageError(f"probe '{name}': {len(jobs)} jobs but {len(rows)} records")
        print(f"          {len(rows)} records")
        pd.DataFrame(rows).to_csv(self.out / "runs" / f"{name}.csv.gz", index=False)
        return rows

    def show_source(self, *qualnames):
        """Show the source of functions as the pipeline environment sees them: inspect.getsource runs
        in the pipeline virtualenv, from the checkout, on the modules the stages execute (the probe
        in use and the released code at the pinned commit). Collapsed, with each file's digest.
        A preview only: not logged, never fatal."""
        from html import escape
        from IPython.display import HTML, display
        if not self.venv_py.exists() or not (self.up / ".git").exists():
            print("source preview needs the pipeline environment (run.prepare()); not available here")
            return
        code = ("import sys, json, inspect, importlib, hashlib; sys.path[:0] = [sys.argv[1], '.']\n"
                "out = []\n"
                "for q in sys.argv[2:]:\n"
                "    mod, _, attr = q.rpartition('.')\n"
                "    m = importlib.import_module(mod)\n"
                "    f = inspect.getsourcefile(m)\n"
                "    out.append({'name': q, 'file': f, 'sha256': hashlib.sha256(open(f, 'rb').read()).hexdigest(),\n"
                "                'line': inspect.getsourcelines(getattr(m, attr))[1],\n"
                "                'source': inspect.getsource(getattr(m, attr))})\n"
                "print(json.dumps(out))")
        spec = self.command_spec([self.venv_py, "-c", code, self.probe.parent, *qualnames], cwd=self.up)
        p = subprocess.run(spec["argv"], cwd=spec["cwd"], env=spec["env"], capture_output=True, text=True)
        if p.returncode != 0:
            print("source preview failed:", p.stderr.strip().splitlines()[-1:] or p.returncode)
            return
        for item in json.loads(p.stdout.strip().splitlines()[-1]):
            where = Path(item["file"])
            where = where.relative_to(self.up) if where.is_relative_to(self.up) else where.name
            display(HTML(f"<details><summary>source of <code>{escape(item['name'])}</code> - {escape(str(where))}, "
                         f"line {item['line']}, sha256 {item['sha256'][:12]}…</summary>"
                         f"<pre>{escape(item['source'])}</pre></details>"))

    def job(self, instance, N, seed, arm, **extra):
        c = self.cfg
        return {"instance": instance, "N": N, "seed": seed, "arm": arm, "num_reads": c["sa_reads"],
                "num_sweeps": c["sa_sweeps"], "penalty_mode": c["penalty_mode"], "solver": "FQS",
                "model_path": str(self.model), **extra}

    def grid_jobs(self, **extra):
        D = self.R["records"]["data_facts"]
        out = []
        for N, seeds in self.cfg["grid"].items():
            for inst in D["test"]:
                if D["n_customers"][inst] < N:          # rule of scripts/run_scaling_100.py L32
                    continue
                for seed in seeds:
                    for arm in self.cfg["arms"]:
                        out.append(self.job(inst, N, seed, arm, **extra))
        return out

    # ---- stages -----------------------------------------------------------
    def prepare(self):
        """Clone the pinned repositories, download the manuscript, build the pipeline
        environment, and run one smoke job per arm."""
        if self._skip("prepare"):
            return
        s = self.cfg["sources"]
        with self.stage("prepare") as meta:
            if not (self.up / ".git").exists():
                self.run_cmd(["git", "clone", "--quiet", s["upstream_url"], self.up], check=True)
            self.run_cmd(["git", "-C", self.up, "checkout", "--quiet", "--detach", s["upstream_ref"]],
                         check=True)
            head = self.run_cmd(["git", "-C", self.up, "rev-parse", "HEAD"], quiet=True).stdout.strip()
            if head != s["upstream_ref"]:
                raise StageError(f"upstream HEAD {head} is not the pinned commit")
            if not self.model.is_file():
                raise StageError(f"released checkpoint missing: {self.model}")
            if not (self.pred / ".git").exists():
                self.run_cmd(["git", "clone", "--quiet", "--filter=blob:none", "--no-checkout",
                              s["predecessor_url"], self.pred], check=True)
            self.run_cmd(["git", "-C", self.pred, "checkout", "--quiet", "--detach",
                          s["predecessor_ref"]], check=True)
            if not self.pdf.exists():
                req = urllib.request.Request(s["manuscript_pdf_url"], headers={"User-Agent": "Mozilla/5.0"})
                self.pdf.write_bytes(urllib.request.urlopen(req, timeout=120).read())

            base_py = self._pipeline_python()
            if not self.venv_py.exists():
                self.run_cmd([base_py, "-m", "venv", self.venv_py.parent.parent], check=True)
            t0 = time.perf_counter()
            self.run_cmd([self.venv_py, "-m", "pip", "install", "--quiet", "--upgrade", "pip"], check=True)
            self.run_cmd([self.venv_py, "-m", "pip", "install", "--quiet",
                          "--index-url", "https://download.pytorch.org/whl/cpu",
                          "--extra-index-url", "https://pypi.org/simple", "torch==2.2.2+cpu"], check=True)
            self.run_cmd([self.venv_py, "-m", "pip", "install", "--quiet", "-r",
                          self.audit_dir / "requirements-pipeline.txt"], check=True)
            freeze = self.run_cmd([self.venv_py, "-m", "pip", "freeze"], quiet=True).stdout
            (self.out / "pip_freeze.txt").write_text(freeze)
            print(f"pipeline packages installed in {time.perf_counter() - t0:.0f} s")
            probe = self.run_cmd(
                [self.venv_py, "-c",
                 "import json, torch, numpy, dimod, torch_geometric, ortools;"
                 "from dwave.samplers import SimulatedAnnealingSampler;"
                 "import src.coarsening, src.gnn_coarsening, src.experiment, src.runner_parallel;"
                 "print(json.dumps({'torch': torch.__version__, 'cuda': torch.cuda.is_available(),"
                 "'numpy': numpy.__version__, 'dimod': dimod.__version__,"
                 "'torch_geometric': torch_geometric.__version__}))"], cwd=self.up, check=True)
            versions = json.loads(probe.stdout.strip().splitlines()[-1])
            if versions["cuda"]:
                raise StageError("the pipeline must run on the CPU")
            self.R["provenance"] = {
                "upstream": {"url": s["upstream_url"], "commit": head},
                "predecessor": {"url": s["predecessor_url"], "commit": s["predecessor_ref"]},
                "manuscript": {"url": s["manuscript_pdf_url"], "sha256": sha256_file(self.pdf)},
                "checkpoint": {"path": s["checkpoint"], "sha256": sha256_file(self.model),
                               "bytes": self.model.stat().st_size},
                "kernel_python": sys.version.split()[0], "platform": platform.platform(),
                "pipeline_python": self.run_cmd([self.venv_py, "-V"], quiet=True).stdout.strip(),
                "pipeline_versions": versions, "pip_freeze": freeze.splitlines(),
                "workers": self.cfg["workers"]}
            smoke = self.run_probe("staged", [self.job("R104", 10, 0, a) for a in self.cfg["arms"]],
                                   name="smoke")
            self.R["records"]["smoke"] = smoke
            meta["smoke_feasible"] = {r["arm"]: r["is_feasible"] for r in smoke}
        p = self.R["provenance"]
        print(f"upstream {p['upstream']['commit'][:7]}, predecessor {p['predecessor']['commit'][:7]}, "
              f"PDF sha256 {p['manuscript']['sha256'][:12]}…, checkpoint sha256 {p['checkpoint']['sha256'][:12]}…")
        print(f"{p['pipeline_python']}; " + ", ".join(f"{k} {v}" for k, v in p["pipeline_versions"].items()))
        print("smoke test (R104, N=10, seed 0) feasible:", meta["smoke_feasible"])

    def _pipeline_python(self):
        """The pinned wheels exist for Python up to 3.12; on a newer kernel use uv's 3.12."""
        if sys.version_info[:2] <= (3, 12):
            return sys.executable
        if shutil.which("uv") is None:
            self.run_cmd([sys.executable, "-m", "pip", "install", "--quiet", "uv"], check=True)
        uv = shutil.which("uv") or str(Path(sys.executable).with_name("uv"))
        self.run_cmd([uv, "python", "install", "3.12"], check=True)
        return self.run_cmd([uv, "python", "find", "3.12"], quiet=True, check=True).stdout.strip()

    def sources(self):
        """Snapshot of what the verification reads from the sources: the PDF text page by page,
        every cited file of [2] and [3] at the pinned commits, the release's scripts, the
        commits that added the predecessor's files, a search for PYTHONHASHSEED, and the
        loader's own self-check."""
        if self._skip("sources"):
            return
        from pypdf import PdfReader
        s = self.cfg["sources"]
        with self.stage("sources", needs=["prepare"]) as meta:
            rem = json.loads((self.audit_dir / "audit_remarks.json").read_text())
            frag = json.loads((self.audit_dir / "audit_fragments.json").read_text())["fragments"]
            wanted = {(q["src"], q["path"]) for r in rem["remarks"] for q in r["quotes"] if q["kind"] == "code"}
            wanted |= {(3, p["path"]) for p in rem["predecessor_pointers"]}
            wanted |= {(3, f["path"]) for f in frag}
            wanted |= {(2, "src/datasets.py"), (2, "data/solomon/C1/C104.csv"),
                       (3, "solomon_dataset/C1/C104.csv")}
            ls = self.run_cmd(["git", "-C", self.up, "ls-tree", "--name-only", s["upstream_ref"],
                               "scripts/"], quiet=True, check=True).stdout.split()
            wanted |= {(2, p) for p in ls if p.endswith(".py")}
            files = {}
            for src, path in sorted(wanted):
                repo, ref = (self.up, s["upstream_ref"]) if src == 2 else (self.pred, s["predecessor_ref"])
                files[f"{src}:{path}"] = self.run_cmd(["git", "-C", repo, "show", f"{ref}:{path}"],
                                                      quiet=True, check=True).stdout
            pages = [p.extract_text() or "" for p in PdfReader(str(self.pdf)).pages]
            pred_first = {}
            for f in ("graph_coarsening.py", "solomon_dataset/C1/C104.csv"):
                added = self.run_cmd(["git", "-C", self.pred, "log", "--diff-filter=A", "--format=%H",
                                      "--", f], quiet=True).stdout.split()
                pred_first[f] = added[-1] if added else None
            g3 = self.run_cmd(["git", "-C", self.pred, "grep", "-n", "PYTHONHASHSEED",
                               s["predecessor_ref"]], quiet=True)
            g2 = self.run_cmd(["git", "-C", self.up, "grep", "-n", "PYTHONHASHSEED",
                               s["upstream_ref"], "--", "src"], quiet=True)
            sc = self.run_cmd([self.venv_py, "-m", "src.datasets"], cwd=self.up)
            self.R["sources"] = {
                "files": files, "pdf_pages": pages, "predecessor_first_commit": pred_first,
                "grep_pythonhashseed": {"[2] src/": g2.stdout.strip(), "[3]": g3.stdout.strip()},
                "loader_selfcheck": {"exit": sc.returncode,
                                     "last_line": (sc.stdout.strip().splitlines() or [sc.stderr[-500:]])[-1]}}
            self.R["provenance"]["manuscript"]["pages"] = len(pages)
            meta.update(files=len(files), pdf_pages=len(pages))
        print(f"{meta['files']} files of [2] and [3] and {meta['pdf_pages']} PDF pages recorded")

    def data_and_scripts(self):
        """Data facts from the authors' loader and split, and the authors' unmodified scripts."""
        if self._skip("data_and_scripts"):
            return
        with self.stage("data_and_scripts", needs=["prepare"]) as meta:
            D = self.run_probe("data")
            self.R["records"]["data_facts"] = D
            for f in ("scorer_ablation.csv", "ortools_cache.csv", "scaling_100_solve.csv"):
                (self.up / "results" / f).unlink(missing_ok=True)
            out = {}
            for s in self.cfg["authors_scripts"]:
                print(f"[{s}] the authors' script, unmodified")
                p = self.run_cmd([self.venv_py, s], cwd=self.up,
                                 env_extra={"PYTHONHASHSEED": str(self.cfg["main_salt"])}, check=True, show=True)
                (self.out / "scripts" / f"{Path(s).stem}.stdout.txt").write_text(p.stdout)
                print(p.stdout)
                csv = {"scripts/run_scorer_ablation.py": "scorer_ablation.csv",
                       "scripts/run_scaling_100.py": "scaling_100_solve.csv"}[s]
                df = pd.read_csv(self.up / "results" / csv)
                df.to_csv(self.out / "runs" / csv, index=False)
                out[s] = {"stdout": p.stdout, "rows": df.to_dict("records")}
            changed = self.run_cmd(["git", "-C", self.up, "status", "--porcelain", "--untracked-files=no"],
                                   quiet=True).stdout.strip()
            if changed:
                raise StageError(f"tracked files of the checkout changed: {changed}")
            self.R["records"]["scripts"] = out
            meta.update(n_train=len(D["train"]), n_test=len(D["test"]),
                        script_rows={k: len(v["rows"]) for k, v in out.items()}, checkout_unchanged=True)
        print(f"{len(D['names'])} instances: {meta['n_train']} training, {meta['n_test']} test; "
              f"script rows {meta['script_rows']}; tracked files of the checkout unchanged")

    def main_grid(self):
        """The three arms on every test instance and size of the grid, recorded by stage."""
        if self._skip("main_grid"):
            return
        with self.stage("main_grid", needs=["data_and_scripts"]) as meta:
            jobs = self.grid_jobs()
            self.R["records"]["main"] = self.run_probe("staged", jobs, name="main")
            meta["jobs"] = len(jobs)
        print(f"{meta['jobs']} runs")

    def edge_order(self):
        """The main grid with every edge created in the opposite order, and the heuristic part
        of the GNN's label pool coarsened in both orders."""
        if self._skip("edge_order"):
            return
        with self.stage("edge_order", needs=["data_and_scripts", "sources"]) as meta:
            jobs = self.grid_jobs(orient="flipped")
            self.R["records"]["flipped"] = self.run_probe("staged", jobs, name="flipped")
            src = self.R["sources"]["files"]["2:scripts/gen_labels_rl.py"]
            sweep = next(ast.literal_eval(n.value) for n in ast.walk(ast.parse(src))
                         if isinstance(n, ast.Assign) and any(getattr(t, "id", "") == "SWEEP" for t in n.targets))
            pool = [self.job(i, N, 0, "params", params=p, tag=k, orient=o)
                    for i in self.R["records"]["data_facts"]["train"] for N in self.cfg["label_pool_ns"]
                    for k, p in enumerate(sweep) for o in ("upstream", "flipped")]
            self.R["records"]["label_pool"] = self.run_probe("signatures", pool, name="label_pool_orientation")
            meta.update(jobs=len(jobs), label_pool_jobs=len(pool), label_pool_settings=sweep)
        print(f"{meta['jobs']} runs in the opposite order; {meta['label_pool_jobs']} label-pool coarsenings")

    def hash_salts(self):
        """String hashing in fresh processes; the coarsening of every main-grid case under other
        salts; the changed cases solved again; a control sample; a deterministic tie-break."""
        if self._skip("hash_salts"):
            return
        c = self.cfg
        salts = [c["main_salt"]] + c["alt_salts"]
        with self.stage("hash_salts", needs=["main_grid"]) as meta:
            hp = []
            for label, salt in [("0", 0), ("0", 0), ("1", 1), ("unset", "unset"), ("unset", "unset")]:
                hp.append({"label": label, **self.run_probe("hashprobe", salt=salt, name="hashprobe")})
            self.R["records"]["hashprobe"] = hp
            keys = list(dict.fromkeys((r["instance"], r["N"], r["arm"]) for r in self.R["records"]["main"]))
            sig_jobs = [self.job(i, N, 0, a) for i, N, a in keys]
            sig = []
            for s in salts:
                sig += self.run_probe("signatures", sig_jobs, salt=s, name=f"signatures_salt{s}")
            base = {(r["instance"], r["N"], r["arm"]): r["sig_layers"] for r in sig if int(r["salt"]) == c["main_salt"]}
            changed = [r for r in sig if r["sig_layers"] != base[(r["instance"], r["N"], r["arm"])]]
            resolved = []
            for s in c["alt_salts"]:
                js = [self.job(r["instance"], r["N"], seed, r["arm"]) for r in changed
                      if int(r["salt"]) == s for seed in c["grid"][r["N"]]]
                if js:
                    resolved += self.run_probe("staged", js, salt=s, name=f"salt{s}")
            # control: up to four unchanged cases per arm (N <= 40) solved under the first alternative salt
            same = sorted((r for r in sig if int(r["salt"]) == c["alt_salts"][0] and r["N"] <= 40
                           and r["sig_layers"] == base[(r["instance"], r["N"], r["arm"])]),
                          key=lambda r: (r["arm"], r["instance"], r["N"]))
            ctrl_keys, per_arm = [], {}
            for r in same:
                if per_arm.get(r["arm"], 0) < 4:
                    per_arm[r["arm"]] = per_arm.get(r["arm"], 0) + 1
                    ctrl_keys.append(r)
            ctrl = self.run_probe("staged", [self.job(r["instance"], r["N"], 0, r["arm"]) for r in ctrl_keys],
                                  salt=c["alt_salts"][0], name="salt_control")
            can = []
            for s in salts:
                can += self.run_probe("signatures", [dict(j, tie="canonical") for j in sig_jobs], salt=s,
                                      name=f"canonical_salt{s}")
            self.R["records"].update(signatures=sig, salt_runs=resolved, salt_control=ctrl, canonical=can)
            meta.update(salts=salts, cases=len(sig_jobs), changed_coarsenings=len(changed),
                        resolved_runs=len(resolved), control_runs=len(ctrl))
        print(f"{len(sig)} coarsenings under salts {salts}; {meta['changed_coarsenings']} differ from salt "
              f"{c['main_salt']}; {meta['resolved_runs']} runs solved again; {meta['control_runs']} control runs")

    def label_stability(self):
        """The label pool of scripts/gen_labels_rl.py rebuilt for a set of training instances,
        every candidate solved with several SA seeds (seed 0 is the script's own evaluation)."""
        if self._skip("label_stability"):
            return
        c = self.cfg
        with self.stage("label_stability", needs=["data_and_scripts"]) as meta:
            train = self.R["records"]["data_facts"]["train"]
            spec = {"Ns": c["label_ns"], "instances": train if c["label_instances"] == "all" else c["label_instances"],
                    "n_gnn": c["label_n_gnn"], "seeds": c["label_seeds"], "num_reads": c["sa_reads"],
                    "num_sweeps": c["sa_sweeps"], "ortools_time": c["label_ortools_seconds"],
                    "model_path": str(self.model), "rng_seed": c["label_rng_seed"]}
            rows = self.run_probe("labels", spec, name="label_noise")
            n_cases = len({(r["instance"], r["N"]) for r in rows})
            expected = len(spec["instances"]) * len(c["label_ns"])
            if n_cases != expected:
                raise StageError(f"label stability: {expected} cases expected, {n_cases} recorded")
            self.R["records"]["labels"] = rows
            meta.update(cases=n_cases, evaluations=len(rows), seeds=c["label_seeds"])
        print(f"{meta['evaluations']} evaluations of {meta['cases']} instance-size cases")

    def finish(self):
        """Mark the results complete when every stage is done, and list the artifacts."""
        if self._skip("finish"):
            return
        not_done = [s for s in STAGES if self.R["stages"][s]["status"] != "done"]
        arts = [self.out / "pip_freeze.txt"] + [p for d in ("runs", "scripts")
                                                for p in sorted((self.out / d).glob("*")) if p.is_file()]
        self.R["artifacts"] = {str(p.relative_to(self.out)): {"sha256": sha256_file(p), "bytes": p.stat().st_size}
                               for p in arts}
        self.R["status"] = "incomplete" if not_done else "complete"
        self.R["finished_utc"] = _now()
        self.save()
        size = self.results_path.stat().st_size / 1e6
        print(f"{self.results_path} ({size:.1f} MB): status {self.R['status']}"
              + (f"; stages not done: {not_done}" if not_done else ""))
        tot = sum(st.get("seconds", 0) for st in self.R["stages"].values())
        print(f"{len(self.R['commands'])} commands; stage time {tot / 60:.1f} min; "
              f"{len(self.R['artifacts'])} artifacts listed")
