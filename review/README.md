# Computational audit of the GNN results ([arXiv:2609.04593v2](https://arxiv.org/abs/2609.04593v2))

Author: Bartosz Tomsia

Date: 2026-10-25

## Purpose and scope

This notebook is a research report with executable evidence for the manuscript
[GNN-Guided Graph Coarsening and Adaptive QUBO Penalties for the Capacitated Vehicle Routing Problem with Time Windows on a Quantum Annealer](https://arxiv.org/abs/2609.04593v2).
It asks how much of the reported feasibility gain is explained by the learned merge score, and what the released code and measurements allow us to conclude.

The audit compares the tuned heuristic, heuristic-fixed and GNN on paired runs. It examines feasibility before and after expansion, pre-repair metrics, sensitivity to edge orientation and hash salt, the test data and C104 instance, and candidate-selection stability. Each analysis presents its evidence, interpretation, limitations and check statuses. Implementation details and full diagnostic tables are available in the notebook appendix.

The notebook downloads the [released repository](https://github.com/YoussefKamelKamel1/cvrptw-qubo-coarsening) at commit `a82c9e30762dc586f537219ac03cc895d12a2c73`, the pinned predecessor repository and the manuscript PDF. Source versions are specified in `content/audit_config.json` and recorded in the results. Experiments run from the downloaded checkout, not from a local copy of the authors' code.

All solving uses classical simulated annealing on CPU and the released GNN checkpoint. No GPU, D-Wave token or quantum hardware is required. The model is not retrained.

## Required files

Keep the notebook and all eight input files together in this layout:

```text
review/
├── README.md
├── cvrptw_gnn_results_audit.ipynb
└── content/
    ├── audit_runner.py              # environment preparation and experiment execution
    ├── audit_probe.py               # measurements inside the pipeline environment
    ├── audit_verify.py              # verification and statistical analyses
    ├── audit_report.py              # tables, figures and interpretations
    ├── audit_config.json            # sources, QUICK/FULL presets, seeds and budgets
    ├── audit_remarks.json           # source passages used in verification
    ├── audit_fragments.json         # code fragments used in verification
    └── requirements-pipeline.txt    # pinned pipeline dependencies
```

The eight files in `content/` are runtime inputs, not optional documentation. The setup cell reports any missing file; it does not recreate the inputs. No notebook-building scripts or development tests are needed to run the report.

## Running in Google Colab

1. Open `cvrptw_gnn_results_audit.ipynb` in Colab using File → Open notebook → GitHub. Select the `review` branch and open `review/cvrptw_gnn_results_audit.ipynb`.

   Alternatively, click the badge below:

   [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bart-q/cvrptw-qubo-coarsening/blob/review/review/cvrptw_gnn_results_audit.ipynb)

2. In the Files panel, upload all eight files from `review/content/` into `/content`. Alternatively, place them in `/content/content`.
3. In the second code cell, choose the execution mode and grid:

   ```python
   MODE = "generate"
   RUN_FULL = False
   ```

4. Select Runtime → Run all.
5. Read the summary and individual analyses. The appendix provides full tables, check details, source verification and the command record.
6. Save the results before the Colab session ends. The appendix creates `/content/gnn_audit_output.zip`, containing the output directory.

Opening the notebook from GitHub does not supply the eight input files automatically; upload them separately.

## Running locally

Use Python 3.11 or 3.12 for a local notebook kernel. The kernel requires Jupyter, `numpy`, `pandas`, `scipy`, `matplotlib` and IPython; the setup cell installs `pypdf` and `jinja2` if missing.

Start Jupyter with `review/` as the working directory, open the notebook, choose `MODE` and `RUN_FULL`, and run all cells in order. Inputs are read from `./content`.

For command-line execution, with Jupyter available and the notebook switches already set:

```bash
git clone --branch review --single-branch https://github.com/bart-q/cvrptw-qubo-coarsening.git
cd cvrptw-qubo-coarsening/review
jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=-1 \
    cvrptw_gnn_results_audit.ipynb --output executed.ipynb
```

Locally, `audit_work/` and `output/` are created in the working directory. If `/content` exists, the notebook uses it as the base for these generated directories instead.

### Environment and resources

The computation pipeline runs in a separate virtual environment. The notebook installs `torch==2.2.2+cpu` and the pinned packages in `content/requirements-pipeline.txt`, including `numpy==1.26.4`, `dwave-samplers==1.8.0`, `torch-geometric==2.8.0` and `ortools==9.12.4544`. These pipeline packages do not need to be installed manually into the notebook kernel. On a kernel newer than Python 3.12, the runner obtains a Python 3.12 pipeline interpreter using `uv`.

Generating results requires `git`, internet access to GitHub, arXiv, PyPI and the PyTorch package server, and approximately 2 GB of disk space. Execution time depends on the CPU, grid and package installation; FULL performs substantially more work than QUICK. Keep the session running until experiment execution and verification finish.

## Execution modes

The second code cell controls whether to run experiments or reanalyse saved observations.

### New experiments: `MODE = "generate"`

This is the default. The notebook prepares the sources and environment, runs the experiments, writes `output/results.json`, verifies the observations and renders the report.

Starting a new generation run immediately replaces the existing results file with a new incomplete run. Save any earlier results elsewhere before starting. The results remain incomplete until all required experiment stages finish.

### Saved observations: `MODE = "reanalyse"`

This mode skips the experiments and recomputes the verification and report from an existing `output/results.json`.

To use it:

1. Restore a saved results file to `output/results.json` locally, or `/content/output/results.json` in Colab.
2. Provide all eight input files, including the matching generation inputs described below.
3. Set `MODE = "reanalyse"` and keep `RUN_FULL` consistent with the saved run.
4. Run all notebook cells in order.

Reanalysis does not recreate the experiment environment. The optional source preview in the appendix is available only where that environment already exists; otherwise the notebook reports that the preview is unavailable.

### Compatibility of saved results

Verification checks the configuration and the hashes of the generation inputs:

- `audit_runner.py`
- `audit_probe.py`
- `audit_config.json`
- `requirements-pipeline.txt`

A saved result made with different versions of these files is rejected, even if the changes appear cosmetic. Keep a copy of the input files with any result you plan to reanalyse. A result from a different QUICK/FULL configuration is also rejected.

## Choosing QUICK or FULL

| Configuration | QUICK: `RUN_FULL = False` | FULL: `RUN_FULL = True` |
|---|---|---|
| Main grid | N = 10, 20, 40 with seeds 0–2; N = 80, 100 with seed 0 | N = 10, 20, 40, 50, 60, 80, 100 with seeds 0–2 |
| Released scripts run unchanged | `run_scorer_ablation.py` | `run_scorer_ablation.py`, `run_scaling_100.py` |
| Hash salts | Main salt 0, plus 1, 2, 3 | Main salt 0, plus 1–7 |
| Candidate-selection analysis | 12 training instances; N = 10, 15, 20; 5 sampler seeds | 43 training instances; N = 10, 15, 20; 10 sampler seeds |

The main grid covers three arms on the held-out test instances, subject to the released size-availability rule. The main-grid sampler budget is 200 reads × 200 sweeps with adaptive penalties. Configuration details are in `content/audit_config.json`.

QUICK runs every analysis, but it is not a reproduction of the published grid. Comparisons requiring the published grid are reported as `insufficient data`. Choose FULL for those comparisons and for effects that may not be observed on the smaller grid. Running FULL does not remove the other scientific limitations of the audit.

## Outputs

| Location, relative to the execution base | Contents |
|---|---|
| `audit_work/` | Downloaded source repositories, manuscript PDF and pipeline environment |
| `output/results.json` | Observations, configuration, input hashes, source versions and execution provenance |
| `output/generation_log.txt` | Experiment commands and their output |
| `output/verification.json` | Verification results and check statuses |
| `output/summary_checks.csv` and `.md` | Check summaries |
| `output/summary_findings.md` | Generated report summary/implications |
| `output/tables/` and `output/figures/` | Analysis tables and figures |
| `output/runs/` and `output/scripts/` | Recorded run tables and released-script output |
| `output/pip_freeze.txt` | Pipeline package versions |
| `gnn_audit_output.zip` | Output archive written by the appendix |

In Colab, the execution base is `/content`. Save the output archive and the matching input files if you want to preserve a reproducible run.

## Verification and interpretation

Verification reads the saved observations and static audit inputs; it does not run the solver, coarsening or OR-Tools. It rejects missing, invalid, incompatible or incomplete results by default.

Each check has one of three statuses:

- `confirmed`: the observations support the statement tested within the scope of this run.
- `not confirmed`: the observations do not confirm the statement tested.
- `insufficient data`: the available grid or completed stages do not provide the required evidence.

These statuses concern the individual statements tested, not a blanket validation of the method or the audit. The report presents conclusions together with their evidence and limitations.

The source check verifies quoted manuscript passages and cited code/data against the pinned sources. Its detailed output is `output/tables/source_verification.csv`.

### Instrumented runs and the released script

The verification compares all shared main-grid runs with the unchanged `run_scorer_ablation.py`: the same instance, size, seed and arm, at N = 10, 20, 40, seeds 0–2 and hash salt 0. Nine output fields are compared with tolerance `1e-9`; see `output/tables/probe_vs_script.csv`.

This comparison does not cover the larger-size main-grid runs, reversed-edge or deterministic-tie-break variants, alternate-salt re-runs, candidate re-evaluation or data facts. It is not an independent validation of the solver, repair procedure or evaluator.

### If execution fails

Inspect the error and `output/generation_log.txt`. A failed experiment stage leaves the results incomplete, and normal verification refuses the file. For partial diagnostics only, setting `allow_incomplete=True` in the verification cell allows analysis of the available observations; sections without the required data are reported as not assessed. A partial run must not be presented as a complete audit.

## Technical and scientific limitations

- The audit executes the released experiment scripts and functions with the released checkpoint, not every original run underlying the manuscript.
- The baseline probe adds measurements around the released functions. Control variants intentionally reverse edge orientation or introduce a deterministic tie-break; these are identified separately in the notebook.
- Hash salts are set before starting probe processes. The salts used for the manuscript's original runs were not recorded.
- Candidate re-evaluation substitutes the released `gnn_model_rl.pt` for the unpublished `gnn_model_multiN.pt`; it is not an exact reproduction of the original label-generation process.
- The audit does not retrain the models, reproduce Table 3 or recover unavailable checkpoints.
- All solving is classical CPU simulated annealing. No conclusion about quantum-hardware performance follows from these runs.
- The predecessor's code is inspected, not executed. Command-line size overrides in the authors' original executions are not verified.
- QUICK results must not be described as a reproduction of the published grid.

The notebook's Limitations section explains the measurement, statistical, determinism and interpretation limits in detail. Consult it when assessing the findings.

## Authorship and AI assistance

Claude Code (Anthropic) was used under the direction and supervision of Bartosz Tomsia. It generated the code of the notebook and audit modules, proposed measurements, tables and checks implementing the analyses specified by the author, drafted explanatory text and result interpretations, and wrote and ran automated development tests. Parts of the text were revised by hand. These automated tests do not constitute independent validation of the scientific results.

Bartosz Tomsia takes responsibility for the review, including its methods, interpretation of results and final conclusions. The notebook also includes an AI-use declaration in its Limitations section.
