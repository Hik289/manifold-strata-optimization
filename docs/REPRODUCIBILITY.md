# Reproducing the experiments

## What the repository includes

The repository contains the optimizer and baseline implementations, training
and data-preparation scripts, experiment configurations, numerical checks,
result-analysis tools and figure scripts. Older training versions are kept in
the `archive_*` directories so readers can trace changes to the implementation.

Paths are relative to the repository. Data, model checkpoints, active job
queues and connection credentials are kept outside version control. The
856 entries in [`configs/experiments.json`](../configs/experiments.json) describe
experiments to run; they do not report which jobs are currently running.

The package keeps the optimizer calculations used in the reference experiments
and adds installation support, a small CPU example and portable paths.
[`reference_environment.json`](reference_environment.json) lists the Python
and package versions used on the training machine.

## Comparing methods

1. Fix the data splits, random seed, model, training length, candidate settings
   and main evaluation metric before starting.
2. Run every candidate setting for every method in the comparison.
3. Compute each candidate's mean validation score over its last 100 recorded
   epochs or evaluation units.
4. Select the candidate with the best validation score for each method.
5. Report the matching test score for every method, including stronger baselines.

Some comparisons test two learning rates per method. Later experiments test
six settings per method. The reduced image comparisons use one rate per
method: the first rate listed in the original design, including for SGD with
momentum and Muon. Keep these comparison rules separate when reporting results.
A failed or unfinished run is recorded as such and has no completed result.

## Checking the software

Run from the repository root:

```bash
python tools/check_release.py
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python experiments/test_numerical.py
python examples/quickstart.py --steps 40
```

The numerical suite checks geometric and update identities. It returns a
nonzero exit status when a check fails and saves a local JSON report, which
Git ignores. More checks of the theory and its assumptions are in
[`experiments/revision/`](../experiments/revision/) and
[`analysis/validate_tight_bounds.py`](../analysis/validate_tight_bounds.py).

The initial release was checked on CPU with Python 3.9.16 and PyTorch 2.3.1.
All 13 numerical checks passed. Installation and importing the package from
outside the repository also passed. In the 40-step example, loss fell from
64.092369 to 59.634636. The queue tool was checked for correct selection,
unknown-ID rejection and refusal to overwrite an existing queue.

These checks confirm software behavior; they are not new benchmark results.
GitHub Actions runs the repository checks, numerical suite and small example
with Python 3.11.

## Building the paper

Install TeX Live and latexmk, then run from the repository root:

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error -cd paper/main.tex
```

The repository includes LaTeX source and PNG figures. Generated PDFs stay
local and are ignored by Git. The paper uses the ICML 2026 style files, with
their original notices preserved.

## Data, logs and memory measurements

Download or generate datasets with the preparation scripts described in the
[experiment guide](EXPERIMENTS.md). Training writes logs and checkpoints to
the chosen output directory. Frequent checkpoints can use substantial disk
space.

The saved result tables are useful for reading comparisons. To recompute a
metric or measure stored optimizer tensors, use the full run logs or relevant
checkpoints. Optimizer tensor storage and peak allocated GPU memory measure
different things; report them separately.
