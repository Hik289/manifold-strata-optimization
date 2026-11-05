# Reproducibility and release contents

## What is preserved

The release includes all original optimizer and baseline implementations,
trainers, preprocessing scripts, scientific/geometry checks, diagnostic and
figure code, expanded experiment recipes, ablation variants and publication
tools. Historical trainer variants remain in the archive directories for
provenance. Machine-specific paths are replaced by paths relative to the
checkout; the read-only snapshot helper takes explicit connection arguments.
No live queue, process state, SSH configuration or private deployment history
is included. `configs/experiments.json` is a recipe catalog, not a live queue.

The optimizer mathematics is unchanged from the reference experiment code.
The public package adds an import surface, packaging, a small CPU example,
portable paths, complete environment instructions and release checks.
`docs/reference_environment.json` records package versions from the reference
training environment without machine identifiers.

## How comparisons are formed

1. Freeze splits, seed, model, budget, candidate sets and primary metric.
2. Run every declared candidate for every method to the stated budget.
3. Recompute the terminal-100 validation mean from full logs.
4. Select each method's configuration using validation only.
5. Read its corresponding test metric and retain every competing method.

The two original rates are not interchangeable with the later six-candidate
extension. The reduced image design uses its fixed first-listed rate for each
method, including SGD with momentum and Muon. A fixed-rate comparison is not
a completed two-rate search. A stopped/failed run has no fabricated endpoint.

## Numerical checks and local checks

```bash
python tools/check_release.py
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python experiments/test_numerical.py
python examples/quickstart.py --steps 40
```

`test_numerical.py` returns a nonzero status if a check fails and writes an
ignored JSON report. These checks test geometric and update identities; they
do not replace multi-seed empirical evaluation. Additional named-theorem and
bound diagnostics are in `experiments/revision/validate_*.py`,
`actual_theory_diagnostics.py` and `analysis/validate_tight_bounds.py`.

The release was checked on CPU with Python 3.9.16 and PyTorch 2.3.1:
all 13 numerical checks passed, editable installation and import from outside
the checkout passed, and the 40-update quickstart reduced its example loss
from 64.092369 to 59.634636. Queue initialization was checked for explicit
selection, unknown-ID rejection and refusal to overwrite an existing queue.
These are software checks, not additional research experiments. GitHub Actions
runs the source, numerical and quickstart checks on Python 3.11.

## Building the manuscript

With TeX Live and latexmk installed, from the repository root:

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error -cd paper/main.tex
```

The repository includes the manuscript source and PNG figures. Compile locally
to generate the PDF; PDF outputs are excluded from version control. The original
style notices are retained. The source uses the ICML 2026 format; it is not
relabeled as an official 2027 template.

## Data and storage

Datasets and model checkpoints are acquired or generated locally. Do not add
raw dataset archives, tokens, Arrow caches, checkpoints or authentication files
to Git. Frequent checkpointing and geometric diagnostics can use substantial
disk space. A small public results table cannot replace the full logs required
for independent metric recomputation or checkpoint-based resource audits.
