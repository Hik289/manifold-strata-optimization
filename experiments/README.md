# Experiment code

Install the package from the repository root with `python -m pip install -e
".[experiments]"`. See [the complete experiment guide](../docs/EXPERIMENTS.md)
for data preparation, device requirements and copyable commands.

- `strata_pkg/`: optimizer, geometry, baselines and diagnostics.
- `test_numerical.py`: 13 float64 numerical checks, including the one-level reduction.
- `train_*.py`: original vision, graph, Burgers and Poisson trainers.
- `revision/`: expanded task families, controlled ablations, queues and audits.
- `run_*.sh`: original full-budget launchers; inspect GPU assignments before use.

All defaults use checkout-relative paths. Activate your Python environment
before launching. Data, checkpoints and generated results are not part of the
source distribution. To inspect a trainer without training, use `--help`.
