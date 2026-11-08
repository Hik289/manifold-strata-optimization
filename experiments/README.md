# Experiment code

Install the training dependencies from the repository root:

```bash
python -m pip install -e ".[experiments]"
```

The [experiment guide](../docs/EXPERIMENTS.md) gives data-preparation steps,
CPU/GPU requirements and commands for each task.

| Path | Purpose |
| --- | --- |
| `strata_pkg/` | STRATA, the low-rank baselines and geometry functions. |
| `test_numerical.py` | 13 float64 checks of the optimizer's mathematical identities. |
| `train_*.py` | Original image, graph, Burgers and Poisson training scripts. |
| `revision/` | Additional tasks, experiments that disable or change parts of the method, queue tools and result checks. |
| `run_*.sh` | Original batch launchers. Review their GPU assignments before use. |

Activate your Python environment before running a script. Paths refer to the
repository rather than a particular machine. Datasets, checkpoints and new
training logs are saved locally and ignored by Git. Use a training script's
`--help` option to see its arguments without starting training.
