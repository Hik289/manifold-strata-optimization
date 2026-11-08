# Additional experiments and result analysis

Run the training commands in this directory from `experiments/`. The
[experiment guide](../../docs/EXPERIMENTS.md) covers data preparation and direct
training commands. The [configuration guide](../../configs/README.md) explains
how to run a selected group of jobs.

The scripts cover graphs, tabular classification, continual learning, image
data, language, forecasting, rating prediction, sensor drift and generated
Darcy data. Each run fixes its method, data split, training length and settings.
The paper describes how validation scores select configurations for comparison.

## Queue tools

`tools/init_queue.py`, run from the repository root, creates a queue from the
configurations you select. Workers read `revision/queue.json`, take pending
jobs under a file lock and record whether each job completes or fails.
The lock prevents two workers from claiming the same job at the same time.
No active queue is included in the repository.

The activation, restart, priority and handoff scripts manage longer training
sessions on Linux. Read a script's inputs before running it; these tools may
change queue entries or launch many jobs.

## Downloading run records

`pull_remote_snapshot.py` takes `--host`, `--remote-root` and an optional
`--jump` argument. It downloads prepared run records without changing the
remote files or automatically replacing an active local queue. Supply your own
connection settings.

## Analyzing results

Collection scripts include only runs that reach their required training
length. Comparison scripts select settings using validation scores and retain
all competing methods. Memory analysis needs the relevant trusted checkpoints
and reports optimizer-state storage separately from peak allocated GPU memory.

Scripts that update paper tables and loss curves need the original local logs.
The small saved result tables alone do not contain enough information to
rebuild a full training curve or measure checkpoint tensors.

The `archive_*` directories contain earlier implementations for reference.
Use the current scripts named in the experiment guide for new runs. The Muon
baseline is copied unchanged from its upstream repository, with its license
and source commit recorded in `vendor/`.
