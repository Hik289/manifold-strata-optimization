# Experiment configurations

[`experiments.json`](experiments.json) lists 856 experiment configurations.
Each entry specifies a training script, its arguments, required data files and
output directory. Some configurations were never run. This file contains no
live job status or machine-specific connection details.

Classification, graph, continual-learning, tabular and sequence runs use 1000
epochs. Generated Darcy experiments use 2000 epochs. GPT-2 uses 1000 evaluation
units, each containing 10 updates: 20.48 million tokens at batch size 8 and
sequence length 256. These units are not full passes through the corpus.
Forecasting, recommendation and sensor-drift comparisons test six settings
per method.

## Create and run a queue

Prepare the data first using the [experiment guide](../docs/EXPERIMENTS.md).
Then choose a task family. From the repository root:

```bash
python tools/init_queue.py --family graph
cd experiments
python revision/run_breadth_queue.py --gpu 0 --families graph
```

The first command creates `experiments/revision/queue.json`. The second starts
a worker that runs the selected graph jobs on GPU 0. It exits when there are
no eligible pending jobs.

To select specific configurations instead, start from a fresh queue and run
this command from the repository root:

```bash
python tools/init_queue.py --ids 20,21,22,23,24,25,26,27,28,29
cd experiments
python revision/run_breadth_queue.py --gpu 0 --families continual
```

Those IDs belong to the `continual` family. The worker's family must match the
selected jobs. The tool rejects unknown IDs, empty selections and attempts to
overwrite an existing queue.

## Missing data and existing queues

A job starts as `pending` when all its declared data files exist. Otherwise it
starts as `waiting_data`, and the worker will skip it. Preparing the files later
does not automatically change that status.

The initializer accepts `--queue` to write a separate queue for inspection, but
`run_breadth_queue.py` reads only `experiments/revision/queue.json`. Do not expect
a custom queue path to change what that worker runs. Keep an active queue in
place; prepare and review a new queue separately if you need to replace it.

The queue tools use Linux/POSIX file locks. For a first setup check, use the
small [quickstart example](../examples/quickstart.py), which launches no batch
of experiments.
