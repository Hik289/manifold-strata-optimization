# Experiment recipes

`experiments.json` contains all 856 configuration recipes from the expanded
experiment design, including unrun candidates. **A recipe is not a completed
result.** Runtime status, worker identifiers, machine names and private paths
have been removed. Optimizer arguments and budgets are preserved.

The full design includes 1000-epoch classifier, graph, continual, tabular and
sequence runs; generated Darcy uses 2000 epochs. GPT-2 uses 1000 evaluation
units of 10 updates each (20.48M tokens at batch 8 and length 256), not 1000
corpus epochs. New forecasting/recommendation/sensor settings use six
candidates per method. Consult the original trainer arguments before running.

Choose an explicit subset; never launch the entire grid just to test setup:

```bash
python tools/init_queue.py --family graph
cd experiments
python revision/run_breadth_queue.py --gpu 0 --families graph
```

Alternatively, initialize a fresh queue with explicit recipe IDs, for example
`python tools/init_queue.py --ids 20,21,22,23,24,25,26,27,28,29` from the
repository root. These IDs belong to `continual`, so use
`--families continual` with the worker for that selection.

If your selection matches no recipes, initialization fails. Existing queues
are never overwritten. Missing declared data prerequisites become
`waiting_data`; rerun initialization after preparing data and choosing a
fresh queue location, or use the original activation tools after inspection.
Queue workers and process handoff scripts target Linux/POSIX systems.
