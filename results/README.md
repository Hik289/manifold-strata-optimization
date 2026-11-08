# Saved experiment results

These files match the paper source as of **2026-11-05 04:08 UTC**. They are a
fixed record of that version, not a live training dashboard.

| File | Contents |
| --- | --- |
| [`completed_configurations.csv`](completed_configurations.csv) | 556 runs that reached their specified training length, with settings, validation/test metrics, timing and peak memory. |
| [`complete_cohorts.json`](complete_cohorts.json) | Completed groups of method comparisons, including each method's validation-selected configuration. |

A configuration is one combination of task, model, optimizer and
hyperparameters. It is not a separate random seed or an entire method
comparison. The runs use seed 42, so the tables do not establish statistical
significance across seeds.

## How scores are selected

For each method, the analysis computes the mean validation score over the last
100 epochs or evaluation units. It selects the best candidate using that score,
then reports the corresponding test score. Selection starts only after the
specified candidates for the comparison have finished or recorded a failure.
Test scores do not choose the learning rate.

For language models, perplexity is computed by exponentiating the mean
cross-entropy over the same final evaluation window. Data-preparation scripts
keep training, validation and test splits separate and fit preprocessing on
training data only.

## What the counts mean

The 856 configurations in `configs/` include unrun experiments. At the saved
version above, 278 configurations had been deferred and one LSTM configuration
had failed. Other long runs were still unfinished. Deferred and failed jobs
are not counted as completed runs. The paper retains results where other
methods perform better and records failures.

The `source` field identifies the expected local run folder. Full logs,
datasets and training checkpoints are not included. Rerunning an experiment
creates its logs locally.

## Memory and timing

Optimizer-state storage counts the tensors held by the optimizer, including
fixed random probes. Peak allocated GPU memory measures the largest allocation
during training and also includes model and training tensors. A reduction in
one does not establish a reduction in the other.

Recorded times come from a shared training machine. They should not be read as
speed measurements from an otherwise idle GPU.
