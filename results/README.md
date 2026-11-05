# Frozen evidence snapshot

These tables accompany the included manuscript snapshot (2026-11-05 04:08 UTC).
The ledger contains 556 full-budget configurations; a configuration is not an
independent seed or a complete comparison. The complete-cohort JSON includes
all methods within its stated search budget, including stronger baselines.
All benchmark results use seed 42. No multi-seed significance is claimed.

Configuration selection uses the terminal-100 validation metric. Test scores
never choose the learning rate. Full method grids must finish before a
validation-selected comparison is reported. GPT-2 perplexity is the exponential
of mean validation/test cross-entropy over its terminal evaluation units.
Dataset acquisition scripts preserve splits and training-only preprocessing.

The repository's 856 recipes include candidates that were never run. The
published snapshot includes 278 deferred configurations and one failed LSTM
candidate; none is a completed endpoint. This release does not imply that all
recipes or all proposal-scale experiments are complete. Failed or adverse
comparisons are retained in the manuscript. Raw datasets and full training
checkpoints are not bundled; rerunning a recipe produces local logs.

Optimizer tensor storage (including fixed probes) and peak allocated GPU
memory are separate measurements. A reduction in one is not a reduction in
the other. Shared-server timings are not isolated throughput measurements.
