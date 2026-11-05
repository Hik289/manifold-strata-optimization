# Expanded experiments and analysis

Run direct trainer commands from `experiments/`. Start with
[EXPERIMENTS.md](../../docs/EXPERIMENTS.md) and
[the recipe catalog](../../configs/README.md).

Training/preparation modules cover graph learning, tabular data, continual and
shifted tasks, real image streams, language, forecasting, recommendation,
sensor drift and generated Darcy operators. Methods, split rules, full budgets
and validation-based selection are described in the manuscript.

Queue tools use `revision/queue.json`, protected by `queue.lock`. The live queue
is intentionally absent from the distribution. `tools/init_queue.py` creates
an explicit selected subset. A worker claims pending jobs and retains failures.
The Linux activation, restart, priority and handoff helpers are advanced
orchestration utilities; inspect them before running them.

`pull_remote_snapshot.py` accepts `--host`, `--remote-root` and optional `--jump`
for a read-only download. No connection configuration is shipped. Downloaded
snapshots never replace an active local queue automatically.

Postprocessors collect only completed budgets. Selection audits use validation;
full and adverse comparisons remain in the resulting tables. Resource audits
require full trusted checkpoints and count unique tensor storage separately
from peak allocation. Manuscript/curve publishers expect the corresponding
local results; they do not manufacture missing trajectories.

`archive_*` directories retain historical implementations for provenance and
are not the documented current training entry points. The vendored Muon file
is unchanged; its license and upstream commit are recorded beside it.
