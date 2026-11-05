<div align="center">
  <h1>STRATA: Flag-Manifold Optimizer State</h1>
  <p align="center">
    <a href="LICENSE"><img alt="MIT license" src="https://img.shields.io/badge/License-MIT-green.svg?style=flat"></a>
    <a href="pyproject.toml"><img alt="Python 3.9+" src="https://img.shields.io/badge/Python-3.9%2B-3776AB.svg?style=flat"></a>
  </p>
</div>

> A single subspace mixes persistent and rapidly changing gradient directions.
> STRATA tracks them through a nested flag and combines blockwise adaptive
> updates in the original, unconstrained weight space.

<p align="center">
  <a href="fig_method_update_detailed.png">
    <img src="fig_method_update_detailed.png" width="100%" alt="Detailed STRATA update: orthogonal gradient splitting, blockwise Adam, lifting and residual correction, followed by multi-timescale covariance sketches, flag retraction, Procrustes alignment and moment transport.">
  </a>
</p>

## The idea

A single gradient subspace mixes directions that persist throughout training with
directions that change quickly. **STRATA-AdamW** represents optimizer state on a
nested flag, $\mathcal U_1\subset\mathcal U_2\subset\mathcal U_3$, and uses
multi-timescale covariance sketches to track this structure online.

| Track | Organize | Update |
| :--- | :--- | :--- |
| Exponential covariance sketches capture different memory horizons. | Nested cumulative subspaces define orthogonal incremental blocks. | Blockwise Adam directions and a normalized residual combine in the original weight space. |

At a flag refresh, the optimizer aligns coordinates using blockwise Procrustes
rotations, transports first moments, and restarts second-moment statistics with
separate bias-correction counters. This changes **optimizer state**, not the
model architecture or the feasible set of weights.

**Included:** the optimizer and geometric primitives; four baseline optimizers;
all experiment trainers, data preparation, ablations, queue recipes, verification
and plotting code; the manuscript source and a frozen results snapshot.

[Algorithm specification](docs/algorithm.md) · [Optimizer API](docs/API.md) · [Reproducibility](docs/REPRODUCIBILITY.md)

## Quickstart

The optimizer and numerical checks run on CPU. The image, stream, graph and
language benchmark trainers generally require CUDA; see the experiment guide
for per-trainer requirements. Linux is recommended for the queue workers.

```bash
git clone https://github.com/Hik289/manifold-strata-optimization.git
cd manifold-strata-optimization
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip

# Install a PyTorch wheel appropriate for your CPU/CUDA platform first if needed.
python -m pip install -e .
python examples/quickstart.py --steps 40
python experiments/test_numerical.py
```

The quickstart is a small, data-free regression problem. The numerical suite
checks orthogonality, nesting, gradient decomposition, energy conservation,
gauge identities, retraction, transport and the one-level reduction to
Single-Grassmann. Neither command launches a benchmark grid.

## Experiments

Install the optional benchmark dependencies:

```bash
python -m pip install -e ".[experiments]"
```

| Task family | Included settings | Entry point |
| :--- | :--- | :--- |
| Image classification | MNIST, Fashion-MNIST, CIFAR-10/100, Tiny-ImageNet, ImageNet-100 | [`train_vision.py`](experiments/train_vision.py), [`train_image_benchmarks.py`](experiments/revision/train_image_benchmarks.py) |
| Continual learning & shifts | Rotated-MNIST, Split-CIFAR-100, CIFAR shifts, DomainNet, CORe50 | [`train_stream.py`](experiments/revision/train_stream.py), [`train_image_benchmarks.py`](experiments/revision/train_image_benchmarks.py) |
| Graph learning | Cora, CiteSeer, PubMed; GCN, GAT, GraphSAGE, GCNII | [`train_graph_extra.py`](experiments/revision/train_graph_extra.py) |
| Tabular classification | Adult, Covertype, red/white Wine Quality | [`train_tabular.py`](experiments/revision/train_tabular.py) |
| Language & sequences | GPT-2 Small/Medium, decoder baseline; WikiText-2 LSTM/GRU/Transformer | [`train_lm.py`](experiments/revision/train_lm.py), [`train_sequence.py`](experiments/revision/train_sequence.py) |
| Forecasting | ETTh1 / ETTh2, L96 / H24 | [`train_new_families.py`](experiments/revision/train_new_families.py) |
| Recommendation | MovieLens 100K / 1M explicit ratings | [`train_new_families.py`](experiments/revision/train_new_families.py) |
| Sensor drift | UCI gas-sensor drift, fine-tuning and replay | [`train_new_families.py`](experiments/revision/train_new_families.py) |
| Scientific ML | Burgers, Poisson, generated Darcy / FNO | [`train_pinn.py`](experiments/train_pinn.py), [`train_poisson.py`](experiments/train_poisson.py), [`train_darcy.py`](experiments/revision/train_darcy.py) |

### Run one configuration

```bash
cd experiments
python revision/prepare_tabular.py --root ../data
python revision/train_tabular.py \
  --dataset wine_red --width 512 --method strata \
  --lr 0.0003 --epochs 1000 --data ../data \
  --out runs_revision/wine_red_example
```

Data preparation downloads the tabular suite and freezes its splits. This is
one full-budget run. A comparison requires every declared method and candidate
configuration; do not select a rate from test outcomes. For a short execution
check, use the trainer's `--smoke --epochs 2` with a **different output folder**.

[Full commands and data setup →](docs/EXPERIMENTS.md)
[All 856 configuration recipes →](configs/experiments.json)
[Choose a queue subset →](configs/README.md)

## Repository map

```text
experiments/strata_pkg/    Optimizer, geometry, baselines and diagnostics
experiments/revision/     Expanded trainers, preprocessing, ablations and audits
experiments/train_*.py    Original vision, graph and scientific-ML trainers
configs/                  Portable experiment recipes and queue instructions
examples/                 Data-free optimizer quickstart
analysis/                 Independent verification scripts
figures/                  Original figure-generation code
data/scripts/             Data-pipeline and gradient-structure checks
tools/                    Queue initialization and release checks
results/                  Frozen, configuration-level evidence tables
paper/                    Manuscript LaTeX source and figures
docs/                     Guides and supporting visuals
fig_method_update_detailed.png  Full-resolution method diagram
```

The historical launchers and archived variants are retained for completeness.
Use the documented direct trainer commands for a first run. Queue activation
and service scripts can start long jobs and are intended for deliberate use
after inspecting their configuration.

## Citation

Use GitHub's **Cite this repository** button, or:

```bibtex
@software{strata2026,
  author = {STRATA Contributors},
  title = {STRATA: Flag-Manifold Optimizer State},
  year = {2026},
  url = {https://github.com/Hik289/manifold-strata-optimization}
}
```
