# Running experiments

First install the training dependencies from the repository root:

```bash
python -m pip install -e ".[experiments]"
cd experiments
```

Run all remaining commands on this page from `experiments/`. The `../data`
path points to the repository's local data directory. Each section states
whether its training command needs CUDA.

## Settings shared across experiments

- Classification, graph, tabular, sequence and continual-learning experiments
  use 1000 epochs and seed 42. Generated Darcy experiments use 2000 epochs.
- GPT-2 uses 1000 evaluation units, with 10 updates per unit, batch size 8 and
  sequence length 256. This processes 20,480,000 tokens. A unit is not a full
  pass through the corpus.
- Give each method, learning rate and setting its own output directory.
  Several trainers skip a directory that already contains `summary.json`.
- A `--smoke` run checks that the script executes. Save it in a separate folder
  so it cannot be mistaken for a completed experiment.
- The method names in `revision/` are `strata`, `single`, `adamw`, `galore`
  and `independent`. The image trainer also supports `sgdm` and `muon`.
  Original trainers use `single_grassmann` and `independent_grassmann`.

Start with a direct training command. To run a batch later, follow the
[queue instructions](../configs/README.md).

## Tabular classification — CUDA

```bash
python revision/prepare_tabular.py --root ../data
python revision/train_tabular.py --dataset wine_red --width 512 \
  --method strata --lr 0.0003 --epochs 1000 \
  --data ../data --out runs_revision/wine_red_strata
```

Preparation downloads Adult, red and white Wine Quality, and Covertype. Adult
uses its official test set. Wine Quality is a binary task: quality scores of
6 or higher form the positive class. Preprocessing is fitted on training data
only. Data preparation runs on CPU; this training script requires CUDA.

## Continual learning and changing data distributions — CUDA

```bash
python revision/train_stream.py --task rotated_mnist_cl --base replay \
  --method strata --lr 0.001 --rank 64 --epochs 1000 \
  --data ../data --out runs_revision/rotated_replay_strata
```

MNIST and CIFAR are downloaded through torchvision. Rotated-MNIST presents
ten rotations in sequence, with 100 epochs per rotation and two hidden layers
of width 256. Validation averages performance on tasks seen so far.

Replay keeps a small collection of earlier examples, including their task
rotations. Every method uses the same buffer rule. `finetune`, `replay` and
`ewc` are separate settings; compare methods within the same setting.

## Graph learning — CUDA

```bash
python revision/train_graph_extra.py --dataset PubMed --model sage \
  --method strata --lr 0.001 --rank 64 --epochs 1000 \
  --data ../data --out runs_revision/pubmed_sage_strata
```

The trainer uses the public Planetoid train, validation and test masks.
The included models use a single attention head for GAT and fixed mixing
coefficients for GCNII. Options such as `--ablation` and `--flag-xi` change the
method being tested; report these runs separately from the default settings.

## Forecasting, rating prediction and sensor drift — CPU by default

```bash
python revision/prepare_new_families.py --data ../data
python revision/train_new_families.py --task ETTh2 --base static \
  --method strata --lr 0.001 --flag-xi 0.1 --rank 32 \
  --epochs 1000 --device cpu --threads 2 \
  --data ../data --out runs_revision/etth2_strata_example
```

Supported tasks are `ETTh1`, `ETTh2`, `ml100k`, `ml1m` and `gas_drift`. Use
`--base finetune` or `--base replay` for gas drift and `static` for the other
tasks. The configuration file contains six candidate settings per method;
the command above runs just one of them.

ETT uses a small MLP that predicts 24 time steps from the preceding 96. These
experiments compare optimizers on that model. MovieLens experiments predict
ratings; download the data under the provider's terms.

## Image classification and real-world task sequences — CUDA

Prepare the image lists and splits before training. For Tiny-ImageNet, supply
its extracted dataset directory:

```bash
python revision/prepare_image_benchmarks.py --dataset tinyimagenet \
  --root ../data --source ../data/tiny-imagenet-200
python revision/prepare_image_benchmarks.py --dataset domainnet --root ../data
python revision/prepare_image_benchmarks.py --dataset core50 --root ../data
python revision/train_image_benchmarks.py --dataset tinyimagenet \
  --model resnet18 --base finetune --method strata --lr 0.0003 \
  --epochs 1000 --data ../data --out runs_revision/tiny_strata
```

ImageNet-100 preparation accepts an existing `clane9/imagenet-100` Arrow cache
through `--source`. It checks for 126,689 training images and 5,000 original
validation images. The original validation set becomes the test set; a fixed
hash rule takes internal validation examples from the training set.

DomainNet and CORe50 preparation downloads and checks the public archives.
These datasets are large. The paper source records the exact splits and
image sizes used in the experiments.

## Language models — CUDA

For WikiText-2:

```bash
python revision/prepare_sequence.py --root ../data
python revision/train_sequence.py --model lstm --method strata --lr 0.0003 \
  --epochs 1000 --data ../data --out runs_revision/wikitext2_lstm_strata
```

The script uses the official splits and builds its word vocabulary from the
training set only.

For GPT-2 on WikiText-103:

```bash
python revision/prepare_language.py --root ../data
python revision/train_lm.py --dataset wikitext103 --model gpt2-small \
  --method strata --rank 16 --lr 0.0003 --units 1000 \
  --steps-per-unit 10 --batch 8 --length 256 \
  --data ../data --out runs_revision/wikitext103_gpt2_strata
```

The model starts with random weights and reuses the GPT-2 tokenizer.
`materialize_corpora.py` also includes data-preparation code for OpenWebText
and C4. Its presence does not mean experiments on those datasets have finished.

## Scientific machine learning

The original Burgers and Poisson training scripts are `train_pinn.py` and
`train_poisson.py`. Use `--help` to inspect their options.

The Darcy training script fits a Fourier neural operator (FNO) on generated
data and requires CUDA:

```bash
python revision/train_darcy.py --method strata --lr 0.001 --rank 64 \
  --epochs 2000 --data ../data --out runs_revision/darcy_strata
```

This script generates a 16×16 dataset with a finite-volume solver. It does not
use an external Darcy benchmark. Real and imaginary spectral tensors use dense
optimizer state; eligible pointwise weight matrices use the subspace update.

## Files written by training

Depending on the task, a run writes:

| File or directory | Contents |
| --- | --- |
| `config.json` | Arguments, seed, model and data information. |
| `log.jsonl` | One record per epoch or evaluation unit. |
| `summary.json` | Completion status and summary scores. |
| `flags/` | Saved subspace measurements, where enabled. |
| `checkpoints/` | Model and optimizer checkpoints, where enabled. |

These outputs stay local and are ignored by Git. Analysis scripts recompute
scores from the logs, usually using the final 100 records. Paper-table and
loss-curve scripts need the full logs for the comparison; memory measurements
also need the relevant checkpoints.

Older shell launchers contain fixed GPU assignments. Review them before use.
For a new setup, use the direct commands above before starting a batch.
