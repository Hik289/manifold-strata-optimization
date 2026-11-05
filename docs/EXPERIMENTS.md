# Reproducing experiments

Run commands below from `experiments/` after installing `pip install -e
".[experiments]"` from the repository root. `../data` is local to your checkout.
No account, private server or API credential is needed for the optimizer.
Some datasets have their own access and redistribution terms.

## Execution rules

- Formal classifier, graph, tabular, sequence and continual trials use 1000
  epochs and seed 42. Generated Darcy uses 2000 epochs.
- GPT-2 uses a fixed token budget: 1000 units × 10 updates × batch 8 × length
  256 = 20,480,000 tokens. A unit is **not** a full-corpus epoch.
- Use distinct output directories for each method, rate, setting and smoke.
  A directory with an existing `summary.json` is treated as already complete.
- `--smoke` is an execution check, never a formal endpoint. Keep its output
  separate. Logs, flags and frequent checkpoints can require substantial disk.
- The revision method names are `strata`, `single`, `adamw`, `galore`,
  `independent`; image benchmarks additionally support `sgdm` and `muon`.
  Original trainers use `single_grassmann` / `independent_grassmann` instead.
- Queue orchestration uses POSIX file locks and Linux process inspection.
  Direct training commands are the easiest way to start.

## Tabular classification (CPU or CUDA)

```bash
python revision/prepare_tabular.py --root ../data
python revision/train_tabular.py --dataset wine_red --width 512 \
  --method strata --lr 0.0003 --epochs 1000 \
  --data ../data --out runs_revision/wine_red_strata
```

Adult uses its official test split. Wine Quality uses the documented binary
quality target. Preprocessing is fitted on training data only. The preparation
script downloads Adult, both Wine Quality datasets and Covertype.

## Continual learning and artificial shifts (CUDA)

```bash
python revision/train_stream.py --task rotated_mnist_cl --base replay \
  --method strata --lr 0.001 --rank 64 --epochs 1000 \
  --data ../data --out runs_revision/rotated_replay_strata
```

MNIST/CIFAR downloads use torchvision. Rotated-MNIST has ten ordered rotations,
100 epochs per rotation, and a two-layer 256-wide MLP. Validation averages seen
tasks. Replay retains task rotations and uses the same buffer rule for every
method. `--base finetune`, `replay` and `ewc` are separate protocols.

## Graph learning (CUDA)

```bash
python revision/train_graph_extra.py --dataset PubMed --model sage \
  --method strata --lr 0.001 --rank 64 --epochs 1000 \
  --data ../data --out runs_revision/pubmed_sage_strata
```

The trainer uses public Planetoid masks. Models are the included reference
implementations, including single-head GAT and fixed-mixing GCNII. A setting
can specify `--ablation` or `--flag-xi`; do not mix these with default cohorts.

## Forecasting, recommendation and sensor drift (CPU by default)

```bash
python revision/prepare_new_families.py --data ../data
python revision/train_new_families.py --task ETTh2 --base static \
  --method strata --lr 0.001 --flag-xi 0.1 --rank 32 \
  --epochs 1000 --device cpu --threads 2 \
  --data ../data --out runs_revision/etth2_strata_example
```

Tasks: `ETTh1`, `ETTh2`, `ml100k`, `ml1m`, `gas_drift`. Gas drift requires
`--base finetune` or `--base replay`; other tasks use `static`. This is one
candidate, not necessarily the validation-selected winner. The complete
six-candidate-per-method design is retained in the recipes. ETT uses a compact
MLP forecaster; it is not a claim against specialized forecasting architectures.
MovieLens data must be acquired under the provider's terms and are not bundled.

## Image classification and real continual learning (CUDA)

Prepare manifests once. Source folders below are examples you provide:

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
via `--source`. It verifies the expected 126689 training / 5000 validation
images. The original validation images become test data and a fixed hash split
of training images becomes internal validation. Tiny-ImageNet requires its
original extracted directory. DomainNet and CORe50 preparation download and
validate public archives; these downloads are large. The manuscript specifies
the exact split and image-resolution conventions.

## Sequence models and GPT-2 language modeling (CUDA)

```bash
python revision/prepare_sequence.py --root ../data
python revision/train_sequence.py --model lstm --method strata --lr 0.0003 \
  --epochs 1000 --data ../data --out runs_revision/wikitext2_lstm_strata
```

WikiText-2 uses official splits and a training-only word vocabulary. For GPT-2,
prepare WikiText-103 using `prepare_language.py` first:

```bash
python revision/prepare_language.py --root ../data
python revision/train_lm.py --dataset wikitext103 --model gpt2-small \
  --method strata --rank 16 --lr 0.0003 --units 1000 \
  --steps-per-unit 10 --batch 8 --length 256 \
  --data ../data --out runs_revision/wikitext103_gpt2_strata
```

Model weights are initialized from scratch; the GPT-2 tokenizer is reused.
Prospective OpenWebText/C4 materialization is included in
`materialize_corpora.py`; its availability is not a completed benchmark claim.

## Scientific machine learning

Original Burgers and Poisson runners are `train_pinn.py` and `train_poisson.py`.
Their CLI options are listed by `--help`. For the generated Darcy/FNO setting:

```bash
python revision/train_darcy.py --method strata --lr 0.001 --rank 64 \
  --epochs 2000 --data ../data --out runs_revision/darcy_strata
```

The Darcy dataset is a retained 16×16 finite-volume generator, not an external
standard dataset. Spectral real/imaginary tensors use dense optimizer state;
only eligible pointwise matrices use the projected state.

## Outputs and analysis

Revision trainers write `config.json`, `log.jsonl`, `summary.json`, `flags/`
and `checkpoints/` as applicable. These local outputs are ignored by Git.
The audit/collector scripts reconstruct terminal-100 metrics and preserve
failures. Paper publishers expect the complete corresponding local logs and,
for state measurements, the relevant checkpoints. They are not a way to
reconstruct missing measurements from the small public CSV alone.

The original shell launchers retain their historical GPU assignments; inspect
and adjust them before execution. Do not run activation/service scripts as a
quickstart. To reproduce a controlled subset, prefer the
[portable queue initializer](../configs/README.md).
