# Optimizer API

Import STRATA and the included low-rank baselines:

```python
from strata_pkg import (
    STRATAAdamW,
    SingleGrassmannAdamW,
    IndependentGrassmannAdamW,
    GaLoreAdamW,
)
```

STRATA uses the usual PyTorch optimizer methods: `zero_grad()`, `step()`,
`state_dict()` and `load_state_dict()`. A complete small example is in
[`examples/quickstart.py`](../examples/quickstart.py).

## Main options

| Option | Meaning |
| --- | --- |
| `lr` | Learning rate for model weights. |
| `weight_decay` | AdamW weight-decay coefficient. |
| `ranks` | Sizes of the nested subspaces, in increasing order. The last value is the largest subspace size. |
| `rhos` | Decay rates for gradient-correlation estimates. A value closer to 1 keeps a longer history. List them from longest to shortest memory. |
| `beta1s`, `beta2s` | Decay rates for the gradient and squared-gradient averages in each block. |
| `a_ks` | Weight assigned to each block's update. |
| `a_perp` | Weight assigned to the gradient component outside the largest subspace. |
| `omegas` | Weights of the subspaces in the objective used to update the basis. |
| `tau` | Interval between basis updates, measured in optimizer steps. |
| `xi` | Step size for updating the basis. |
| `p` | Extra columns used in each compact covariance estimate. The total cannot exceed the matrix dimension. |
| `gamma` | Positive regularizer added before solving the small covariance system. |
| `eps` | Positive constant in the adaptive-update denominator. The default is `1e-6`. |
| `seed` | Seed used to initialize basis matrices and random probes. |
| `adam_betas` | Adam decay rates for parameters that use the dense update. |

`ranks`, `rhos`, `beta1s`, `beta2s`, `a_ks` and `omegas` refer to the same
number of levels. When omitted, the block moment rates are `0.9` and `0.999`,
and the block and objective weights are all `1.0`.

## Subspace sizes and block sizes

The `ranks` values are cumulative. With `ranks=[16, 32, 64]`, the three
subspaces have dimensions 16, 32 and 64. The separate blocks therefore have
sizes **16, 16 and 32**. Gradient components from different blocks are
orthogonal to one another.

Each eligible weight matrix has its own basis and optimizer state. A 4D
convolution kernel is viewed as a matrix before the update and restored to its
original shape afterward. Parameters other than matrices or 4D kernels use
dense AdamW. A matrix also uses dense AdamW when its smaller dimension is
less than or equal to the largest requested rank.

## What happens when the basis changes

STRATA rotates each new basis block to align it with the old block. This is
called **orthogonal Procrustes alignment**. The implementation rotates the
stored gradient average with the corresponding matrix. It resets the
squared-gradient average to zero because those statistics depend on the
coordinate system.

The two averages use separate step counts for Adam's bias correction. The
count for the gradient average continues; the count for the reset
squared-gradient average restarts. The
[algorithm guide](algorithm.md) gives the exact update order and equations.

## Precision and checkpoints

Model weights and optimizer state use float32 in the reference runs. The image
and sequence trainers use BF16 autocast for parts of the model computation;
check each run's `config.json` for its precision settings. The numerical checks
use float64. Distributed and fused-optimizer training are not covered by the
reported experiments.

Save and restore optimizer state with PyTorch's `state_dict()` methods. Load
checkpoints only from sources you trust.
