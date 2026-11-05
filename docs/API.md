# Optimizer API

```python
from strata_pkg import (
    STRATAAdamW,
    SingleGrassmannAdamW,
    IndependentGrassmannAdamW,
    GaLoreAdamW,
)
```

STRATA follows the standard PyTorch optimizer interface: `zero_grad()`,
`step()`, `state_dict()` and `load_state_dict()`.

| Parameter | Meaning |
| --- | --- |
| `ranks` | Strictly increasing cumulative dimensions; the last is the maximum rank. |
| `rhos` | Covariance-sketch decay rates, ordered from slow to fast memory. |
| `beta1s`, `beta2s` | Per-block adaptive-moment decay rates. |
| `a_ks`, `a_perp` | Weights of incremental-block and residual directions. |
| `tau` | Number of optimizer steps between flag refreshes. |
| `xi` | Riemannian flag step size. |
| `p` | Sketch oversampling above each cumulative rank, capped by dimension. |
| `gamma` | Regularization in the Nyström approximation. |
| `eps` | Projected adaptive-update denominator floor; experiment default is `1e-6`. |
| `seed` | Reproducible initialization of optimizer sketches and bases. |

Nested **cumulative** subspaces share one orthonormal basis. Adaptive moments
live in the mutually orthogonal **incremental** blocks. These are different
objects; `[16, 32, 64]` gives incremental dimensions `[16, 16, 32]`.

At refresh, blockwise Procrustes alignment rotates first moments into the
updated coordinates. Second moments restart and their bias-correction clock
restarts with them; the mature first-moment clock is retained. The
implementation includes a normalized residual direction outside the largest
subspace. Dense weights are not constrained to a manifold.

The included trainers use full-precision float32 (numerical identities use
float64). Mixed-precision, distributed and fused-optimizer configurations are
not established by the reference benchmark protocol. Checkpoint loading uses
PyTorch's standard serialization; load only checkpoints you trust.

See [the detailed specification](algorithm.md) and the implementation for exact
formulas, small-matrix fallback, convolution flattening and state layout.
