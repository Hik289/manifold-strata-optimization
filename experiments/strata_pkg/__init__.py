"""STRATA: Multi-Scale Manifold Optimization with Nested Gradient Subspaces.

包结构:
- geometry: 旗流形几何原语 (投影 / 水平梯度 / retraction / Procrustes / Nyström)
- strata_adamw: STRATA-AdamW 主优化器 (README §3)
- baselines: GaLore / Single-Grassmann / Independent-Grassmann
- hooks: §5.3 全量日志诊断钩子 (principal angles / nesting gap / level energy / recon error)
"""

from . import geometry  # noqa: F401

from .strata_adamw import STRATAAdamW
from .baselines import SingleGrassmannAdamW, GaLoreAdamW, IndependentGrassmannAdamW

__all__ = ["STRATAAdamW", "SingleGrassmannAdamW", "GaLoreAdamW", "IndependentGrassmannAdamW", "geometry"]
