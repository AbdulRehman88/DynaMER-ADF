
from __future__ import annotations

from src.dynamer.models.dynamer_v3_model import DynaMERv3Model

# DynaMER-v4 uses the same dual-path architecture as v3.
# The v4 refinement is controlled by config/training:
# spike_mix=0.00 and label_smoothing=0.05.
DynaMERv4Model = DynaMERv3Model
