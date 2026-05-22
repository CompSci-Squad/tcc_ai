"""Phase E alternative encoder package.

Exposes all AltEncoder implementations and the central REGISTRY dict.

Usage:
    from tcc_itransformer.encoders.alt import REGISTRY, AltEncoder

    enc = REGISTRY["moment"]()
    Z_train = enc.encode(train_windows)  # [n, T, N] → [n, d]
    enc.fit(train_windows)               # no-op for zero-shot encoders
"""

from __future__ import annotations

from tcc_itransformer.encoders.alt.base import AltEncoder
from tcc_itransformer.encoders.alt.moment_enc import MOMENTEncoder
from tcc_itransformer.encoders.alt.moirai_enc import MoiraiEncoder
from tcc_itransformer.encoders.alt.ts2vec_enc import TS2VecEncoder
from tcc_itransformer.encoders.alt.patchtst_enc import PatchTSTEncoder
from tcc_itransformer.encoders.alt.timesnet_enc import TimesNetEncoder
from tcc_itransformer.encoders.alt.tfc_enc import TFCEncoder
from tcc_itransformer.encoders.alt.classical_enc import (
    HamiltonHMMEncoder,
    MSVAREncoder,
    BOCPDEncoder,
)

# Central registry — name → callable (returns AltEncoder instance).
# Run `make phase-e` to execute all entries in this registry.
REGISTRY: dict[str, type[AltEncoder]] = {
    # ── Foundation models (zero-shot, $0) ─────────────────────────────────
    "moment": MOMENTEncoder,
    "moirai": MoiraiEncoder,
    # ── Self-supervised / trainable neural (CPU, ~5-15 min each) ──────────
    "ts2vec": TS2VecEncoder,
    "patchtst": PatchTSTEncoder,
    "timesnet": TimesNetEncoder,
    "tfc": TFCEncoder,
    # ── Classical macro baselines (no embedding, $0) ──────────────────────
    "hamilton_hmm": HamiltonHMMEncoder,
    "ms_var": MSVAREncoder,
    "bocpd": BOCPDEncoder,
}

__all__ = [
    "AltEncoder",
    "REGISTRY",
    "MOMENTEncoder",
    "MoiraiEncoder",
    "TS2VecEncoder",
    "PatchTSTEncoder",
    "TimesNetEncoder",
    "TFCEncoder",
    "HamiltonHMMEncoder",
    "MSVAREncoder",
    "BOCPDEncoder",
]
