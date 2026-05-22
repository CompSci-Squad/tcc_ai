"""PatchTST encoder adapter.

PatchTST (Nie et al., ICLR 2023) applies patching + channel-independent
transformer for time-series representation.  It is architecturally
complementary to the iTransformer: iTransformer mixes variates (columns
as tokens), while PatchTST is variate-independent (patches of each series
as tokens, processed in parallel).

We use the HuggingFace ``transformers.PatchTSTModel`` (encoder-only,
masked pre-training objective) to avoid implementing the architecture from
scratch.

Install:
    uv add --optional phase_e "transformers>=4.40.0,<5.0.0"

Paper: https://arxiv.org/abs/2211.14730
HF:   https://huggingface.co/docs/transformers/model_doc/patchtst
"""

from __future__ import annotations

import logging
from typing import ClassVar

import numpy as np

from tcc_itransformer.encoders.alt.base import AltEncoder

logger = logging.getLogger(__name__)

_PATCHTST_D_MODEL = 64
_PATCHTST_N_HEADS = 4
_PATCHTST_N_LAYERS = 3
_PATCHTST_PATCH_LEN = 4  # patch length (≤ T=6)
_PATCHTST_STRIDE = 2
_PATCHTST_LR = 3e-4
_PATCHTST_N_EPOCHS = 30
_PATCHTST_BATCH_SIZE = 32


class PatchTSTEncoder(AltEncoder):
    """PatchTST encoder trained with masked-patch pre-training.

    Trains a lightweight PatchTST encoder (3 layers, 64 d_model) on the
    training-split windows using the masked-reconstruction objective from
    ``transformers.PatchTSTForPretraining``.

    After training, forward windows through the encoder + mean-pool the
    patch token embeddings to obtain per-window representations.
    """

    name: ClassVar[str] = "patchtst"
    tier: ClassVar[str] = "trainable"
    d_out: ClassVar[int] = _PATCHTST_D_MODEL

    def __init__(
        self,
        d_model: int = _PATCHTST_D_MODEL,
        n_heads: int = _PATCHTST_N_HEADS,
        n_layers: int = _PATCHTST_N_LAYERS,
        patch_len: int = _PATCHTST_PATCH_LEN,
        stride: int = _PATCHTST_STRIDE,
        n_epochs: int = _PATCHTST_N_EPOCHS,
        lr: float = _PATCHTST_LR,
        batch_size: int = _PATCHTST_BATCH_SIZE,
    ) -> None:
        self._d_model = d_model
        self._n_heads = n_heads
        self._n_layers = n_layers
        self._patch_len = patch_len
        self._stride = stride
        self._n_epochs = n_epochs
        self._lr = lr
        self._batch_size = batch_size
        self._model = None
        self._config = None
        self._n_features: int | None = None
        self._seq_len: int | None = None

    def _check_import(self) -> None:
        try:
            import transformers  # noqa: F401, PLC0415
        except ImportError as exc:
            msg = (
                "PatchTST requires transformers: "
                "uv add --optional phase_e 'transformers>=4.40.0,<5.0.0'"
            )
            raise ImportError(msg) from exc

    def fit(self, windows: np.ndarray, seed: int = 42) -> None:
        """Train PatchTST with masked-patch pre-training.

        Parameters
        ----------
        windows : np.ndarray, shape [n_windows, T, N]
        """
        self._check_import()
        import torch  # noqa: PLC0415
        from torch.optim import AdamW  # noqa: PLC0415
        from torch.utils.data import DataLoader, TensorDataset  # noqa: PLC0415
        from transformers import PatchTSTConfig, PatchTSTForPretraining  # noqa: PLC0415

        torch.manual_seed(seed)
        n, T, N = windows.shape
        self._n_features = N
        self._seq_len = T

        logger.info(
            "Training PatchTST: n=%d T=%d N=%d d_model=%d epochs=%d",
            n, T, N, self._d_model, self._n_epochs,
        )

        config = PatchTSTConfig(
            context_length=T,
            patch_length=min(self._patch_len, T),
            stride=min(self._stride, T // 2 or 1),
            num_input_channels=N,
            d_model=self._d_model,
            num_attention_heads=self._n_heads,
            num_hidden_layers=self._n_layers,
            ffn_dim=self._d_model * 4,
            dropout=0.1,
            head_dropout=0.0,
            # Masked pre-training objective
            mask_type="random",
            random_mask_ratio=0.4,
            channel_attention=False,
            positional_encoding_type="sincos",
            use_cls_token=True,
        )
        self._config = config

        model = PatchTSTForPretraining(config)
        model.train()

        X = torch.from_numpy(windows.astype(np.float32))  # [n, T, N]
        loader = DataLoader(
            TensorDataset(X), batch_size=self._batch_size, shuffle=True, drop_last=False
        )
        optimizer = AdamW(model.parameters(), lr=self._lr, weight_decay=1e-4)

        for epoch in range(self._n_epochs):
            total_loss = 0.0
            for (xb,) in loader:
                optimizer.zero_grad()
                out = model(past_values=xb)
                loss = out.loss
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            if (epoch + 1) % 10 == 0:
                logger.info("  epoch %d/%d loss=%.4f", epoch + 1, self._n_epochs, total_loss)

        self._model = model.model  # encoder part (PatchTSTModel)
        self._model.eval()
        logger.info("PatchTST training complete.")

    def encode(self, windows: np.ndarray) -> np.ndarray:
        """Encode windows → patch-token mean-pooled embeddings.

        Parameters
        ----------
        windows : np.ndarray, shape [n_windows, T, N]

        Returns
        -------
        np.ndarray, shape [n_windows, d_model]
        """
        if self._model is None:
            msg = "PatchTSTEncoder.fit() must be called before encode()."
            raise RuntimeError(msg)
        import torch  # noqa: PLC0415

        all_Z: list[np.ndarray] = []
        n = windows.shape[0]

        for start in range(0, n, self._batch_size):
            xb = torch.from_numpy(
                windows[start : start + self._batch_size].astype(np.float32)
            )
            with torch.no_grad():
                out = self._model(past_values=xb)
            # last_hidden_state: [B, N, n_patches, d_model] → mean over N and patches
            h = out.last_hidden_state  # [B, N, n_patches, d_model]
            Z = h.mean(dim=(1, 2)).cpu().numpy()  # [B, d_model]
            all_Z.append(Z)

        return np.concatenate(all_Z, axis=0).astype(np.float32)

    @classmethod
    def is_available(cls) -> bool:
        try:
            from transformers import PatchTSTConfig  # noqa: F401, PLC0415
            return True
        except ImportError:
            return False
