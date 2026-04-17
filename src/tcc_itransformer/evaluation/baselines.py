"""Baseline models for clustering comparison: random, raw PCA, linear AE, windowed PCA."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score


class RandomEmbeddingBaseline:
    """B0: Random embeddings → K-Means."""

    def generate(
        self, n_samples: int, n_dims: int, random_state: int = 42
    ) -> np.ndarray:
        rng = np.random.default_rng(random_state)
        return rng.standard_normal((n_samples, n_dims))


class RawPCABaseline:
    """B1: PCA on raw features (no windowing, no model) → K-Means."""

    def __init__(self) -> None:
        self._pca: PCA | None = None

    def fit_transform(
        self, train_data: np.ndarray, n_components: int
    ) -> np.ndarray:
        self._pca = PCA(n_components=n_components)
        return self._pca.fit_transform(train_data)

    def transform(self, data: np.ndarray) -> np.ndarray:
        if self._pca is None:
            msg = "Must call fit_transform before transform."
            raise RuntimeError(msg)
        return self._pca.transform(data)


class LinearAEBaseline:
    """B2: Single-layer linear autoencoder → extract bottleneck → K-Means.

    Uses torch for a simple Linear(input_dim, latent) → Linear(latent, input_dim)
    model. Train for n_epochs on the same data with MSE loss.
    """

    def __init__(
        self, input_dim: int, latent_dim: int, random_state: int = 42
    ) -> None:
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.random_state = random_state
        torch.manual_seed(random_state)
        self.encoder = nn.Linear(input_dim, latent_dim)
        self.decoder = nn.Linear(latent_dim, input_dim)

    def fit(
        self,
        windows: np.ndarray,
        n_epochs: int = 100,
        lr: float = 1e-3,
    ) -> None:
        """Train the linear autoencoder on flattened windows."""
        # Flatten windows: (N, W, F) -> (N, W*F) or already 2D
        if windows.ndim == 3:
            flat = windows.reshape(windows.shape[0], -1)
        else:
            flat = windows

        x = torch.from_numpy(flat.astype(np.float32))
        optimizer = torch.optim.Adam(
            list(self.encoder.parameters()) + list(self.decoder.parameters()),
            lr=lr,
        )
        criterion = nn.MSELoss()

        self.encoder.train()
        self.decoder.train()
        for _ in range(n_epochs):
            optimizer.zero_grad()
            z = self.encoder(x)
            x_hat = self.decoder(z)
            loss = criterion(x_hat, x)
            loss.backward()
            optimizer.step()

    def transform(self, windows: np.ndarray) -> np.ndarray:
        """Extract bottleneck embeddings."""
        if windows.ndim == 3:
            flat = windows.reshape(windows.shape[0], -1)
        else:
            flat = windows

        x = torch.from_numpy(flat.astype(np.float32))
        self.encoder.eval()
        with torch.no_grad():
            return self.encoder(x).numpy()


class WindowedPCABaseline:
    """B3: PCA on flattened windows (W*N → latent_dim) → K-Means."""

    def __init__(self) -> None:
        self._pca: PCA | None = None

    def fit_transform(
        self, windows: np.ndarray, n_components: int
    ) -> np.ndarray:
        if windows.ndim == 3:
            flat = windows.reshape(windows.shape[0], -1)
        else:
            flat = windows
        self._pca = PCA(n_components=n_components)
        return self._pca.fit_transform(flat)

    def transform(self, windows: np.ndarray) -> np.ndarray:
        if self._pca is None:
            msg = "Must call fit_transform before transform."
            raise RuntimeError(msg)
        if windows.ndim == 3:
            flat = windows.reshape(windows.shape[0], -1)
        else:
            flat = windows
        return self._pca.transform(flat)


def run_all_baselines(
    train_windows: np.ndarray,
    eval_windows: np.ndarray,
    n_components: int,
    k: int,
    random_state: int = 42,
) -> dict[str, dict]:
    """Run all 4 baselines, return results per baseline.

    Returns:
        ``{name: {'embeddings': np.ndarray, 'labels': np.ndarray, 'silhouette': float}}``
    """
    results: dict[str, dict] = {}
    n_eval = eval_windows.shape[0]

    # B0: Random
    b0 = RandomEmbeddingBaseline()
    emb0 = b0.generate(n_eval, n_components, random_state=random_state)
    km0 = KMeans(n_clusters=k, n_init=20, random_state=random_state)
    lbl0 = km0.fit_predict(emb0)
    results["random"] = {
        "embeddings": emb0,
        "labels": lbl0,
        "silhouette": float(silhouette_score(emb0, lbl0)),
    }

    # B1: Raw PCA (treat flattened windows as raw features)
    b1 = RawPCABaseline()
    flat_train = train_windows.reshape(train_windows.shape[0], -1) if train_windows.ndim == 3 else train_windows
    flat_eval = eval_windows.reshape(eval_windows.shape[0], -1) if eval_windows.ndim == 3 else eval_windows
    b1.fit_transform(flat_train, n_components)
    emb1 = b1.transform(flat_eval)
    km1 = KMeans(n_clusters=k, n_init=20, random_state=random_state)
    lbl1 = km1.fit_predict(emb1)
    results["raw_pca"] = {
        "embeddings": emb1,
        "labels": lbl1,
        "silhouette": float(silhouette_score(emb1, lbl1)),
    }

    # B2: Linear AE
    input_dim = flat_train.shape[1]
    b2 = LinearAEBaseline(input_dim, n_components, random_state=random_state)
    b2.fit(train_windows)
    emb2 = b2.transform(eval_windows)
    km2 = KMeans(n_clusters=k, n_init=20, random_state=random_state)
    lbl2 = km2.fit_predict(emb2)
    results["linear_ae"] = {
        "embeddings": emb2,
        "labels": lbl2,
        "silhouette": float(silhouette_score(emb2, lbl2)),
    }

    # B3: Windowed PCA
    b3 = WindowedPCABaseline()
    b3.fit_transform(train_windows, n_components)
    emb3 = b3.transform(eval_windows)
    km3 = KMeans(n_clusters=k, n_init=20, random_state=random_state)
    lbl3 = km3.fit_predict(emb3)
    results["windowed_pca"] = {
        "embeddings": emb3,
        "labels": lbl3,
        "silhouette": float(silhouette_score(emb3, lbl3)),
    }

    return results
