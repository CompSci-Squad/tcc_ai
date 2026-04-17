"""Unit tests for ExperimentConfig YAML serialization and validation."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from tcc_itransformer.config import ExperimentConfig


class TestConfigYamlRoundTrip:
    def test_yaml_round_trip(self, tmp_path: Path) -> None:
        config = ExperimentConfig(seed=99, latent_dim=6, d_model=32, n_heads=4)
        yaml_path = tmp_path / "config.yaml"
        config.to_yaml(yaml_path)
        loaded = ExperimentConfig.from_yaml(yaml_path)
        assert loaded == config

    def test_yaml_round_trip_all_defaults(self, tmp_path: Path) -> None:
        config = ExperimentConfig()
        yaml_path = tmp_path / "default.yaml"
        config.to_yaml(yaml_path)
        loaded = ExperimentConfig.from_yaml(yaml_path)
        assert loaded == config


class TestConfigValidation:
    def test_n_heads_divides_d_model(self) -> None:
        with pytest.raises(ValueError, match="n_heads.*must divide.*d_model"):
            ExperimentConfig(d_model=32, n_heads=3)

    def test_latent_dim_le_d_model(self) -> None:
        with pytest.raises(ValidationError):
            ExperimentConfig(d_model=32, latent_dim=64)

    def test_valid_config_no_error(self) -> None:
        config = ExperimentConfig(d_model=32, n_heads=4, latent_dim=8)
        assert config.d_model == 32
        assert config.n_heads == 4
        assert config.latent_dim == 8

    def test_invalid_window_size(self) -> None:
        with pytest.raises(Exception):
            ExperimentConfig(window_size=10)


class TestModelDumpForMlflow:
    def test_flat_dict(self) -> None:
        config = ExperimentConfig()
        flat = config.model_dump_for_mlflow()
        for k, v in flat.items():
            assert isinstance(v, (int, float, str)), f"Key {k} has type {type(v)}"

    def test_all_keys_present(self) -> None:
        config = ExperimentConfig()
        flat = config.model_dump_for_mlflow()
        model_keys = config.model_dump().keys()
        assert set(flat.keys()) == set(model_keys)
