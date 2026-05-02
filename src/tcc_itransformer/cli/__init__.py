"""Typer-based command-line interface for the iTransformer experiments.

Invoke via the ``tcc`` console script (registered in ``pyproject.toml``):

    tcc --help
    tcc data download
    tcc train single --config configs/default.yaml
"""

from tcc_itransformer.cli.main import app

__all__ = ["app"]
