"""End-to-end experiment pipelines (training, evaluation, ablations, baselines).

Library-level entry points imported by the typer CLI and the SageMaker
entrypoint. Keeping the pipeline logic inside the package (rather than under
``scripts/``) ensures it is importable without manipulating ``sys.path``.
"""
