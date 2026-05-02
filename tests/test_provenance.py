"""Provenance gate: every value in configs/*.yaml has a non-empty source tag."""
from __future__ import annotations

from pathlib import Path

import pytest

from pilot.provenance import load_and_validate

CONFIGS_DIR = Path(__file__).resolve().parents[1] / "configs"


@pytest.mark.parametrize("name", ["price_card", "methods", "embedding", "models"])
def test_config_provenance(name: str) -> None:
    """Each config file passes the provenance walker without error."""
    path = CONFIGS_DIR / f"{name}.yaml"
    assert path.exists(), f"missing config: {path}"
    config = load_and_validate(path)
    assert isinstance(config, dict)
    assert config, f"{name}.yaml parsed empty"
