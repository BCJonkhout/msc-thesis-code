"""Provenance-aware YAML config loader.

Per pilot plan § 9, every locked value in `configs/*.yaml` must carry
a non-empty `source:` tag pointing at one of:
  - pilot:5.8#N            pilot decision-matrix row reference
  - pilot:6                pilot plan § 6 decisions table
  - lit:KEY                literature citation key
  - rule:NAME              methodology rule
  - provider:URL           provider documentation snapshot
  - thesis-msc/...         thesis workspace internal reference

This module loads YAML configs, walks the tree, and validates that any
leaf value of the form `{value: ..., source: ...}` has a non-empty
`source` field. Values without the `value:` key are left as-is (e.g.,
plain scalar config like `enabled: false` or rate cards keyed by model
ID).

The validator is the gate behind decision-matrix row #N/A "every
config value carries a source tag".
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


class ProvenanceError(ValueError):
    """Raised when a config value lacks a required source tag."""


_SOURCE_PREFIXES = ("pilot:", "lit:", "rule:", "provider:", "thesis-msc/")


def _is_provenance_block(node: Any) -> bool:
    """A dict that has `value` is a provenance block; it must have `source`."""
    return isinstance(node, dict) and "value" in node


def _validate_source(source: Any, path: str) -> None:
    if not isinstance(source, str) or not source.strip():
        raise ProvenanceError(
            f"{path}: source must be a non-empty string, got {source!r}"
        )
    if not source.startswith(_SOURCE_PREFIXES):
        raise ProvenanceError(
            f"{path}: source {source!r} must start with one of "
            f"{_SOURCE_PREFIXES}"
        )


def _walk(node: Any, path: str, errors: list[str]) -> None:
    if _is_provenance_block(node):
        if "source" not in node:
            errors.append(f"{path}: provenance block has `value` but no `source`")
        else:
            try:
                _validate_source(node["source"], path)
            except ProvenanceError as e:
                errors.append(str(e))
    if isinstance(node, dict):
        for key, value in node.items():
            _walk(value, f"{path}.{key}" if path else key, errors)
    elif isinstance(node, list):
        for i, value in enumerate(node):
            _walk(value, f"{path}[{i}]", errors)


def load_config(path: Path) -> dict[str, Any]:
    """Load a YAML config file and return the parsed dict.

    Does not validate provenance; use `validate_provenance` for that.
    """
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def validate_provenance(config: dict[str, Any], file_label: str = "<config>") -> None:
    """Validate that every provenance block in `config` has a valid `source`.

    Raises ProvenanceError if any block is missing or malformed.
    """
    errors: list[str] = []
    _walk(config, "", errors)
    if errors:
        joined = "\n  - ".join(errors)
        raise ProvenanceError(
            f"{file_label}: {len(errors)} provenance errors:\n  - {joined}"
        )


def load_and_validate(path: Path) -> dict[str, Any]:
    """Load a config file and validate its provenance in one call."""
    config = load_config(path)
    validate_provenance(config, file_label=str(path))
    return config


def get_value(config: dict[str, Any], dotted_path: str) -> Any:
    """Look up a dotted path in a config dict, unwrapping {value: ...} blocks.

    Example:
        >>> cfg = {"sampling": {"temperature": {"value": 0, "source": "rule:T0"}}}
        >>> get_value(cfg, "sampling.temperature")
        0
    """
    node: Any = config
    for part in dotted_path.split("."):
        if not isinstance(node, dict):
            raise KeyError(f"path {dotted_path!r}: cannot descend into {type(node)}")
        if part not in node:
            raise KeyError(f"path {dotted_path!r}: key {part!r} not found")
        node = node[part]
    if _is_provenance_block(node):
        return node["value"]
    return node
