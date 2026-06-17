"""CLI entrypoints for the pilot / validation harnesses.

The ``step_0``–``step_4`` and ``phase_f`` CLIs are the calibration-pilot
and methodology-validation harnesses, not the canonical main-study path.
They walk the original pilot procedure forward — from the Step 0 toy-doc
smoke that exercises the full plumbing (prompt template → provider call →
ledger row write → price-card roundtrip) through the later phases — and are
retained as the reproducibility record behind the validation step. The
canonical main-study drivers live alongside the analysis scripts (see
code/docs/CODEMAP.md), not here.

Importing this module loads environment variables from `code/.env`
or `./.env` if present (process env takes precedence).
"""

from pilot.env import load_env

# Auto-load .env on first import of any CLI module.
load_env()
