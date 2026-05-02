"""CLI entrypoints for each pilot step.

Step 0 (this commit): toy-doc smoke that exercises the full plumbing
(prompt template → provider call → ledger row write → price-card
roundtrip).

Steps 1–7 land in subsequent commits as the pilot procedure walks
forward.

Importing this module loads environment variables from `code/.env`
or `./.env` if present (process env takes precedence).
"""

from pilot.env import load_env

# Auto-load .env on first import of any CLI module.
load_env()
