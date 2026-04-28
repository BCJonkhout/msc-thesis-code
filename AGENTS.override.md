# Code Workspace Rules

- `code/` owns experiment code, transforms, and exporter scripts.
- Raw outputs belong under `code/outputs/`.
- Only promote paper-ready artifacts into `../thesis-msc/generated/` via `make export-assets`.
- Keep scripts deterministic and non-interactive.
- Prefer ASCII filenames for exported assets.
- If a task would take a long time, write logs into `../runs/` and surface the command instead of blocking silently.
