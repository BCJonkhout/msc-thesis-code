"""Live build / evaluation progress display (clean rich TUI).

One live region with two lines:

  * **Evaluation** — a determinate bar over all cells in this run_index
    (``completed / total``) with elapsed + ETA. This is the "main
    evaluation section" progress.
  * **Build** — the activity of the document build currently in flight,
    showing BOTH the Ollama embeddings and the Google summary/extraction
    calls it makes. The counters are driven from the single ``CostLedger``
    row hook every embed + LLM call already passes through, so no progress
    plumbing has to be threaded into the vendored RAPTOR / GraphRAG code.

The display is a no-op when ``enabled`` is false (stdout is not a TTY, or
``--no-tui`` was passed), so redirected / cron runs fall back to plain
logging without emitting terminal control codes.
"""
from __future__ import annotations

import threading
from typing import Any


def tui_supported() -> bool:
    """True when a live TUI can render to this stdout (an interactive TTY)."""
    import sys
    try:
        return bool(sys.stdout) and sys.stdout.isatty()
    except Exception:
        return False


class RunProgress:
    """Thread-safe live progress for one ``run_dry_run`` invocation."""

    def __init__(self, *, enabled: bool) -> None:
        self.enabled = enabled
        self._lock = threading.Lock()
        self._live = None
        self._eval = None
        self._build = None
        self._eval_task = None
        self._build_task = None
        self._embeds = 0
        self._google = 0
        self._build_active = False
        self._build_label = ""

    # ── lifecycle ────────────────────────────────────────────────────
    def __enter__(self) -> "RunProgress":
        if not self.enabled:
            return self
        from rich.console import Group
        from rich.live import Live
        from rich.progress import (
            BarColumn, MofNCompleteColumn, Progress, SpinnerColumn,
            TaskProgressColumn, TextColumn, TimeElapsedColumn,
            TimeRemainingColumn,
        )
        self._eval = Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TaskProgressColumn(),
            TextColumn("elapsed"),
            TimeElapsedColumn(),
            TextColumn("eta"),
            TimeRemainingColumn(),
        )
        self._build = Progress(
            SpinnerColumn(),
            TextColumn("{task.description}"),
        )
        self._live = Live(
            Group(self._eval, self._build),
            refresh_per_second=6,
            transient=False,
            # Capture stray stdout/stderr (per-document cache logs, provider
            # warnings) and render them above the live region instead of
            # letting them tear the bars.
            redirect_stdout=True,
            redirect_stderr=True,
        )
        self._live.__enter__()
        return self

    def __exit__(self, *exc: Any) -> None:
        if self._live is not None:
            try:
                self._live.__exit__(*exc)
            except Exception:
                pass
            self._live = None

    def log(self, msg: str) -> None:
        """Print a line above the live bars (or fall back to stderr)."""
        if self._eval is not None:
            try:
                self._eval.console.log(msg)
                return
            except Exception:
                pass
        import sys
        print(msg, file=sys.stderr)

    # ── evaluation bar ───────────────────────────────────────────────
    def start_eval(self, *, total: int, completed: int,
                   run_index: int, num_runs: int) -> None:
        if not self.enabled or self._eval is None:
            return
        desc = f"Evaluation (run {run_index + 1}/{num_runs})"
        self._eval_task = self._eval.add_task(desc, total=total, completed=completed)
        self._build_task = self._build.add_task("[dim]waiting…[/dim]", total=None)

    def advance_eval(self, n: int = 1) -> None:
        if not self.enabled or self._eval is None or self._eval_task is None:
            return
        self._eval.advance(self._eval_task, n)

    # ── build line ───────────────────────────────────────────────────
    def start_build(self, label: str) -> None:
        if not self.enabled:
            return
        with self._lock:
            self._embeds = 0
            self._google = 0
            self._build_active = True
            self._build_label = label
        self._render_build()

    def end_build(self) -> None:
        if not self.enabled:
            return
        with self._lock:
            self._build_active = False
        if self._build is not None and self._build_task is not None:
            self._build.update(self._build_task, description="[dim]waiting…[/dim]")

    def on_row(self, rec: Any) -> None:
        """CostLedger hook: tally the in-flight build's embed vs LLM calls."""
        if not self.enabled or not self._build_active:
            return
        model = (getattr(rec, "model", "") or "")
        stage = (getattr(rec, "stage", "") or "")
        with self._lock:
            if not self._build_active:
                return
            if model.startswith("bge") or "embed" in model.lower():
                self._embeds += 1
            elif stage == "preprocess":
                self._google += 1
            else:
                return
        self._render_build()

    def _render_build(self) -> None:
        if self._build is None or self._build_task is None:
            return
        with self._lock:
            if not self._build_active:
                return
            label, embeds, google = self._build_label, self._embeds, self._google
        self._build.update(
            self._build_task,
            description=(
                f"🔨 building [bold]{label}[/bold] · "
                f"embeds [cyan]{embeds}[/cyan] · google [magenta]{google}[/magenta]"
            ),
        )
