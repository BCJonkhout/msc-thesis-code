"""Live build / evaluation progress — one self-overwriting ASCII status line.

Deliberately plain: a single line rewritten in place with a carriage return,
using only ASCII, no rich / no ANSI / no Unicode. The earlier rich-based live
region either silently failed to appear (stdout not reported as a TTY under
wrapper scripts / integrated terminals) or crashed encoding box-drawing glyphs
on a legacy cp1252 Windows console. A plain ``\\r`` line — the same mechanism
pip and curl use — renders reliably in any of those contexts and on any code
page.

The line shows overall evaluation progress for this run_index (cells done /
total, an ASCII bar, throughput, ETA) and, while a document build is in flight,
its embedding + LLM call counts — so a long build never looks frozen.
"""
from __future__ import annotations

import sys
import threading
import time
from typing import Any


def tui_supported() -> bool:
    """The plain status line works on any stream, so it is always available."""
    return True


def _fmt_eta(seconds: float) -> str:
    if not (seconds > 0) or seconds != seconds:  # <=0 or NaN
        return "--"
    s = int(seconds)
    h, s = divmod(s, 3600)
    m, s = divmod(s, 60)
    if h:
        return f"{h}h{m:02d}m"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


class RunProgress:
    """Thread-safe single-line progress for one ``run_dry_run`` invocation."""

    def __init__(self, *, enabled: bool) -> None:
        self.enabled = enabled
        self._lock = threading.Lock()
        self._active = False
        self._total = 0
        self._done = 0
        self._start_done = 0
        self._run_index = 0
        self._num_runs = 1
        self._build_label = ""
        self._embeds = 0
        self._google = 0
        self._build_active = False
        self._t0 = 0.0
        self._last = 0.0
        self._interval = 0.5  # seconds between in-place repaints
        self._width = 0       # last line length, for clean overwrite

    # ── lifecycle ────────────────────────────────────────────────────
    def __enter__(self) -> "RunProgress":
        if self.enabled:
            self._t0 = time.monotonic()
            self._active = True
        return self

    def __exit__(self, *exc: Any) -> None:
        if self._active:
            self._render(force=True)
            try:
                sys.stdout.write("\n")
                sys.stdout.flush()
            except Exception:
                pass
            self._active = False

    def log(self, msg: str) -> None:
        """Emit a one-off message on its own line, above the status line."""
        if not self._active:
            print(msg, file=sys.stderr)
            return
        try:
            sys.stdout.write("\r" + (" " * self._width) + "\r")
            print(msg)
            sys.stdout.flush()
            self._width = 0
        except Exception:
            print(msg)
        self._render(force=True)

    # ── evaluation bar ───────────────────────────────────────────────
    def start_eval(self, *, total: int, completed: int,
                   run_index: int, num_runs: int) -> None:
        if not self.enabled:
            return
        with self._lock:
            self._total = total
            self._done = completed
            self._start_done = completed
            self._run_index = run_index
            self._num_runs = num_runs
        self._render(force=True)

    def advance_eval(self, n: int = 1) -> None:
        if not self.enabled:
            return
        with self._lock:
            self._done += n
        self._render(force=True)

    # ── build line ───────────────────────────────────────────────────
    def start_build(self, label: str) -> None:
        if not self.enabled:
            return
        with self._lock:
            self._build_label = label
            self._embeds = 0
            self._google = 0
            self._build_active = True
        self._render(force=True)

    def end_build(self) -> None:
        if not self.enabled:
            return
        with self._lock:
            self._build_active = False

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
        self._render()

    # ── rendering ────────────────────────────────────────────────────
    def _render(self, force: bool = False) -> None:
        if not self._active:
            return
        now = time.monotonic()
        with self._lock:
            if not force and (now - self._last) < self._interval:
                return
            self._last = now
            total = self._total or 1
            done = self._done
            pct = 100.0 * done / total
            nfill = max(0, min(24, int(round(24 * done / total))))
            bar = "#" * nfill + "." * (24 - nfill)
            elapsed = max(1e-6, now - self._t0)
            sess = done - self._start_done
            # Hold off on rate/ETA until there is enough signal, so the first
            # cell after a resume doesn't print an absurd instantaneous rate.
            if sess >= 3 and elapsed >= 5.0:
                per_sec = sess / elapsed
                rate_txt = f"{per_sec * 60.0:4.0f}/min"
                eta = _fmt_eta((total - done) / per_sec)
            else:
                rate_txt = "  --/min"
                eta = "--"
            line = (
                f"[progress] run {self._run_index + 1}/{self._num_runs} "
                f"[{bar}] {pct:5.1f}%  {done}/{total}  {rate_txt}  eta {eta}"
            )
            if self._build_active:
                line += f"  building {self._build_label} (emb {self._embeds} llm {self._google})"
            pad = max(0, self._width - len(line))
            self._width = len(line)
            out = "\r" + line + (" " * pad)
        try:
            sys.stdout.write(out)
            sys.stdout.flush()
        except Exception:
            pass
