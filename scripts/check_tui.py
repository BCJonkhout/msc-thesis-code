"""Quick self-test for the live progress display.

Renders a few seconds of the same bars the main study uses, so you can confirm
the TUI works in *your* terminal before launching the multi-day run. Exits 0 on
success. If your console cannot render it, it falls back to plain text instead
of crashing.

Run:  .\.venv\Scripts\python.exe scripts\check_tui.py

Provenance (see docs/CODEMAP.md): DEV UTILITY only. It is a self-test for the
live progress display and produces no result that feeds the thesis -- no scored
cells, no figures, no tables. It exists purely to verify the TUI renders in a
given terminal before committing to the multi-day main-study run.
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pilot.progress as P  # noqa: E402


class _Row:
    def __init__(self, model, stage):
        self.model = model
        self.stage = stage


def main() -> int:
    print(f"stdout encoding at startup: {sys.stdout.encoding} | isatty: {sys.stdout.isatty()}")
    rp = P.RunProgress(enabled=True)  # force on, like the launcher's --tui
    rp.__enter__()
    rp.start_eval(total=40, completed=8, run_index=0, num_runs=5)
    rp.start_build("B43 . graphrag")
    for i in range(40):
        rp.on_row(_Row("bge-m3", "retrieval"))
        if i % 4 == 0:
            rp.on_row(_Row("gemini-3.1-flash-lite-preview", "preprocess"))
        if i % 5 == 0:
            rp.advance_eval()
        # A stray log line (the path that crashed before): must NOT crash now.
        if i == 20:
            print("[selftest] a normal log line while the bar is live")
        time.sleep(0.05)
    rp.end_build()
    rp.__exit__(None, None, None)
    print(f"stdout encoding now: {sys.stdout.encoding}")
    print("SELFTEST OK - the live progress display works in this terminal.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
