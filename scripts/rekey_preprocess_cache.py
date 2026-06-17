"""Re-key the on-disk preprocess cache to a new code_version_hash.

The preprocess cache key includes ``code_version_hash()`` = ``git rev-parse
HEAD`` of ``code/``. Committing a build-irrelevant fix (ledger lock,
embed-metrics) -- or a fix that is a no-op for already-built docs
(``reg_covar``) -- bumps HEAD and would orphan every cached RAPTOR tree /
GraphRAG graph, forcing a full rebuild that re-runs the expensive Gemini
summaries. Instead we RE-KEY the existing artifacts: recompute each entry's
key with the new HEAD (using the code's own ``hash_cache_key``, so it is
exact) and move the entry, so a post-commit lookup hits.

Only entries whose recorded ``code_version_hash == --old-hash`` are
re-keyed; stale entries from other HEADs are left untouched. Run this
AFTER committing the fixes, passing the *pre-commit* HEAD as --old-hash.

    # dry run (default) -- shows what would move, moves nothing:
    python scripts/rekey_preprocess_cache.py --old-hash <pre-commit-HEAD>
    # apply:
    python scripts/rekey_preprocess_cache.py --old-hash <pre-commit-HEAD> --apply

Provenance note on reg_covar: re-keyed RAPTOR trees were built at the old
HEAD without reg_covar. This is sound because reg_covar=1e-4 is a no-op for
the well-conditioned clusterings that built successfully; the collapse
cases failed and were never cached, so they rebuild fresh under the new
HEAD.

Provenance (see docs/CODEMAP.md): MAINTENANCE UTILITY only. It re-keys the
on-disk preprocess cache (RAPTOR trees / GraphRAG graphs) to a new
code_version_hash so a build-irrelevant commit does not orphan the cache and
force expensive Gemini-summary rebuilds. It produces no thesis result -- no
scored cells, figures, or tables -- and is invoked by hand around a commit, not
by the canonical main-study pipeline.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT / "src"))

from pilot.preprocess_cache import hash_cache_key, code_version_hash  # noqa: E402

CACHE_ROOT = CODE_ROOT / "outputs" / "preprocess_cache"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--old-hash", required=True,
                    help="recorded code_version_hash of the entries to re-key (the pre-commit HEAD).")
    ap.add_argument("--new-hash", default=None,
                    help="target hash (default: current git HEAD = the post-commit HEAD).")
    ap.add_argument("--apply", action="store_true",
                    help="actually move entries (default: dry run, moves nothing).")
    args = ap.parse_args()

    new_hash = args.new_hash or code_version_hash()
    if new_hash == args.old_hash:
        print(f"old-hash == new-hash ({new_hash}); nothing to do "
              f"(commit the fixes first so HEAD differs).", file=sys.stderr)
        return 1
    if not CACHE_ROOT.exists():
        print(f"no cache at {CACHE_ROOT}", file=sys.stderr)
        return 1

    metas = sorted(CACHE_ROOT.glob("*/*/*/build_meta.json"))
    rekeyed = skipped_other = already = collisions = 0
    for mp in metas:
        try:
            meta = json.loads(mp.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        cki = meta.get("cache_key_inputs") or {}
        if cki.get("code_version_hash") != args.old_hash:
            skipped_other += 1
            continue
        new_cki = {**cki, "code_version_hash": new_hash}
        new_key = hash_cache_key(new_cki)
        old_dir = mp.parent
        if new_key == old_dir.name:
            already += 1
            continue
        new_dir = old_dir.parent / new_key
        if new_dir.exists():
            print(f"COLLISION (target exists, skipping): {new_dir.relative_to(CACHE_ROOT)}")
            collisions += 1
            continue
        rel = old_dir.relative_to(CACHE_ROOT)
        print(f"{'MOVE ' if args.apply else 'DRY  '}{rel} -> {new_key}")
        if args.apply:
            shutil.move(str(old_dir), str(new_dir))
            meta["code_version_hash"] = new_hash
            meta["cache_key_inputs"] = new_cki
            (new_dir / "build_meta.json").write_text(
                json.dumps(meta, indent=2), encoding="utf-8"
            )
        rekeyed += 1

    verb = "re-keyed" if args.apply else "would re-key"
    print(f"\n{verb}: {rekeyed}   already-correct: {already}   "
          f"other-HEAD (left alone): {skipped_other}   collisions: {collisions}")
    print(f"old {args.old_hash[:12]} -> new {new_hash[:12]}")
    if not args.apply:
        print("(dry run -- re-run with --apply to move)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
