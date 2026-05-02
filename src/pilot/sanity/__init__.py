"""Plumbing-correctness sanity checks (Phase B / decision-matrix rows #1–5).

- kvcache_check.py:    KV-cache reuse verification per provider (row #1, #5)
- mc_postprocessor.py: Multiple-choice answer parser (row #3)

The cost-ledger correctness assertion (row #1) and cached/uncached
attribution assertion (row #2) live in the test suite under
`tests/test_ledger.py`.

The provider rate-limit headroom check (row #4) is a Step 1
concern, not Step 0.
"""
from pilot.sanity.mc_postprocessor import parse_mc_answer

__all__ = ["parse_mc_answer"]
