"""Pilot dataset acquisition and split utilities.

Per pilot plan § 2 the pilot evaluates against three datasets:

  - QASPER  (allenai/qasper, dev = calibration, test = held-out)
  - NovelQA (gated; manual download from official repo)
  - QuALITY (nyu-mll/quality, dev = operational evaluation)

This module owns dataset download, split materialisation, and the
calibration-pool builder. All datasets land under code/data/ which
is gitignored — raw corpora and full novel texts are not committed
to either git repo.
"""
