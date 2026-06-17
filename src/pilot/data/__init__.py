"""Dataset acquisition and split utilities.

The study evaluates against two datasets (§ 2):

  - QASPER  (allenai/qasper, dev = calibration, test = held-out;
             scored with local Answer-F1)
  - NovelQA (gated HuggingFace dataset; multiple choice scored against
             held-out Codabench gold)

QuALITY was scoped out and is no longer a live dataset; the downloader
below still carries QuALITY acquisition code as historical scaffolding.

This module owns dataset download, split materialisation, and the
calibration-pool builder. All datasets land under code/data/ which
is gitignored — raw corpora and full novel texts are not committed
to either git repo.
"""
