# Pilot experiment make targets.
#
# Self-contained: every target invokes the venv interpreter directly,
# so this Makefile works after a clean clone with no parent harness.
# `uv sync --extra test` creates `.venv/` before any target runs.

PYTHON := .venv/Scripts/python.exe
ifneq (,$(wildcard .venv/bin/python))
	# Linux / macOS path inside a uv-managed venv.
	PYTHON := .venv/bin/python
endif

.PHONY: test \
	step-0 step-1 step-2 step-3-encoder \
	data-download build-calibration

test:
	@$(PYTHON) -m pytest tests/ -v

step-0:
	@$(PYTHON) -m pilot.cli.step_0_smoke

step-1:
	@$(PYTHON) -m pilot.cli.step_1_smoke

step-2:
	@$(PYTHON) -m pilot.cli.step_2_kvcache

step-3-encoder:
	@$(PYTHON) -m pilot.cli.step_3_encoder_recall

data-download:
	@$(PYTHON) -m pilot.data.download

build-calibration:
	@$(PYTHON) -m pilot.data.build_calibration_pool
