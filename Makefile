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
	step-0 step-1 step-2 step-3-encoder step-3-dry-run \
	step-3-dry-run-flat step-3-dry-run-naive-rag \
	step-3-dry-run-raptor step-3-dry-run-graphrag \
	step-3-summary phase-f-kendall \
	data-download build-calibration \
	codabench-format codabench-submit codabench-extract

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

step-3-dry-run:
	@$(PYTHON) -m pilot.cli.step_3_dry_run

step-3-dry-run-flat:
	@$(PYTHON) -m pilot.cli.step_3_dry_run --architectures flat

step-3-dry-run-naive-rag:
	@$(PYTHON) -m pilot.cli.step_3_dry_run --architectures naive_rag

step-3-dry-run-raptor:
	@$(PYTHON) -m pilot.cli.step_3_dry_run --architectures raptor

step-3-dry-run-graphrag:
	@$(PYTHON) -m pilot.cli.step_3_dry_run --architectures graphrag

step-3-summary:
	@test -n "$(RUN)" || (echo "RUN=<run_id> is required" >&2; exit 1)
	@$(PYTHON) -m pilot.cli.step_3_summary --run outputs/runs/$(RUN)

phase-f-kendall:
	@test -n "$(RUN_A)" || (echo "RUN_A=<run_id_primary> is required" >&2; exit 1)
	@test -n "$(RUN_B)" || (echo "RUN_B=<run_id_alternate> is required" >&2; exit 1)
	@$(PYTHON) -m pilot.cli.phase_f_kendall \
		--run-a outputs/runs/$(RUN_A) --run-b outputs/runs/$(RUN_B) \
		--label-a $(if $(LABEL_A),$(LABEL_A),primary) \
		--label-b $(if $(LABEL_B),$(LABEL_B),alternate)

data-download:
	@$(PYTHON) -m pilot.data.download

build-calibration:
	@$(PYTHON) -m pilot.data.build_calibration_pool

codabench-format:
	@test -n "$(PRED)" || (echo "PRED=<predictions.jsonl path> is required" >&2; exit 1)
	@test -n "$(OUT)" || (echo "OUT=<submission.zip path> is required" >&2; exit 1)
	@$(PYTHON) -m pilot.codabench.format --predictions $(PRED) --out $(OUT)

codabench-submit:
	@test -n "$(ZIP)" || (echo "ZIP=<submission.zip path> is required" >&2; exit 1)
	@$(PYTHON) -m pilot.codabench.submit --zip $(ZIP)

codabench-extract:
	@test -n "$(SID)" || (echo "SID=<submission_id> is required" >&2; exit 1)
	@$(PYTHON) -m pilot.codabench.extract_score --submission-id $(SID)
