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

# RAPTOR's vendored UMAP+GMM clustering and GraphRAG's
# NetworkX+faiss imports both pull in OpenMP-threaded native code.
# Pinning to single-thread before Python starts avoids the Windows
# segfault we hit on first import. Python-level os.environ.setdefault
# is too late — OpenMP reads the env var at native-library-load time.
export OMP_NUM_THREADS := 1
export NUMBA_NUM_THREADS := 1

.PHONY: test \
	step-0 step-1 step-2 step-3-encoder step-3-dry-run \
	step-3-dry-run-flat step-3-dry-run-naive-rag \
	step-3-dry-run-raptor step-3-dry-run-graphrag \
	step-3-summary step-4-variance phase-f-kendall phase-f-pareto \
	resume-phase-f1-v2 \
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

step-4-variance:
	@test -n "$(RUNS)" || (echo "RUNS='<run_id_1> <run_id_2> ...' is required" >&2; exit 1)
	@$(PYTHON) -m pilot.cli.step_4_variance \
		--runs $(addprefix outputs/runs/,$(RUNS)) \
		--architecture $(if $(ARCH),$(ARCH),flat) \
		--metric $(if $(METRIC),$(METRIC),answer_f1)

phase-f-pareto:
	@test -n "$(REF)" || (echo "REF=<reference_run_id> is required" >&2; exit 1)
	@test -n "$(CANDIDATES)" || (echo "CANDIDATES='<id_a> <id_b> ...' is required" >&2; exit 1)
	@$(PYTHON) -m pilot.cli.phase_f_pareto \
		--reference outputs/runs/$(REF) \
		--reference-label $(if $(REF_LABEL),$(REF_LABEL),reference) \
		--candidates $(addprefix outputs/runs/,$(CANDIDATES)) \
		$(if $(OUT),--out $(OUT))

# Crash-recovery helper for the Phase F.1 v2 sweep. Auto-detects the
# most recent matching run dir, ensures Ollama is up, and resumes
# (Flash Lite / 4-arch / QASPER / literature prompt). Idempotent —
# re-run after a laptop crash to pick up where the sweep left off.
resume-phase-f1-v2:
	@bash scripts/resume_phase_f1_v2.sh

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
