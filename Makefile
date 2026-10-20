# TrainHub quality gates [doc 16 FAST-01/02].
# PYTHON selects the backend interpreter:
#   make verify                  (laptop: backend/.venv/bin/python)
#   make verify PYTHON=python    (CI, after `pip install -e "./backend[dev]"`)
PYTHON ?= backend/.venv/bin/python
# Absolute path so `cd backend && $(PY_BACKEND) ...` works from the repo root.
PY_BACKEND := $(abspath $(shell command -v $(PYTHON) 2>/dev/null || echo $(abspath $(PYTHON))))

.PHONY: help verify verify-full test-fast test-full test-gpu e2e dev check-python

help: ## List available targets
	@echo "verify       lint + type-check + fast tests + frontend build  [doc 16 FAST-01]"
	@echo "verify-full  same, but the full suite incl. slow crash tests  [milestones, CI]"
	@echo "test-fast    pytest, excluding slow and gpu tests             [doc 16 FAST-02]"
	@echo "test-full    pytest, everything (milestone-end only)"
	@echo "test-gpu     pytest -m gpu (GPU server only)                  [doc 14 §3.5]"
	@echo "e2e          Playwright wizard smoke (needs backend on :8080)"
	@echo "dev          run backend :8080 + frontend :5173 locally"

verify: check-python ## The fast gate: ruff + mypy + fast tests + frontend build
	cd backend && $(PY_BACKEND) -m ruff check app tests
	cd backend && $(PY_BACKEND) -m mypy app
	$(MAKE) test-fast
	npm --prefix frontend run build

verify-full: check-python ## Full gate incl. slow crash tests [doc 16 FAST-02]
	cd backend && $(PY_BACKEND) -m ruff check app tests
	cd backend && $(PY_BACKEND) -m mypy app
	$(MAKE) test-full
	npm --prefix frontend run build

test-fast: check-python ## Fast tests only (< 60 s target, no network, no GPU) [doc 16 FAST-02]
	cd backend && $(PY_BACKEND) -m pytest -m "not slow and not gpu"

test-full: check-python ## Full suite incl. slow tests; run at milestone end [doc 16 FAST-02]
	cd backend && $(PY_BACKEND) -m pytest

test-gpu: check-python ## GPU smoke tests; requires a CUDA server [doc 16 FAST-02]
	cd backend && $(PY_BACKEND) -m pytest -m gpu

e2e: ## Playwright wizard smoke test; backend must be running on :8080
	npm --prefix frontend run e2e

dev: check-python ## Run backend (:8080) and frontend (:5173) locally
	@trap 'kill 0' INT TERM; \
	(cd backend && TRAINHUB_HOME=./.dev-data $(PY_BACKEND) -m uvicorn app.main:app --port 8080 --reload) & \
	cd frontend && npm run dev

check-python:
	@test -x $(PY_BACKEND) || { echo "backend interpreter '$(PY_BACKEND)' not found; use: make verify PYTHON=python3" >&2; exit 1; }
