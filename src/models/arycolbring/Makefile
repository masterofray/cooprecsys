# Makefile — arycolbring convenience targets
# Requires: make, Python ≥ 3.8, GCC with OpenMP

PYTHON   ?= python3
PIP      ?= pip
PYTEST   ?= pytest

.DEFAULT_GOAL := help

# ── help ─────────────────────────────────────────────────────────────────────
.PHONY: help
help:
	@echo ""
	@echo "  arycolbring — build & test targets"
	@echo ""
	@echo "  make build      Compile Cython extensions in-place"
	@echo "  make install    Build + pip install (editable)"
	@echo "  make test       Run the full pytest suite"
	@echo "  make example    Run the end-to-end demo script"
	@echo "  make clean      Remove compiled artefacts and __pycache__"
	@echo "  make lint       Run basic Python syntax checks"
	@echo ""

# ── build ─────────────────────────────────────────────────────────────────────
.PHONY: build
build:
	@echo "[build] Compiling Cython extensions …"
	$(PYTHON) setup.py build_ext --inplace
	@echo "[build] Done."

# ── install ───────────────────────────────────────────────────────────────────
.PHONY: install
install:
	@echo "[install] Installing in editable mode …"
	$(PIP) install -e .

# ── deps ──────────────────────────────────────────────────────────────────────
.PHONY: deps
deps:
	@echo "[deps] Installing requirements …"
	$(PIP) install -r requirements.txt

# ── test ──────────────────────────────────────────────────────────────────────
.PHONY: test
test: build
	@echo "[test] Running pytest …"
	$(PYTEST) tests/ -v --tb=short 2>&1

# ── test-no-build (if extensions already compiled) ────────────────────────────
.PHONY: test-fast
test-fast:
	$(PYTEST) tests/ -v --tb=short 2>&1

# ── example ───────────────────────────────────────────────────────────────────
.PHONY: example
example: build
	@echo "[example] Running end-to-end demo …"
	$(PYTHON) example.py

# ── clean ─────────────────────────────────────────────────────────────────────
.PHONY: clean
clean:
	@echo "[clean] Removing compiled artefacts …"
	find . -name "*.so"      -delete
	find . -name "*.pyd"     -delete
	find . -name "*.c"       -delete
	find . -name "*.html"    -path "*/cy/*" -delete
	find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
	rm -rf build/ dist/ *.egg-info/
	@echo "[clean] Done."

# ── lint ──────────────────────────────────────────────────────────────────────
.PHONY: lint
lint:
	@echo "[lint] Checking Python syntax …"
	$(PYTHON) -m py_compile arycolbring/__init__.py
	$(PYTHON) -m py_compile arycolbring/model.py
	$(PYTHON) -m py_compile arycolbring/cross_validation.py
	$(PYTHON) -m py_compile arycolbring/evaluation.py
	$(PYTHON) -m py_compile arycolbring/data_utils.py
	$(PYTHON) -m py_compile example.py
	@echo "[lint] All Python files OK."
