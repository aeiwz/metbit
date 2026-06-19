#!/usr/bin/env bash
# run_tests.sh - Run the full metbit test suite with optional performance benchmarks.
#
# Usage:
#   ./scripts/run_tests.sh               # unit + integration tests only
#   ./scripts/run_tests.sh --perf        # also run perf benchmarks and generate report
#   ./scripts/run_tests.sh --perf-quick  # perf benchmarks with smaller datasets
#   ./scripts/run_tests.sh --slow        # include slow/permutation tests
#   ./scripts/run_tests.sh -k stocsy     # pass-through to pytest
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export PYTHONPATH="$ROOT_DIR:${PYTHONPATH:-}"

mkdir -p reports

# Parse metbit-specific flags before passing the rest to pytest
RUN_PERF=0
PERF_QUICK=""
PYTEST_EXTRA=()

for arg in "$@"; do
  case "$arg" in
    --perf)       RUN_PERF=1 ;;
    --perf-quick) RUN_PERF=1; PERF_QUICK="--quick" ;;
    *)            PYTEST_EXTRA+=("$arg") ;;
  esac
done

# ---------------------------------------------------------------------------
# 1. Functional tests
# ---------------------------------------------------------------------------
echo "=== Running functional tests ==="
pytest -q \
  --html=reports/pytest-report.html --self-contained-html \
  --junitxml=reports/junit.xml \
  --cov=metbit \
  --cov-report=term-missing \
  --cov-report=html:reports/coverage \
  -m "not slow and not perf" \
  "${PYTEST_EXTRA[@]+"${PYTEST_EXTRA[@]}"}"

coverage json -o reports/coverage.json

# ---------------------------------------------------------------------------
# 2. Performance benchmarks (optional)
# ---------------------------------------------------------------------------
if [ "$RUN_PERF" -eq 1 ]; then
  echo ""
  echo "=== Running performance benchmarks ==="
  python scripts/perf_report.py $PERF_QUICK --output reports/
fi

# ---------------------------------------------------------------------------
# 3. Test summary (always)
# ---------------------------------------------------------------------------
python scripts/generate_test_summary.py

echo ""
echo "Reports written to: reports/"
