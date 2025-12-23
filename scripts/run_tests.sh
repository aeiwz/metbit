#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export PYTHONPATH="$ROOT_DIR:${PYTHONPATH:-}"

mkdir -p reports

pytest -q \
  --html=reports/pytest-report.html --self-contained-html \
  --junitxml=reports/junit.xml \
  --cov=metbit \
  --cov-report=term-missing \
  --cov-report=html:reports/coverage \
  "$@"

coverage json -o reports/coverage.json

python scripts/generate_test_summary.py
