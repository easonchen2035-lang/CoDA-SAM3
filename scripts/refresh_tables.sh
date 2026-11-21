#!/bin/bash
PROJECT_ROOT="$(cd -- "$(dirname -- "$0")/.." && pwd)"
cd "$PROJECT_ROOT"
.venv/bin/python scripts/make_test_table.py > /dev/null
.venv/bin/python scripts/make_ablation_tables.py > /dev/null
echo "tables refreshed $(date -Is)"
