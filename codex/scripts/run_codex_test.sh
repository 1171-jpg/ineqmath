#!/bin/bash
set -e

# Run IneqMath evaluations with Codex CLI

# 1: data_path
Testing_Number="${1:-5}"

echo "Testing IneqMath evaluation with Codex CLI"
echo "================================================"

python ./codex/run_codex_baseline.py \
    --data_path ./data/json/dev.json \
    --llm_engine_name gpt-4o-mini \
    --run_label test \
    --testing_number "$Testing_Number"

echo ""
echo "================================================"
echo "Testing complete!"
echo "Results saved in codex/codex_implementation/test/"
