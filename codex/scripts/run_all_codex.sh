#!/bin/bash
set -e

# Run IneqMath evaluations with Codex CLI

# 1: data_path
DATA_PATH="${1:-./data/json/dev.json}"
# 2: llm_engine_name
LLM_ENGINE_NAME="${2:-gpt-4o-mini}"
# 3: run_label
RUN_LABEL="${3:-exp1}"

echo "Running IneqMath evaluation with Codex CLI"
echo "Data path : $DATA_PATH"
echo "LLM engine: $LLM_ENGINE_NAME"
echo "Run Label : $RUN_LABEL"
echo "================================================"

python ./codex/run_codex_baseline.py \
    --data_path "$DATA_PATH" \
    --llm_engine_name "$LLM_ENGINE_NAME" \
    --run_label "$RUN_LABEL"

echo ""
echo "================================================"
echo "All evaluations complete!"
echo "Results saved in codex/codex_implementation/${RUN_LABEL}/"
