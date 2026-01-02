# Codex-Style Evaluation for IneqMath

This directory contains a Codex-style implementation for evaluating OpenAI GPT models on the **IneqMath** benchmark [Solving Inequality Proofs with Large Language Models](https://arxiv.org/abs/2506.07927).

IneqMath recasts inequality proving into two automatically checkable subtasks:

- **Bound Estimation** (`type = "bound"`)
- **Relation Prediction** (`type = "relation"`)

Our implementation runs models via the **Codex CLI** and computes **final-answer accuracy** following the official IneqMath format.

---

## Overview

The code in this folder lets you:

1. Run an OpenAI model (e.g. `gpt-4o-mini`) on IneqMath JSON data (`train/dev/test`).
2. Save model outputs in the **official IneqMath format** (`results.json`).
3. Post-process the outputs with a **Final-Answer Judge**:
   - For **relation** questions, check whether the predicted relation matches the ground-truth option.
   - For **bound** questions, extract the predicted bound and optionally use an LLM-as-judge checker to evaluate expression equivalence.

The final outputs include both **per-problem evaluation** and **summary accuracy statistics**, compatible with the IneqMath evaluation platform.:contentReference[oaicite:2]{index=2}  

---

## Architecture

```text
codex/
├── run_codex_baseline.py      # Main script: generate and score IneqMath results with OpenAI models
├── scripts/
│   └── run_ineqmath.sh        # Convenience shell script to run a full evaluation
└── (results/)                 # Generated evaluation outputs (created on run)

