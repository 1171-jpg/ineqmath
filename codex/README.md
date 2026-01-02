Codex-Style Evaluation for IneqMath
===================================

This directory contains a light-weight implementation for evaluating models on the [IneqMath benchmark](https://arxiv.org/abs/2506.07927) from with Codex CLI intergation.

The implementation:
- reads IneqMath JSON files (e.g. data/json/dev.json),
- calls an OpenAI model (e.g. gpt-4o-mini),
- extracts final answers for each problem,
- verifies correctness (relation / bound),
- and writes both per-problem results and aggregate scores.

You can treat this as a Codex-Style Evaluation for IneqMath in the same spirit as
the original baselines.

--------------------------------------------------
Directory Layout
--------------------------------------------------


```
codex/
├── codex_agent.py       # Core agent class for interacting with Codex CLI
├── run_codex_baseline.py # Main script for running evaluations
├── scripts/             # Convenience shell scripts
│   ├── run_all_codex.sh # Run full evaluation
│   └── test_codex.sh    # Quick test with limited examples
├── codex-setup.sh       # CLI installation script
└── results/             # Generated evaluation outputs (created on run)
```
--------------------------------------------------
Requirements
--------------------------------------------------

1. Node.js: v18+ required for Codex CLI
2. Codex CLI: Install via `npm install -g @openai/codex`
4. Python Dependencies: Listed in parent `requirements.txt`(version compatible with the rest of the repo)

   Example installation (from the repository root):

     pip install -r requirements.txt

3. An OpenAI API key exposed as an environment variable:

     export OPENAI_API_KEY=your_api_key_here

   or stored in a local .env file that is loaded by dotenv.

4. IneqMath data in JSON format, e.g.:
```
   data/json/
   ├── dev.json  # wget https://huggingface.co/datasets/AI4Math/IneqMath/resolve/main/json/dev.json
   └── test.json
```
--------------------------------------------------
Basic Setup
--------------------------------------------------

From the repository root:
```bash
  # (optional) create and activate an environment
  # conda create -n ineqmath python=3.10
  # conda activate ineqmath

  # install Python dependencies
  pip install -r requirements.txt

  # make sure OPENAI_API_KEY is set
  export OPENAI_API_KEY=your_api_key_here

  # Install Codex CLI (if needed)
  cd codex
  chmod +x codex-setup.sh
  ./codex-setup.sh
  
  # Verify installation
  codex --version
```

## Supported Models

The available models depend on your Codex CLI authentication method.

### With OpenAI API Key

| Model Name | Description |
|------------|-------------|
| `gpt-4o` | GPT-4o via Codex CLI |
| `gpt-4o-mini` | GPT-4o-mini via Codex CLI |
| `gpt-4.1` | GPT-4.1 model |
| `gpt-4.1-mini` | GPT-4.1-mini model |
| `o1` | OpenAI o1 reasoning model |
| `o1-mini` | OpenAI o1-mini model |
| `o3` | OpenAI o3 reasoning model |
| `o4-mini` | OpenAI o4-mini model |

**Note**: Check `codex --help` for the latest supported models.

--------------------------------------------------
Quick Start (bash script)
--------------------------------------------------

You can also use the bash wrapper under codex/scripts:
```bash
# bash codex/scripts/run_ineqmath.sh

# By default, the script uses:
# - data_path    = ./data/json/dev.json
# - llm_engine   = gpt-4o-mini
# - run_label    = exp1
# - task_prompt  = ""

cd ineqmath
bash ./codex/scripts/run_all_codex.sh data/json/dev.json gpt-4o-mini exp1
```



--------------------------------------------------
Output Files
--------------------------------------------------

For a given run_label (e.g., exp1), outputs are organized as:

```bash
  ./results/codex_implementation/exp1/
    ├── raw           # folder with raw model responses
    ├── results.json  # summary model responses within json file
    └── scores.json   #  Aggregated statistics across the dataset, split by problem type ("bound" / "relation") and an "all" category.
```



--------------------------------------------------
Troubleshooting
--------------------------------------------------

Common issues and fixes:

- OPENAI_API_KEY not found

  Make sure you either:
  - export the key in your shell (export OPENAI_API_KEY=...), or
  - store it in a .env file and call load_dotenv() at program start.

- Very low accuracy or many empty_responses

  Check:
  - that the prompt clearly instructs the model to output a final answer
    in a consistent format (e.g., "The final answer is ..."),
  - that the extraction functions (locate_answer / extract_bound_answer /
    extract_relation_answer) still match your prompt style.

- FileNotFoundError for JSON or prompt files

  Confirm that:
  - --data_path is correct relative to the repository root,
  - the prompt files exist at the paths given by the *_prompt_path flags.

- Rate limits or network timeouts

  Try:
  - lowering --max_workers,
  - adding simple retries around OpenAI API calls (if not already present).

--------------------------------------------------
Notes and Extensions
--------------------------------------------------

- This implementation is intentionally simple and self-contained, so it is
  easy to swap the underlying model, modify the prompts, or plug in your
  own equivalence checker.

- If you want to run purely symbolic / numeric equivalence (without using
  an LLM as judge), you can replace verify_bound_answer_with_llm with
  your own matach-based checker and keep the rest of the pipeline unchanged.

--------------------------------------------------
Citation
--------------------------------------------------
If you use this implementation, please cite the IneqMath paper:

```bibtex
@misc{lu2025solvinginequalityproofslarge,
      title={Solving Inequality Proofs with Large Language Models}, 
      author={Pan Lu and Jiayi Sheng and Luna Lyu and Jikai Jin and Tony Xia and Alex Gu and James Zou},
      year={2025},
      eprint={2506.07927},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2506.07927}, 
}
```

This README is inspired by Xuandong Zhao's codex implementation on [GPQA](https://github.com/XuandongZhao/gpqa).
