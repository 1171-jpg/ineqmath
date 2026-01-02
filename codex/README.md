Codex-Style Evaluation for IneqMath
===================================

This directory contains a light-weight implementation for evaluating OpenAI
models on the IneqMath benchmark from within the `codex/` subfolder.

The implementation:
- reads IneqMath JSON files (e.g. data/json/dev.json),
- calls an OpenAI model (e.g. gpt-4o-mini),
- extracts final answers for each problem,
- verifies correctness (relation / bound),
- and writes both per-problem results and aggregate scores.

You can treat this as an "OpenAI baseline" for IneqMath in the same spirit as
the original Codex/GPQA baselines, but adapted to the IneqMath JSON format.

--------------------------------------------------
Directory Layout
--------------------------------------------------

codex/
├── run_codex_baseline.py        # main driver: run model + score results
├── scripts/
│   └── run_ineqmath.sh          # convenience shell script wrapper
└── (results/)                   # output directory created on first run

- run_codex_baseline.py

  The core script that:
  * parses command-line arguments,
  * loads an IneqMath-style JSON file,
  * builds prompts and calls the model specified by --llm_engine_name,
  * extracts final answers (for bound / relation),
  * optionally calls an LLM-based equivalence checker for bound problems,
  * aggregates scores and writes them to disk.

- scripts/run_ineqmath.sh

  A small helper script that forwards command-line arguments into
  run_codex_baseline.py so that you can run experiments with a single bash
  command instead of typing the full Python invocation every time.

--------------------------------------------------
Requirements
--------------------------------------------------

1. Python 3.10+ (or close)
2. OpenAI Python SDK (version compatible with the rest of the repo)

   Example installation (from the repository root):

     pip install -r requirements.txt

3. An OpenAI API key exposed as an environment variable:

     export OPENAI_API_KEY=your_api_key_here

   or stored in a local .env file that is loaded by dotenv.

4. IneqMath data in JSON format, e.g.:

   data/json/
   ├── train.json
   ├── dev.json
   ├── test.json
   └── theorems.json

--------------------------------------------------
Basic Setup
--------------------------------------------------

From the repository root:

  # (optional) create and activate an environment
  # conda create -n ineqmath python=3.10
  # conda activate ineqmath

  # install Python dependencies
  pip install -r requirements.txt

  # make sure OPENAI_API_KEY is set
  export OPENAI_API_KEY=your_api_key_here

There is no requirement for the Node.js Codex CLI in this implementation;
everything runs through the Python OpenAI client.

--------------------------------------------------
Quick Start (direct Python)
--------------------------------------------------

The main entry point is run_codex_baseline.py. A typical call looks like:

  python codex/run_codex_baseline.py main       --llm_engine_name gpt-4o-mini       --data_path ./data/json/dev.json       --run_label exp1       --task_prompt ""

This will:
- read dev.json,
- evaluate gpt-4o-mini on all problems,
- write results and scores under:

  ./results/codex_implementation/exp1/

--------------------------------------------------
Quick Start (bash script)
--------------------------------------------------

You can also use the bash wrapper under codex/scripts:

  bash codex/scripts/run_ineqmath.sh

By default, the script uses:
- data_path    = data/json/dev.json
- llm_engine   = gpt-4o-mini
- run_label    = exp1
- task_prompt  = ""

If your script accepts positional arguments, you can override these, e.g.:

  bash codex/scripts/run_ineqmath.sh       data/json/test.json       gpt-4o-mini       exp_gpt4omini_test       ""

(adapt this to match the exact signature of your run_ineqmath.sh).

--------------------------------------------------
Command-Line Arguments
--------------------------------------------------

The key arguments defined in run_codex_baseline.py are:

  --llm_engine_name        OpenAI model name (e.g., gpt-4o-mini).
                           Default: "gpt-4o-mini"

  --data_path              Path to the IneqMath JSON file.
                           Default: "./data/json/dev.json"

  --task_prompt            Optional additional task-level instruction
                           prepended to each problem. Default: ""

  --prompt_dir             Directory containing prompt templates.
                           Default: "prompts"

  --run_label              Short string used to name the current run.
                           This becomes the subfolder name under output_path.
                           Default: "exp1"

  --output_path            Root directory for outputs.
                           Default: "./results/codex_implementation"

  --max_workers            Number of workers used for parallel scoring.
                           Default: 16

  --relation_prompt_path   Path to the prompt file used when the model
                           needs help extracting the relation option.
                           Default: "./models/prompts/answer_extraction_relation.md"

  --bound_prompt_path      Path to the prompt file used when the model
                           needs help extracting the numeric/algebraic bound.
                           Default: "./models/prompts/answer_extraction_bound.md"

  --bound_verification_prompt_path
                           Path to the prompt file used when calling an LLM
                           to judge equivalence between the predicted bound
                           and the ground-truth bound.
                           Default: "./models/prompts/answer_verification_bound.md"

--------------------------------------------------
Output Files
--------------------------------------------------

For a given run_label (e.g., exp1), outputs are organized as:

  ./results/codex_implementation/exp1/
    ├── results.json
    └── scores.json

- results.json

  A dictionary keyed by problem id (or data_id). Each entry typically
  contains:
  - the original problem,
  - the model's raw response,
  - the extracted final answer,
  - a small evaluation block with flags like is_solved and is_correct.

- scores.json

  Aggregated statistics across the dataset, split by:
  - data_split (e.g., dev / test),
  - problem type ("bound" / "relation"),
  - and an "all" category.

  For each category, you typically see:
  - total, correct, wrong, empty_responses, accuracy (%).

--------------------------------------------------
High-Level Logic
--------------------------------------------------

At a high level, the pipeline implemented in run_codex_baseline.py does:

1. Load data from --data_path (IneqMath JSON format).

2. For each problem:
   - build a prompt that includes the inequality question,
     plus any global instruction (--task_prompt and templates in --prompt_dir),
   - call the OpenAI model designated by --llm_engine_name,
   - store the raw response string.

3. Post-process the response to find the final answer:
   - locate the sentence that contains "answer is" / "final answer",
     or fall back to the last few lines,
   - for "relation" problems, map the answer into one of the canonical
     options:
       (A) <=, (B) >=, (C) =, (D) <, (E) >, (F) None of the above,
   - for "bound" problems, extract the candidate bound expression, either
     by simple pattern matching (e.g., between $...$) or with another
     LLM prompt (using the bound_prompt_path template).

4. Verify correctness:
   - relation: direct equality check between the extracted option and
     the ground-truth option.
   - bound:
       * first clean up formatting (remove "$" and whitespace),
       * if strings are exactly equal, mark correct,
       * otherwise call an LLM with a strict "equivalence judge" prompt
         (from bound_verification_prompt_path) to decide if the two
         expressions are mathematically identical (no rounding, no
         approximations).

5. Aggregate statistics and save results:
   - build scores per split and per type,
   - write a final scores.json alongside the detailed results.json.

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
  your own CAS-based checker and keep the rest of the pipeline unchanged.
