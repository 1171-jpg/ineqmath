import os
import shlex
import subprocess
import tempfile
from typing import Dict, Any
import json


class CodexAgentIneq:
    """A tiny Codex CLI agent for quick experiments in a notebook."""

    BOUND_HINT_PROMPT = "Task description: Please solve the problem with clear, rigorous, and logically sound steps. At the end of your response, state your answer in exactly this format: 'The answer is $C=X$', where X is your calculated numerical bound value. Example: 'The answer is $C=1$'."
    RELATION_HINT_PROMPT = "Task description: Please solve the problem with clear, rigorous, and logically sound steps. At the end of your response, state your answer in exactly this format: 'The answer is (Letter) Symbol', where Letter is one of the given options. Example: 'The answer is (A) $\\leq$'."

    def __init__(self, model_name: str = "gpt-4o"):
        self.model_name = model_name
        
        # Verify codex CLI is available
        try:
            subprocess.run(["codex", "--version"], capture_output=True, check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            raise RuntimeError("Codex CLI not found. Install it or add to PATH.")

    def _run_codex(self, prompt: str, work_dir: str, max_tokens: int = 10000) -> str:
        """Run codex command with the given prompt."""
        escaped_prompt = shlex.quote(prompt)
        cmd = [
            "codex",
            "exec",
            escaped_prompt,
            "--full-auto",
            "--model", self.model_name,
            "--skip-git-repo-check",
        ]

        # print(f"Running command: {' '.join(cmd)}")
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=work_dir,
            env=os.environ.copy(),  # Pass full environment including OPENAI_API_KEY
        )

        if result.returncode != 0:
            raise RuntimeError(f"Codex failed: {result.stderr or result.stdout}")

        return result.stdout
    

    def _build_query(self, problem: str, prob_type: str, task_prompt: str = "", shot_num: int = 0) -> str:
        if prob_type == "bound":
            task_hint = f"{self.BOUND_HINT_PROMPT}\n\n"
        elif prob_type == "relation":
            task_hint = f"{self.RELATION_HINT_PROMPT}\n\n"
        else:
            raise ValueError(f"Unknown problem type: {prob_type}")

        task_prompt = f"{task_prompt}\n\n" if task_prompt else ""
        
        if shot_num == 0:
            demonstrations = ""
        else:
            # TODO: Implement this
            demonstrations = ""  

        query = f"{task_hint}{task_prompt}{demonstrations}Problem: {problem}\n\nSolution:"
        return query
    
    def _build_md_text(self, problem: str, query: str, response: str) -> str:
        text = f"""
## Problem

{problem}

## Query

{query}

## Response

{response}
"""
        return text.strip()
    

    def _process_problem(self, data: Dict[str, Any], output_dir: str, task_prompt: str, shot_num: int, **kwargs) -> None:
        data_id = data["annot_id"] if "annot_id" in data else data["data_id"] # NOTE

        try:
            problem = data["problem"]
            prob_type = data["type"]
            query = self._build_query(problem, prob_type, task_prompt, shot_num)
            response = self._run_codex(query, work_dir=tempfile.gettempdir())
            if not response:
                print(f'response is empty for problem {data_id}')
            if type(response) == str:
                result = {**data, "prompt": query, "response": response, "success": True, "error": None}
            else:
                print(f"Response of problem {data_id} is not a string. Response: {response}")
                result = {**data, "prompt": query, "response": "", "success": False, "error": f"Response is not a string. Response: {response}"}
        except Exception as e:
            print(f"Error processing problem {data_id}: {str(e)}")
            result = {**data, "query": query, "response": "", "success": False, "error": str(e)}

        # print(f"Saving results for {data_id}...")
        self._save_results(result, output_dir, problem, query)

    def _save_results(self, result: Dict, output_dir: str, problem: str, query: str) -> None:
        os.makedirs(output_dir, exist_ok=True)

        data_id = result["annot_id"] if "annot_id" in result else result["data_id"] 
        
        # Save JSON result
        json_path = os.path.join(output_dir, f"{data_id}.json")
        with open(json_path, "w") as f:
            json.dump(result, f, indent=4)

        # Save markdown
        md_text = self._build_md_text(problem, query, result["response"])
        md_path = os.path.join(output_dir, f"{data_id}.md")
        with open(md_path, "w") as f:
            f.write(md_text)



