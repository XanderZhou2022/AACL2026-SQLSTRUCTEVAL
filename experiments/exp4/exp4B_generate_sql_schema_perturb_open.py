#!/usr/bin/env python3
"""
Experiment 4B (StructEval) – schema perturbation SQL generation (open-source).

This script mirrors the core logic of the internal Experiment 4B generator:
- For a set of example indices, constructs multiple schema variants
  (original, shuffled tables, shuffled columns).
- For each (example_index, schema_version), calls a chat model multiple
  times to generate SQL and writes the results to a JSONL file.

Inputs:
- dev_structeval-style JSON.
- A JSON file listing unstable example indices (e.g., [0, 12, 45, ...]).

Outputs:
- JSONL with one record per (example_index, schema_version):
  - example_index, db_id, schema_version, input_id, question, schema_snapshot, sql_generations

As with other open scripts, this file is model-agnostic and does not include
hardcoded credentials. The CLI uses the optional environment-configured adapter.
"""

from __future__ import annotations

from core.generation_io import OpenAIChatClient, write_records, generation_metadata, parse_indices, positive_int

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Protocol

from core.schema_perturbation import get_schema_versions


@dataclass
class ChatMessage:
    role: str
    content: str


class ChatClient(Protocol):
    def chat_completions(
        self,
        model: str,
        messages: List[ChatMessage],
        n: int,
        temperature: float,
    ) -> List[str]:
        ...


DEFAULT_MODEL_NAME = "gpt-5-mini-2025-08-07"
MAX_N_PER_REQUEST = 8


def load_dev_all(dev_file: Path) -> List[Dict[str, Any]]:
    with dev_file.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_unstable_indices(indices_file: Path) -> List[int]:
    if not indices_file.exists():
        raise FileNotFoundError(f"Missing unstable indices file: {indices_file}")
    with indices_file.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_prompt(question: str, schema: Dict[str, Any], db_id: str) -> str:
    return f"""You are an expert Text-to-SQL system for the Spider benchmark.
Your task is to write a single valid SQL query for the given question and database schema.

Requirements:
- Use ONLY tables and columns that exist in the provided schema.
- Assume the database is SQLite compatible.
- Return exactly ONE SQL query.
- Do NOT include explanations, comments, or Markdown, only the SQL text.

Database ID: {db_id}
Database schema (JSON):
{json.dumps(schema, ensure_ascii=False, indent=2)}

Question:
{question}

SQL:
"""


def generate_sql_for_one_input(
    client: ChatClient,
    model_name: str,
    question: str,
    schema: Dict[str, Any],
    db_id: str,
    num_generations: int,
    temperature: float = 1.0,
) -> List[str]:
    prompt = build_prompt(question, schema, db_id)
    messages = [ChatMessage(role="user", content=prompt)]
    sql_list: List[str] = []
    remaining = max(1, num_generations)
    while remaining > 0:
        this_n = min(MAX_N_PER_REQUEST, remaining)
        completions = client.chat_completions(
            model=model_name,
            messages=messages,
            n=this_n,
            temperature=temperature,
        )
        for content in completions:
            sql_list.append((content or "").strip())
        remaining -= this_n
    return sql_list[:num_generations]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="StructEval Experiment 4B – schema perturbation SQL generation (open-source)."
    )
    parser.add_argument(
        "--dev_file",
        type=str,
        required=True,
        help="Path to dev_structeval.json.",
    )
    parser.add_argument(
        "--indices_file",
        type=str,
        required=True,
        help="JSON file listing unstable example indices (e.g., [0, 12, ...]).",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="Output JSONL for SQL generations.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default=DEFAULT_MODEL_NAME,
        help="Model name identifier passed to the chat client.",
    )
    parser.add_argument(
        "--num_generations",
        type=positive_int,
        default=10,
        help="Number of SQL generations per (example, schema_version).",
    )
    parser.add_argument("--temperature", type=float, default=1.0)
    args = parser.parse_args()

    dev_path = Path(args.dev_file)
    indices_path = Path(args.indices_file)
    output_path = Path(args.output_file)

    client = OpenAIChatClient()
    examples = load_dev_all(dev_path)
    indices = load_unstable_indices(indices_path)
    def records():
        for i in sorted(set(indices)):
            if not isinstance(i, int) or not 0 <= i < len(examples):
                raise ValueError("Schema-perturbation index is outside the dev file")
            ex = examples[i]
            for version, schema in get_schema_versions(ex["schema"], i).items():
                sqls = generate_sql_for_one_input(client, args.model_name, ex["question"], schema, ex["db_id"], args.num_generations, args.temperature)
                yield dict(ex, example_index=i, schema=schema, schema_snapshot=schema, schema_version=version, input_id="original" if version == "schema_original" else version, sql_generations=sqls, generation_config=generation_metadata(args, client))
    write_records(output_path, records())


if __name__ == "__main__":
    main()
