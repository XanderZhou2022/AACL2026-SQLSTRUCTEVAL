#!/usr/bin/env python3
"""
Experiment 4A (StructEval) – paraphrase SQL generation (open-source).

This script mirrors the core logic of the internal Experiment 4A generator:
- For each selected example_index, uses:
  - the original question
  - several paraphrased variants
- For each (example_index, input_id), calls a chat model multiple times
  to generate SQL, and writes the results to a JSONL file.

Inputs:
- dev_structeval-style JSON with fields including db_id, question, schema.
- a paraphrases JSONL where each line has:
  {
    "example_index": int,
    "paraphrases": [ "para_0", "para_1", ... ]
  }

Outputs:
- JSONL with one record per (example_index, input_id), with fields:
  - example_index, db_id, input_type, input_id, question, schema_snapshot, sql_generations

As with other open scripts, this file is model-agnostic and does not include
hardcoded credentials. The CLI uses the optional environment-configured adapter.
"""

from __future__ import annotations

from core.generation_io import OpenAIChatClient, write_records, generation_metadata, parse_indices, positive_int

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Protocol, Tuple


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


def load_paraphrases(paraphrases_file: Path) -> Dict[int, Dict[str, Any]]:
    if not paraphrases_file.exists():
        raise FileNotFoundError(f"Missing paraphrases file: {paraphrases_file}")
    out: Dict[int, Dict[str, Any]] = {}
    with paraphrases_file.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            out[int(rec["example_index"])] = rec
    return out


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
        description="StructEval Experiment 4A – paraphrase SQL generation (open-source)."
    )
    parser.add_argument(
        "--dev_file",
        type=str,
        required=True,
        help="Path to dev_structeval.json.",
    )
    parser.add_argument(
        "--paraphrases_file",
        type=str,
        required=True,
        help="JSONL with paraphrases per example_index.",
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
        help="Number of SQL generations per (example, input).",
    )
    parser.add_argument(
        "--max_paraphrases",
        type=positive_int,
        default=5,
        help="Maximum number of paraphrases per example to use.",
    )
    parser.add_argument("--temperature", type=float, default=1.0)
    args = parser.parse_args()

    dev_path = Path(args.dev_file)
    paraphrases_path = Path(args.paraphrases_file)
    output_path = Path(args.output_file)

    client = OpenAIChatClient()
    examples = load_dev_all(dev_path)
    paraphrases = load_paraphrases(paraphrases_path)
    def records():
        for i, rec in sorted(paraphrases.items()):
            if not 0 <= i < len(examples):
                raise ValueError("Paraphrase index is outside the dev file")
            ex = examples[i]
            variants = [("original", ex["question"])] + [(f"paraphrase_{j}", q) for j, q in enumerate(rec["paraphrases"][:args.max_paraphrases])]
            for iid, question in variants:
                sqls = generate_sql_for_one_input(client, args.model_name, question, ex["schema"], ex["db_id"], args.num_generations, args.temperature)
                yield dict(ex, example_index=i, question=question, input_id=iid, input_type="original" if iid == "original" else "paraphrase", schema_snapshot=ex["schema"], sql_generations=sqls, generation_config=generation_metadata(args, client))
    write_records(output_path, records())


if __name__ == "__main__":
    main()
