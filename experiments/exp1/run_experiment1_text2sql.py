#!/usr/bin/env python3
"""
Experiment 1 (StructEval) – Text-to-SQL generation.

This is a model-agnostic, anonymized version of the Experiment 1 script:
- Reads dev examples (question + schema + gold_sql + db_id) from a JSON file.
- For each example, calls a chat model multiple times to generate SQL.
- Saves all generations to a JSONL file for later AST / execution analysis.

Notes:
- This script does NOT contain any API key or provider-specific endpoint.
- The model client is created via a small adapter so that users can plug in
  their own OpenAI-compatible or other chat API.
"""

from __future__ import annotations

from core.generation_io import OpenAIChatClient, write_records, generation_metadata, parse_indices, positive_int

import argparse
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Protocol


@dataclass
class ChatMessage:
    role: str
    content: str


class ChatClient(Protocol):
    """
    Minimal protocol for a chat completion client.

    Users should pass in an implementation with a method:
        chat_completions(model: str, messages: List[ChatMessage], n: int, temperature: float) -> List[str]
    that returns `n` text completions (one SQL string per element).
    """

    def chat_completions(
        self,
        model: str,
        messages: List[ChatMessage],
        n: int,
        temperature: float,
    ) -> List[str]:
        ...


DEFAULT_MODEL_NAME = "gpt-4.1-mini-2025-04-14"


def build_prompt(example: Dict[str, Any]) -> str:
    question = example["question"]
    schema = example["schema"]
    db_id = example["db_id"]

    prompt = f"""You are an expert Text-to-SQL system for the Spider benchmark.
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
    return prompt


def generate_sql_for_example(
    client: ChatClient,
    model_name: str,
    example: Dict[str, Any],
    num_generations: int = 10,
    temperature: float = 1.0,
    max_n_per_call: int = 8,
) -> List[str]:
    """
    Call the chat model to generate `num_generations` SQL candidates.

    If the underlying API only supports n <= max_n_per_call per request, we
    split into multiple calls.
    """
    prompt = build_prompt(example)
    messages = [ChatMessage(role="user", content=prompt)]

    remaining = max(1, num_generations)
    sql_list: List[str] = []

    while remaining > 0:
        this_n = min(max_n_per_call, remaining)
        completions = client.chat_completions(
            model=model_name,
            messages=messages,
            n=this_n,
            temperature=temperature,
        )
        for content in completions:
            sql_text = (content or "").strip()
            sql_list.append(sql_text)
        remaining -= this_n

    return sql_list[:num_generations]


def load_dev_examples(dev_file: Path, limit: int | None = None) -> List[Dict[str, Any]]:
    if not dev_file.exists():
        raise FileNotFoundError(f"dev file not found: {dev_file}")
    with dev_file.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if limit is None:
        return data
    return data[:limit]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "StructEval Experiment 1 – Text-to-SQL generation (model-agnostic, open-source)."
        )
    )
    parser.add_argument(
        "--dev_file",
        type=str,
        required=True,
        help="Path to dev_structeval.json (or compatible dev JSON).",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="Output JSONL file for generations.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default=DEFAULT_MODEL_NAME,
        help="Model name identifier passed to the chat client.",
    )
    parser.add_argument(
        "--max_examples",
        type=positive_int,
        default=50,
        help="Number of dev examples to use (prefix of dev file).",
    )
    parser.add_argument(
        "--num_generations",
        type=positive_int,
        default=10,
        help="Number of SQL generations per question.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature for the model.",
    )
    args = parser.parse_args()

    dev_path = Path(args.dev_file)
    output_path = Path(args.output_file)

    client = OpenAIChatClient()
    examples = load_dev_examples(dev_path, args.max_examples)
    def records():
        for i, example in enumerate(examples):
            sqls = generate_sql_for_example(client, args.model_name, example, args.num_generations, args.temperature)
            yield dict(example, example_index=i, sql_generations=sqls, generation_config=generation_metadata(args, client))
    write_records(output_path, records())


if __name__ == "__main__":
    main()
