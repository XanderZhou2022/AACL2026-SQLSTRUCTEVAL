#!/usr/bin/env python3
"""
Experiment 3 (StructEval) – compile-style generation (open-source version).

This script:
- loads dev_structeval-style examples;
- asks a chat model to output a STRICT JSON query plan (compile-style);
- parses the JSON using core.compile_style_compile_json_to_sql;
- compiles the plan to SQL;
- writes per-example generations to a JSONL file.

The script is model-agnostic and does not contain any API keys. The CLI uses the optional environment-configured adapter; custom clients may
implement the protocol below.
"""

from __future__ import annotations

from core.generation_io import OpenAIChatClient, write_records, generation_metadata, parse_indices, positive_int

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Protocol

from core.compile_style_compile_json_to_sql import (  # type: ignore
    CompileError,
    compile_root_json_to_sql,
    parse_json_strict,
)


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


def load_dev_examples(dev_file: Path) -> List[Dict[str, Any]]:
    if not dev_file.exists():
        raise FileNotFoundError(f"dev file not found: {dev_file}")
    with dev_file.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_prompt(example: Dict[str, Any]) -> str:
    question = example["question"]
    schema = example["schema"]
    db_id = example["db_id"]

    # Prompt text is adapted from the internal script but is provider-agnostic.
    prompt = f"""You are an expert Text-to-SQL system for the Spider benchmark.
Your task is to output a SINGLE strict JSON object that represents a SQL query plan.

CRITICAL RULES:
- Output STRICT JSON only.
  - No natural language, no comments, no Markdown fences, no SQL text.
- Use ONLY tables and columns that exist in the provided database schema.
- The root JSON must be either:
  {{
    "type": "query",
    "query": QUERY_OBJECT
  }}
  or
  {{
    "type": "set_op",
    "set_op": {{
      "op": "union" | "union_all" | "intersect" | "except",
      "left":  QUERY_OBJECT,
      "right": QUERY_OBJECT
    }}
  }}.

QUERY_OBJECT fields:
- select: list of {{"expr": EXPR, "alias": null}}
- from: {{"table": "table_name", "alias": null}}
- joins: list of {{"type": "inner", "table": "table_name", "alias": null, "on": PRED}}
- where, having: lists of PRED combined with AND
- group_by: list of EXPR
- order_by: list of {{"expr": EXPR, "direction": "asc" or "desc"}}
- limit: integer or null; distinct: boolean
EXPR examples: {{"col": ["table_or_alias", "column"]}}, {{"star": true}},
{{"val": "literal"}}, {{"func": "count", "args": [{{"star": true}}]}}.
PRED example: {{"cmp": "=", "left": {{"col": ["table", "column"]}}, "right": {{"val": 1}}}}.
Use empty lists for absent optional clauses.

Database ID: {db_id}
Database schema (JSON):
{json.dumps(schema, ensure_ascii=False, indent=2)}

Question:
{question}
"""
    return prompt


def generate_json_for_example(
    client: ChatClient,
    model_name: str,
    example: Dict[str, Any],
    num_generations: int,
    max_n_per_call: int = 8,
    temperature: float = 1.0,
) -> List[str]:
    prompt = build_prompt(example)
    messages = [ChatMessage(role="user", content=prompt)]
    remaining = max(1, num_generations)
    outs: List[str] = []
    while remaining > 0:
        this_n = min(max_n_per_call, remaining)
        completions = client.chat_completions(
            model=model_name,
            messages=messages,
            n=this_n,
            temperature=temperature,
        )
        for c in completions:
            outs.append((c or "").strip())
        remaining -= this_n
    return outs[:num_generations]


def try_parse_and_compile(raw_text: str, schema_obj: Dict[str, Any]) -> Dict[str, Any]:
    record: Dict[str, Any] = {"raw_json_text": raw_text}
    try:
        root = parse_json_strict(raw_text)
        record["json_parse_ok"] = True
        record["json_obj"] = root
    except Exception as e:
        record["json_parse_ok"] = False
        record["error_stage"] = "json_parse"
        record["error"] = str(e)
        return record

    try:
        sql = compile_root_json_to_sql(root, schema_obj=schema_obj)
        record["compile_ok"] = True
        record["compiled_sql"] = sql
    except Exception as e:
        record["compile_ok"] = False
        record["error_stage"] = "compile"
        record["error"] = str(e)
    return record


def main() -> None:
    parser = argparse.ArgumentParser(
        description="StructEval Experiment 3 – compile-style generation (open-source)."
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
        help="Output JSONL file for compile-style generations.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default=DEFAULT_MODEL_NAME,
        help="Model name identifier passed to the chat client.",
    )
    parser.add_argument(
        "--indices",
        type=str,
        default=None,
        help=(
            "Comma-separated dev indices or ranges, e.g. '0-49,120'. "
            "If omitted, use all examples."
        ),
    )
    parser.add_argument(
        "--num_generations",
        type=positive_int,
        default=10,
        help="Number of JSON generations per question.",
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
    examples = load_dev_examples(dev_path)
    indices = parse_indices(args.indices, len(examples))
    def records():
        for i in indices:
            example = examples[i]
            raw = generate_json_for_example(client, args.model_name, example, args.num_generations, temperature=args.temperature)
            compiled = [try_parse_and_compile(s, example["schema"]) for s in raw]
            # Empty SQL preserves the denominator for failed JSON/compilation attempts.
            sqls = [r.get("compiled_sql", "") for r in compiled]
            yield dict(example, example_index=i, sql_generations=sqls, compile_records=compiled, generation_config=generation_metadata(args, client))
    write_records(output_path, records())


if __name__ == "__main__":
    main()
