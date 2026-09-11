#!/usr/bin/env python3
"""
Compute StructEval structural and execution metrics from saved generations.

Inputs:
- dev_structeval.json: provides gold_sql, db_id, and question for each example.
- generations JSONL: per-example records with fields:
  - example_index
  - db_id
  - question
  - gold_sql
  - sql_generations: list of candidate SQL strings

Outputs:
- per-example JSONL with structural and execution metrics
- a JSON summary with dataset-level aggregates

This script depends only on:
- StructEval_OpenSource/core/ast_metrics.py
- StructEval_OpenSource/core/execution_metrics.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from core.ast_metrics import canonicalize_sql, compute_structure_stats
from core.execution_metrics import execute_sql, get_db_path


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def analyze_single_example(
    rec: Dict[str, Any],
    db_root: Path,
    db_ext: str,
    max_rows: int,
) -> Dict[str, Any]:
    example_index = rec.get("example_index")
    db_id = rec.get("db_id")
    question = rec.get("question")
    gold_sql = rec.get("gold_sql") or ""
    sqls: List[str] = rec.get("sql_generations", []) or []
    num_generations = len(sqls)

    # Structural metrics over all generations
    canonical_all: List[Optional[str]] = []
    for sql in sqls:
        canon, _err = canonicalize_sql(sql)
        canonical_all.append(canon)
    struct_all = compute_structure_stats(canonical_all)

    # Execution metrics and correctness vs gold
    db_path = get_db_path(db_root=db_root, db_id=str(db_id), db_ext=db_ext)

    exec_ok_flags: List[bool] = []
    is_correct_flags: List[bool] = []

    with db_path.open("rb"):
        from core.execution_metrics import open_readonly_database

        with open_readonly_database(db_path) as conn:
            if gold_sql.strip():
                gold_ok, gold_result, gold_error = execute_sql(
                    conn=conn,
                    sql=gold_sql,
                    max_rows=max_rows,
                )
            else:
                gold_ok, gold_result, gold_error = False, None, "Empty gold_sql"

            for sql in sqls:
                ok, result, _err = execute_sql(conn=conn, sql=sql, max_rows=max_rows)
                exec_ok_flags.append(ok)

                is_correct = False
                if ok and gold_ok and gold_result is not None and result is not None:
                    is_correct = result == gold_result
                is_correct_flags.append(is_correct)

    num_exec_success = sum(1 for ok in exec_ok_flags if ok)
    num_exec_error = num_generations - num_exec_success
    num_correct = sum(1 for c in is_correct_flags if c)

    execution_accuracy_all = num_correct / num_generations if num_generations > 0 else 0.0
    execution_accuracy_on_exec_success = (
        num_correct / num_exec_success if num_exec_success > 0 else 0.0
    )

    # Structural metrics restricted to execution-correct generations
    canonical_correct: List[Optional[str]] = [
        canonical_all[i] for i in range(num_generations) if is_correct_flags[i]
    ]
    if canonical_correct:
        struct_correct = compute_structure_stats(canonical_correct)
    else:
        struct_correct = {
            "num_generations": 0,
            "num_parse_success": 0,
            "num_parse_fail": 0,
            "distinct_structures": 0,
            "entropy": 0.0,
            "entropy_normalized": 0.0,
            "majority_structure_ratio": 0.0,
        }

    # Gold canonical structure and alignment with correct generations
    if gold_sql.strip():
        gold_canon, gold_parse_error = canonicalize_sql(gold_sql)
    else:
        gold_canon, gold_parse_error = None, "Empty gold_sql"

    gold_match_count_correct = 0
    if gold_canon is not None and num_correct > 0:
        for i in range(num_generations):
            if is_correct_flags[i] and canonical_all[i] == gold_canon:
                gold_match_count_correct += 1
    any_correct_equal_to_gold = gold_match_count_correct > 0
    all_correct_equal_to_gold = num_correct > 0 and gold_match_count_correct == num_correct

    ast_similarity_correct = struct_correct["majority_structure_ratio"]
    exec_correct_and_struct_diff = num_correct >= 2 and ast_similarity_correct < 0.7

    result: Dict[str, Any] = {
        "example_index": example_index,
        "db_id": db_id,
        "question": question,
        "gold_sql": gold_sql,
        "num_generations": num_generations,
        "num_exec_success": num_exec_success,
        "num_exec_error": num_exec_error,
        "num_exec_correct": num_correct,
        "execution_accuracy_all": execution_accuracy_all,
        "execution_accuracy_on_exec_success": execution_accuracy_on_exec_success,
        # Structural metrics on all generations
        "all_num_parse_success": struct_all["num_parse_success"],
        "all_num_parse_fail": struct_all["num_parse_fail"],
        "all_distinct_structures": struct_all["distinct_structures"],
        "all_entropy": struct_all["entropy"],
        "all_entropy_normalized": struct_all["entropy_normalized"],
        "all_majority_structure_ratio": struct_all["majority_structure_ratio"],
        # Structural metrics on correct generations
        "correct_num_generations": struct_correct["num_generations"],
        "correct_num_parse_success": struct_correct["num_parse_success"],
        "correct_num_parse_fail": struct_correct["num_parse_fail"],
        "correct_distinct_structures": struct_correct["distinct_structures"],
        "correct_entropy": struct_correct["entropy"],
        "correct_entropy_normalized": struct_correct["entropy_normalized"],
        "correct_majority_structure_ratio": struct_correct["majority_structure_ratio"],
        "ast_similarity_correct": ast_similarity_correct,
        # Gold alignment
        "gold_canon": gold_canon,
        "gold_parse_error": gold_parse_error,
        "gold_match_count_correct": gold_match_count_correct,
        "any_correct_equal_to_gold": any_correct_equal_to_gold,
        "all_correct_equal_to_gold": all_correct_equal_to_gold,
        # Killer metric
        "exec_correct_and_struct_diff": exec_correct_and_struct_diff,
        "high_acc_low_struct": (
            execution_accuracy_all >= 0.8 and struct_correct["distinct_structures"] > 1
        ),
    }
    return result


def summarize(per_example: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    stats = list(per_example)
    if not stats:
        return {}

    def avg(field: str) -> float:
        vals = [float(s.get(field, 0.0)) for s in stats]
        return (sum(vals) / len(vals)) if vals else 0.0

    num_examples = len(stats)
    num_high_acc_low_struct = sum(1 for s in stats if s.get("high_acc_low_struct", False))
    num_exec_correct_examples = sum(1 for s in stats if s.get("num_exec_correct", 0) > 0)
    num_exec_correct_and_struct_diff = sum(
        1 for s in stats if s.get("exec_correct_and_struct_diff", False)
    )

    return {
        "num_examples": num_examples,
        "avg_execution_accuracy_all": avg("execution_accuracy_all"),
        "avg_execution_accuracy_on_exec_success": avg(
            "execution_accuracy_on_exec_success"
        ),
        "avg_all_distinct_structures": avg("all_distinct_structures"),
        "avg_correct_distinct_structures": avg("correct_distinct_structures"),
        "avg_all_majority_structure_ratio": avg("all_majority_structure_ratio"),
        "avg_correct_majority_structure_ratio": avg("correct_majority_structure_ratio"),
        "avg_ast_similarity_correct": avg("ast_similarity_correct"),
        "num_high_acc_low_struct": num_high_acc_low_struct,
        "fraction_high_acc_low_struct": (
            num_high_acc_low_struct / num_examples if num_examples > 0 else 0.0
        ),
        "high_acc_threshold": 0.8,
        "low_struct_condition": "correct_distinct_structures > 1",
        "ast_similarity_threshold": 0.7,
        "num_exec_correct_examples": num_exec_correct_examples,
        "num_exec_correct_and_struct_diff": num_exec_correct_and_struct_diff,
        "fraction_exec_correct_and_struct_diff_over_all": (
            num_exec_correct_and_struct_diff / num_examples if num_examples > 0 else 0.0
        ),
        "fraction_exec_correct_and_struct_diff_over_correct": (
            (num_exec_correct_and_struct_diff / num_exec_correct_examples)
            if num_exec_correct_examples > 0
            else 0.0
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute StructEval structural and execution metrics from generations."
    )
    parser.add_argument(
        "--dev_file",
        type=str,
        required=False,
        help=(
            "Path to dev_structeval.json (optional; used only for sanity checks). "
            "If omitted, the script only relies on the generations JSONL."
        ),
    )
    parser.add_argument(
        "--generations_file",
        type=str,
        required=True,
        help="JSONL file with model generations.",
    )
    parser.add_argument(
        "--db_root",
        type=str,
        required=True,
        help="Root directory containing Spider-style SQLite databases.",
    )
    parser.add_argument(
        "--db_ext",
        type=str,
        default=".sqlite",
        help="File extension for SQLite databases (default: .sqlite).",
    )
    parser.add_argument(
        "--max_rows",
        type=int,
        default=10_000,
        help="Maximum number of rows to fetch per query when comparing results.",
    )
    parser.add_argument(
        "--max_examples",
        type=int,
        default=None,
        help="Optional maximum number of examples to analyze (by example_index).",
    )
    parser.add_argument(
        "--stats_out",
        type=str,
        required=True,
        help="Output JSONL file for per-example metrics.",
    )
    parser.add_argument(
        "--summary_out",
        type=str,
        required=True,
        help="Output JSON file for aggregate summary.",
    )

    args = parser.parse_args()

    gen_path = Path(args.generations_file)
    if not gen_path.exists():
        raise SystemExit(f"Generations file not found: {gen_path}")

    records = load_jsonl(gen_path)
    if args.max_examples is not None:
        max_examples = int(args.max_examples)
        records = [r for r in records if int(r.get("example_index", -1)) < max_examples]

    db_root = Path(args.db_root)
    db_ext = args.db_ext
    max_rows = int(args.max_rows)

    stats: List[Dict[str, Any]] = []
    for idx, rec in enumerate(records):
        ex_idx = rec.get("example_index")
        db_id = rec.get("db_id")
        print(f"[{idx+1}/{len(records)}] example_index={ex_idx} db_id={db_id}")
        try:
            stat = analyze_single_example(
                rec=rec,
                db_root=db_root,
                db_ext=db_ext,
                max_rows=max_rows,
            )
        except Exception as e:
            stat = {
                "example_index": ex_idx,
                "db_id": db_id,
                "question": rec.get("question"),
                "gold_sql": rec.get("gold_sql"),
                "error": str(e),
            }
            print(f"  ERROR during analysis: {e}")
        stats.append(stat)

    stats_out = Path(args.stats_out)
    with stats_out.open("w", encoding="utf-8") as f:
        for s in stats:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    summary = summarize(stats)
    summary_out = Path(args.summary_out)
    with summary_out.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("Done.")


if __name__ == "__main__":
    main()
