#!/usr/bin/env python3
"""
Experiment 2 (StructEval) – Execution accuracy vs structural reliability.

This script is an open-source, model-agnostic version of Experiment 2:
- Takes Experiment 1 generations JSONL as input.
- Uses canonicalization + structural metrics from core.ast_metrics.
- Uses execution utilities from core.execution_metrics.
- Computes execution accuracy and structural metrics (overall and on correct
  generations), plus the key "high_acc_low_struct" / "exec_correct_and_struct_diff"
  indicators discussed in the paper.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from core.ast_metrics import canonicalize_sql, compute_structure_stats
from core.execution_metrics import execute_sql, get_db_path


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"JSONL file not found: {path}")
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

    db_path = get_db_path(db_root=db_root, db_id=str(db_id), db_ext=db_ext)

    from core.execution_metrics import open_readonly_database

    with open_readonly_database(db_path) as conn:
        # Gold execution
        if gold_sql.strip():
            gold_ok, gold_result, gold_error = execute_sql(
                conn=conn,
                sql=gold_sql,
                max_rows=max_rows,
            )
        else:
            gold_ok, gold_result, gold_error = False, None, "Empty gold_sql"

        exec_ok_flags: List[bool] = []
        exec_error_messages: List[Optional[str]] = []
        is_correct_flags: List[bool] = []

        canonical_all: List[Optional[str]] = []

        for sql in sqls:
            ok, result, err = execute_sql(conn=conn, sql=sql, max_rows=max_rows)
            exec_ok_flags.append(ok)
            exec_error_messages.append(err)

            is_correct = False
            if ok and gold_ok and gold_result is not None and result is not None:
                is_correct = result == gold_result
            is_correct_flags.append(is_correct)

            canon, _parse_err = canonicalize_sql(sql)
            canonical_all.append(canon)

    struct_all = compute_structure_stats(canonical_all)

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

    num_exec_success = sum(1 for ok in exec_ok_flags if ok)
    num_exec_error = num_generations - num_exec_success
    num_correct = sum(1 for c in is_correct_flags if c)

    execution_accuracy_all = num_correct / num_generations if num_generations > 0 else 0.0
    execution_accuracy_on_exec_success = (
        num_correct / num_exec_success if num_exec_success > 0 else 0.0
    )

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
        "gold_exec_ok": gold_ok,
        "gold_exec_error": gold_error,
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
        # Key indicators
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
        description=(
            "StructEval Experiment 2 – execution vs structure metrics "
            "on Experiment 1 generations."
        )
    )
    parser.add_argument(
        "--generations_file",
        type=str,
        required=True,
        help="Experiment 1 generations JSONL file.",
    )
    parser.add_argument(
        "--db_root",
        type=str,
        required=True,
        help="Root directory containing Spider SQLite databases.",
    )
    parser.add_argument(
        "--db_ext",
        type=str,
        default=".sqlite",
        help="File extension for SQLite databases.",
    )
    parser.add_argument(
        "--max_rows",
        type=int,
        default=10_000,
        help="Maximum number of rows to fetch per query.",
    )
    parser.add_argument(
        "--max_examples",
        type=int,
        default=None,
        help="Optional maximum number of examples to evaluate (by example_index).",
    )
    parser.add_argument(
        "--stats_out",
        type=str,
        required=True,
        help="Output JSONL file for per-example stats.",
    )
    parser.add_argument(
        "--summary_out",
        type=str,
        required=True,
        help="Output JSON file for global summary.",
    )

    args = parser.parse_args()

    gen_path = Path(args.generations_file)
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
