#!/usr/bin/env python3
"""
Experiment 3 (StructEval) – error / failure mode analysis (open-source version).

This script:
- Loads baseline and compile-style per-example stats (from Experiment 2–style runs).
- Computes high-level summaries and execution × structure quadrants.
- Optionally:
  - writes a CSV template for manual error-type annotation;
  - aggregates a filled annotation CSV into error-type proportions.

The logic mirrors the original Experiment 3 analysis but removes any
project-specific paths and plotting dependencies are optional.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


@dataclass
class ExampleStats:
    example_index: int
    db_id: str
    question: str
    gold_sql: str
    num_generations: int
    num_exec_success: int
    num_exec_error: int
    num_exec_correct: int
    execution_accuracy_all: float
    execution_accuracy_on_exec_success: float
    all_distinct_structures: int
    all_majority_structure_ratio: float
    correct_distinct_structures: int
    correct_majority_structure_ratio: float
    ast_similarity_correct: float
    high_acc_low_struct: bool
    exec_correct_and_struct_diff: bool


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def parse_stats_record(rec: Dict[str, Any]) -> ExampleStats:
    return ExampleStats(
        example_index=int(rec.get("example_index", -1)),
        db_id=str(rec.get("db_id", "")),
        question=str(rec.get("question", "")),
        gold_sql=str(rec.get("gold_sql", "")),
        num_generations=int(rec.get("num_generations", 0)),
        num_exec_success=int(rec.get("num_exec_success", 0)),
        num_exec_error=int(rec.get("num_exec_error", 0)),
        num_exec_correct=int(rec.get("num_exec_correct", 0)),
        execution_accuracy_all=float(rec.get("execution_accuracy_all", 0.0)),
        execution_accuracy_on_exec_success=float(
            rec.get("execution_accuracy_on_exec_success", 0.0)
        ),
        all_distinct_structures=int(rec.get("all_distinct_structures", 0)),
        all_majority_structure_ratio=float(rec.get("all_majority_structure_ratio", 0.0)),
        correct_distinct_structures=int(rec.get("correct_distinct_structures", 0)),
        correct_majority_structure_ratio=float(
            rec.get("correct_majority_structure_ratio", 0.0)
        ),
        ast_similarity_correct=float(rec.get("ast_similarity_correct", 0.0)),
        high_acc_low_struct=bool(rec.get("high_acc_low_struct", False)),
        exec_correct_and_struct_diff=bool(rec.get("exec_correct_and_struct_diff", False)),
    )


def summarize_overall(stats: Iterable[ExampleStats]) -> Dict[str, float]:
    stats_list = list(stats)
    n = len(stats_list)
    if n == 0:
        return {}

    def avg(fn) -> float:
        return sum(fn(s) for s in stats_list) / n

    num_high_acc_low_struct = sum(1 for s in stats_list if s.high_acc_low_struct)
    num_exec_correct_examples = sum(1 for s in stats_list if s.num_exec_correct > 0)
    num_exec_correct_and_struct_diff = sum(
        1 for s in stats_list if s.exec_correct_and_struct_diff
    )

    return {
        "num_examples": float(n),
        "avg_execution_accuracy_all": avg(lambda s: s.execution_accuracy_all),
        "avg_execution_accuracy_on_exec_success": avg(
            lambda s: s.execution_accuracy_on_exec_success
        ),
        "avg_all_distinct_structures": avg(lambda s: float(s.all_distinct_structures)),
        "avg_correct_distinct_structures": avg(
            lambda s: float(s.correct_distinct_structures)
        ),
        "avg_all_majority_structure_ratio": avg(
            lambda s: s.all_majority_structure_ratio
        ),
        "avg_correct_majority_structure_ratio": avg(
            lambda s: s.correct_majority_structure_ratio
        ),
        "avg_ast_similarity_correct": avg(lambda s: s.ast_similarity_correct),
        "fraction_high_acc_low_struct": num_high_acc_low_struct / n,
        "num_exec_correct_examples": float(num_exec_correct_examples),
        "fraction_exec_correct_and_struct_diff_over_all": (
            num_exec_correct_and_struct_diff / n
        ),
        "fraction_exec_correct_and_struct_diff_over_correct": (
            num_exec_correct_and_struct_diff / num_exec_correct_examples
            if num_exec_correct_examples > 0
            else 0.0
        ),
    }


def classify_quadrant(
    s: ExampleStats,
    exec_threshold: float = 0.8,
    majority_threshold: float = 0.7,
) -> str:
    exec_correct = s.execution_accuracy_all >= exec_threshold
    struct_concentrated = s.correct_majority_structure_ratio >= majority_threshold

    if exec_correct and struct_concentrated:
        return "EC_SC"
    if exec_correct and not struct_concentrated:
        return "EC_SD"
    if not exec_correct and struct_concentrated:
        return "EW_SC"
    return "EW_SD"


def quadrant_distribution(
    stats: Iterable[ExampleStats],
    exec_threshold: float = 0.8,
    majority_threshold: float = 0.7,
) -> Dict[str, float]:
    stats_list = list(stats)
    n = len(stats_list)
    if n == 0:
        return {}
    counts: Dict[str, int] = {"EC_SC": 0, "EC_SD": 0, "EW_SC": 0, "EW_SD": 0}
    for s in stats_list:
        q = classify_quadrant(s, exec_threshold=exec_threshold, majority_threshold=majority_threshold)
        counts[q] += 1
    return {k: v / n for k, v in counts.items()}


def write_annotation_template_csv(
    out_path: Path,
    baseline_stats: List[ExampleStats],
    compile_stats: List[ExampleStats],
    max_rows: int = 200,
) -> None:
    """
    Prepare a CSV for manual error-type annotation.

    Columns include:
    - example_index, db_id, question, gold_sql
    - baseline_execution_accuracy_all, compile_execution_accuracy_all
    - baseline_ast_similarity_correct, compile_ast_similarity_correct
    - flags for exec_correct_and_struct_diff / high_acc_low_struct in both settings
    - empty columns for primary / secondary error-type labels
    """
    baseline_by_idx = {s.example_index: s for s in baseline_stats}
    compile_by_idx = {s.example_index: s for s in compile_stats}

    candidates: List[int] = []
    for idx in sorted(baseline_by_idx.keys()):
        b = baseline_by_idx[idx]
        c = compile_by_idx.get(idx)
        flagged = (
            b.exec_correct_and_struct_diff
            or b.high_acc_low_struct
            or (c is not None and (c.exec_correct_and_struct_diff or c.high_acc_low_struct))
        )
        if flagged:
            candidates.append(idx)
    if len(candidates) < max_rows:
        for idx in sorted(baseline_by_idx.keys()):
            if idx not in candidates:
                candidates.append(idx)
                if len(candidates) >= max_rows:
                    break
    else:
        candidates = candidates[:max_rows]

    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "example_index",
                "db_id",
                "question",
                "gold_sql",
                "baseline_execution_accuracy_all",
                "compile_execution_accuracy_all",
                "baseline_ast_similarity_correct",
                "compile_ast_similarity_correct",
                "baseline_exec_correct_and_struct_diff",
                "compile_exec_correct_and_struct_diff",
                "baseline_high_acc_low_struct",
                "compile_high_acc_low_struct",
                "primary_error_type",
                "secondary_error_types",
                "notes",
            ]
        )

        for idx in candidates:
            b = baseline_by_idx[idx]
            c = compile_by_idx.get(idx)
            writer.writerow(
                [
                    idx,
                    b.db_id,
                    b.question,
                    b.gold_sql,
                    f"{b.execution_accuracy_all:.3f}",
                    f"{c.execution_accuracy_all:.3f}" if c is not None else "",
                    f"{b.ast_similarity_correct:.3f}",
                    f"{c.ast_similarity_correct:.3f}" if c is not None else "",
                    str(b.exec_correct_and_struct_diff),
                    str(c.exec_correct_and_struct_diff) if c is not None else "",
                    str(b.high_acc_low_struct),
                    str(c.high_acc_low_struct) if c is not None else "",
                    "",
                    "",
                    "",
                ]
            )


def aggregate_annotation_csv(path: Path) -> Dict[str, float]:
    """
    Aggregate manually annotated error types from a CSV produced by
    write_annotation_template_csv.

    The annotator is expected to fill:
    - primary_error_type: one of {A,B,C,D,E,F}
    - secondary_error_types: optional comma-separated list of additional tags
    """
    if not path.exists():
        raise FileNotFoundError(path)

    total = 0
    counts: Dict[str, int] = {}
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            etype = (row.get("primary_error_type") or "").strip().upper()
            if not etype:
                continue
            if etype not in {"A", "B", "C", "D", "E", "F"}:
                continue
            total += 1
            counts[etype] = counts.get(etype, 0) + 1

    if total == 0:
        return {}

    return {k: v / total for k, v in sorted(counts.items())}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Experiment 3 error / failure mode analysis (open-source)."
    )
    parser.add_argument(
        "--baseline_stats",
        type=str,
        required=True,
        help="Baseline per-example stats JSONL (Experiment 2-style).",
    )
    parser.add_argument(
        "--compile_stats",
        type=str,
        required=True,
        help="Compile-style per-example stats JSONL.",
    )
    parser.add_argument(
        "--write_annotation_csv",
        type=str,
        default="",
        help="If set, write an annotation CSV template to this path.",
    )
    parser.add_argument(
        "--aggregate_annotation_csv",
        type=str,
        default="",
        help="If set, read an annotated CSV from this path and print error-type proportions.",
    )
    parser.add_argument(
        "--exec_threshold",
        type=float,
        default=0.8,
        help="Execution-accuracy threshold for quadrant classification.",
    )
    parser.add_argument(
        "--majority_threshold",
        type=float,
        default=0.7,
        help="Majority-structure-ratio threshold for quadrant classification.",
    )

    args = parser.parse_args()

    baseline_recs = load_jsonl(Path(args.baseline_stats))
    compile_recs = load_jsonl(Path(args.compile_stats))
    baseline_stats = [parse_stats_record(r) for r in baseline_recs]
    compile_stats = [parse_stats_record(r) for r in compile_recs]

    baseline_summary = summarize_overall(baseline_stats)
    compile_summary = summarize_overall(compile_stats)

    print("=== Overall baseline summary ===")
    print(json.dumps(baseline_summary, indent=2))
    print("=== Overall compile-style summary ===")
    print(json.dumps(compile_summary, indent=2))

    baseline_quads = quadrant_distribution(
        baseline_stats,
        exec_threshold=args.exec_threshold,
        majority_threshold=args.majority_threshold,
    )
    compile_quads = quadrant_distribution(
        compile_stats,
        exec_threshold=args.exec_threshold,
        majority_threshold=args.majority_threshold,
    )

    print("=== Quadrant distribution (baseline) ===")
    print(json.dumps(baseline_quads, indent=2))
    print("=== Quadrant distribution (compile-style) ===")
    print(json.dumps(compile_quads, indent=2))

    if args.write_annotation_csv:
        out_csv = Path(args.write_annotation_csv)
        write_annotation_template_csv(
            out_path=out_csv,
            baseline_stats=baseline_stats,
            compile_stats=compile_stats,
            max_rows=200,
        )
        print(f"Wrote annotation template CSV to {out_csv}")

    if args.aggregate_annotation_csv:
        in_csv = Path(args.aggregate_annotation_csv)
        proportions = aggregate_annotation_csv(in_csv)
        print("=== Error-type proportions from annotation CSV ===")
        print(json.dumps(proportions, indent=2))


if __name__ == "__main__":
    main()
