#!/usr/bin/env python3
"""
Experiment 4A (StructEval) – paraphrase robustness AST analysis (open-source).

Given a JSONL file where each record contains:
- example_index
- input_id (e.g., "original", "paraphrase_0", ...)
- sql_generations: list of SQL strings for that input,

this script:
- canonicalizes all SQLs using core.ast_metrics.canonicalize_sql;
- computes, per example:
  - distinct structures per input,
  - cross-paraphrase AST similarity,
  - overall distinct structure count,
  - perturbation sensitivity and number of inputs that differ from original.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

from core.ast_metrics import canonicalize_sql


def load_generations(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(
            f"Generations not found: {path}."
        )
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Experiment 4A – paraphrase AST robustness analysis (open-source)."
    )
    parser.add_argument(
        "--generations_file",
        type=str,
        required=True,
        help="JSONL file with fields example_index, input_id, sql_generations.",
    )
    parser.add_argument(
        "--stats_out",
        type=str,
        required=True,
        help="Output JSONL file for per-example AST stats.",
    )
    parser.add_argument(
        "--summary_out",
        type=str,
        required=True,
        help="Output JSON file for global summary.",
    )

    args = parser.parse_args()
    generations_path = Path(args.generations_file)
    records = load_generations(generations_path)

    print(f"Loaded {len(records)} generation records from {generations_path}.")

    by_example: Dict[int, List[Dict[str, Any]]] = {}
    for rec in records:
        ei = rec.get("example_index", 0)
        by_example.setdefault(int(ei), []).append(rec)

    per_example_stats: List[Dict[str, Any]] = []
    for example_index in sorted(by_example.keys()):
        inputs = by_example[example_index]
        input_canon_lists: Dict[str, List[Optional[str]]] = {}
        input_majority_canon: Dict[str, Optional[str]] = {}
        input_distinct: Dict[str, int] = {}
        all_canons_for_example: List[str] = []

        for rec in inputs:
            input_id = str(rec.get("input_id", ""))
            sqls = rec.get("sql_generations", []) or []
            canon_list: List[Optional[str]] = []
            for sql in sqls:
                c, _ = canonicalize_sql(sql)
                canon_list.append(c)
                if c is not None:
                    all_canons_for_example.append(c)
            input_canon_lists[input_id] = canon_list
            success = [c for c in canon_list if c is not None]
            if success:
                cnt = Counter(success)
                majority_canon = cnt.most_common(1)[0][0]
                input_majority_canon[input_id] = majority_canon
                input_distinct[input_id] = len(cnt)
            else:
                input_majority_canon[input_id] = None
                input_distinct[input_id] = 0

        majority_list = [
            input_majority_canon.get(iid) for iid in sorted(input_canon_lists.keys())
        ]
        majority_list = [m for m in majority_list if m is not None]
        if len(majority_list) >= 2:
            agreements = 0
            pairs = 0
            for i in range(len(majority_list)):
                for j in range(i + 1, len(majority_list)):
                    pairs += 1
                    if majority_list[i] == majority_list[j]:
                        agreements += 1
            cross_paraphrase_similarity = agreements / pairs if pairs else 0.0
        else:
            cross_paraphrase_similarity = 1.0 if len(majority_list) <= 1 else 0.0

        distinct_overall = len(set(c for c in all_canons_for_example if c))
        orig_majority = input_majority_canon.get("original")
        num_different_from_original = 0
        if orig_majority is not None:
            for iid, maj in input_majority_canon.items():
                if iid != "original" and maj is not None and maj != orig_majority:
                    num_different_from_original += 1
        num_other_inputs = max(1, len(input_majority_canon) - 1)
        perturbation_sensitivity = num_different_from_original / num_other_inputs

        stat = {
            "example_index": example_index,
            "num_inputs": len(inputs),
            "input_ids": sorted(input_canon_lists.keys()),
            "distinct_per_input": input_distinct,
            "cross_paraphrase_ast_similarity": round(cross_paraphrase_similarity, 4),
            "distinct_structure_count_overall": distinct_overall,
            "perturbation_sensitivity": round(perturbation_sensitivity, 4),
            "num_inputs_different_from_original": num_different_from_original,
        }
        per_example_stats.append(stat)

    stats_out = Path(args.stats_out)
    with stats_out.open("w", encoding="utf-8") as f:
        for s in per_example_stats:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(
        f"Wrote per-example AST stats to {stats_out} "
        f"({len(per_example_stats)} examples)."
    )

    summary_out = Path(args.summary_out)
    if per_example_stats:
        avg_sim = sum(
            s["cross_paraphrase_ast_similarity"] for s in per_example_stats
        ) / len(per_example_stats)
        avg_distinct = sum(
            s["distinct_structure_count_overall"] for s in per_example_stats
        ) / len(per_example_stats)
        avg_sens = sum(
            s["perturbation_sensitivity"] for s in per_example_stats
        ) / len(per_example_stats)
        num_sensitive = sum(
            1 for s in per_example_stats if s["perturbation_sensitivity"] > 0
        )
        summary = {
            "num_examples": len(per_example_stats),
            "avg_cross_paraphrase_ast_similarity": round(avg_sim, 4),
            "avg_distinct_structure_count_overall": round(avg_distinct, 2),
            "avg_perturbation_sensitivity": round(avg_sens, 4),
            "num_paraphrase_sensitive_examples": num_sensitive,
            "fraction_paraphrase_sensitive": round(
                num_sensitive / len(per_example_stats), 4
            ),
        }
        with summary_out.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"Wrote summary to {summary_out}.")
        print("Summary:", summary)
    print("Done.")


if __name__ == "__main__":
    main()
