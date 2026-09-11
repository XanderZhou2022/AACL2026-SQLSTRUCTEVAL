#!/usr/bin/env python3
"""
Core AST-based structural metrics for StructEval (open-source subset).

This module provides:
- normalize_sql_text: lightweight text-level SQL normalization
- canonicalize_sql: AST-level canonicalization using sqlglot
- shannon_entropy: Shannon entropy over integer counts
- compute_structure_stats: aggregate structural statistics over canonical SQLs

Dependencies:
- sqlglot  (pip install sqlglot)
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:
    import sqlglot
    from sqlglot import parse_one, exp as E
except ImportError as e:  # pragma: no cover - runtime check
    raise SystemExit(
        "sqlglot is required for AST analysis. "
        "Please install it with `pip install sqlglot`."
    ) from e


ALIAS_RE = re.compile(r"\s+as\s+[a-zA-Z_][a-zA-Z0-9_]*")


def normalize_sql_text(sql: str) -> str:
    """
    Normalize SQL at text level:
    - strip leading/trailing whitespace and trailing semicolon
    - convert newlines to spaces
    - remove simple column aliases in SELECT list (AS alias)
    - collapse multiple spaces
    - lowercase everything
    """
    s = sql.strip().rstrip(";")
    s = s.replace("\n", " ")
    s = ALIAS_RE.sub("", s)
    s = re.sub(r"\s+", " ", s)
    s = s.lower().strip()
    return s


def canonicalize_sql(sql: str, dialect: str = "sqlite") -> Tuple[Optional[str], Optional[str]]:
    """
    Parse SQL with sqlglot and return a canonical normalized SQL string.

    Canonicalization steps:
    - Parse into AST.
    - Normalize table aliases to a canonical sequence (t1, t2, ...).
    - Normalize commutative AND conditions by sorting conjuncts.
    - Render back to SQL with sqlglot and apply text-level normalization.

    Returns:
        (canonical_str, error_message)
    If parsing fails, canonical_str is None and error_message contains details.
    """

    def _normalize_table_aliases(expr: E.Expression) -> None:
        alias_map: Dict[str, str] = {}
        next_id = 1

        # First pass: remap table aliases
        for table in expr.find_all(E.Table):
            table_alias = table.args.get("alias")
            if table_alias and isinstance(table_alias.this, E.Identifier):
                old = table_alias.this.name
                if old not in alias_map:
                    alias_map[old] = f"t{next_id}"
                    next_id += 1
                new = alias_map[old]
                table_alias.set("this", E.Identifier(this=new))

        # Second pass: update column.table references
        if alias_map:
            for col in expr.find_all(E.Column):
                table_name = col.table
                if table_name in alias_map:
                    col.set("table", E.Identifier(this=alias_map[table_name]))

    def _flatten_and(node: E.Expression) -> List[E.Expression]:
        if isinstance(node, E.And):
            return _flatten_and(node.this) + _flatten_and(node.expression)
        return [node]

    def _normalize_and(expr: E.Expression) -> None:
        for and_node in list(expr.find_all(E.And)):
            parts = _flatten_and(and_node)
            parts_sorted = sorted(
                parts,
                key=lambda e: e.sql(dialect=dialect, pretty=False),
            )
            new_expr = parts_sorted[0]
            for part in parts_sorted[1:]:
                new_expr = E.And(this=new_expr, expression=part)
            and_node.replace(new_expr)

    try:
        expr = parse_one(sql, read=dialect)

        _normalize_table_aliases(expr)
        _normalize_and(expr)

        normalized = expr.sql(dialect=dialect, pretty=False)
        canon = normalize_sql_text(normalized)
        return canon, None
    except Exception as e:  # pragma: no cover - parsing failures are reported as errors
        return None, str(e)


def shannon_entropy(counts: Sequence[int]) -> float:
    """
    Compute Shannon entropy (base 2) for a list of counts.
    H = - sum_i p_i log2 p_i, where p_i = count_i / total.
    """
    total = sum(counts)
    if total <= 0:
        return 0.0
    entropy = 0.0
    for c in counts:
        if c <= 0:
            continue
        p = c / total
        entropy -= p * math.log2(p)
    return entropy


def compute_structure_stats(
    canonical_list: Sequence[Optional[str]],
) -> Dict[str, Any]:
    """
    Compute StructEval-style AST-level statistics from a list of canonical SQL strings.
    The list may contain None entries for parse failures.
    """
    freq: Counter[str] = Counter(c for c in canonical_list if c is not None)
    counts = list(freq.values())

    num_generations = len(canonical_list)
    num_parse_success = sum(counts)
    num_parse_fail = num_generations - num_parse_success

    if counts:
        entropy = shannon_entropy(counts)
        majority_ratio = max(counts) / num_parse_success if num_parse_success > 0 else 0.0
        if len(freq) > 1:
            max_entropy = math.log2(len(freq))
            entropy_normalized = entropy / max_entropy if max_entropy > 0 else 0.0
        else:
            entropy_normalized = 0.0
    else:
        entropy = 0.0
        entropy_normalized = 0.0
        majority_ratio = 0.0

    return {
        "num_generations": num_generations,
        "num_parse_success": num_parse_success,
        "num_parse_fail": num_parse_fail,
        "distinct_structures": len(freq),
        "entropy": entropy,
        "entropy_normalized": entropy_normalized,
        "majority_structure_ratio": majority_ratio,
    }
