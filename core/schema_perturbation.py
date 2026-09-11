#!/usr/bin/env python3
"""
Schema perturbation utilities for StructEval Experiment 4B.

These functions implement table-order and column-order shuffles over a
StructEval-style schema object and are API-free.
"""

from __future__ import annotations

import copy
import random
from typing import Any, Dict


def schema_shuffle_table(schema: Dict[str, Any], seed: int = 0) -> Dict[str, Any]:
    """
    Return a deep copy of schema with tables list shuffled.
    Does not change columns or foreign_keys; only the order of tables.
    """
    out = copy.deepcopy(schema)
    tables = out.get("tables", [])
    if not tables:
        return out
    rng = random.Random(seed)
    rng.shuffle(tables)
    out["tables"] = tables
    return out


def schema_shuffle_column(schema: Dict[str, Any], seed: int = 0) -> Dict[str, Any]:
    """
    Return a deep copy of schema with each table's columns list shuffled.
    Does not change table order or foreign_keys.
    """
    out = copy.deepcopy(schema)
    tables = out.get("tables", [])
    rng = random.Random(seed)
    for t in tables:
        cols = t.get("columns", [])
        if cols:
            cols = list(cols)
            rng.shuffle(cols)
            t["columns"] = cols
    out["tables"] = tables
    return out


def get_schema_versions(schema: Dict[str, Any], example_index: int) -> Dict[str, Dict[str, Any]]:
    """
    Return three schema versions for one example:
    - schema_original: unchanged
    - schema_shuffle_table: table order shuffled (seed = example_index)
    - schema_shuffle_column: column order shuffled per table (seed = example_index + 1000)
    """
    return {
        "schema_original": copy.deepcopy(schema),
        "schema_shuffle_table": schema_shuffle_table(schema, seed=example_index),
        "schema_shuffle_column": schema_shuffle_column(schema, seed=example_index + 1000),
    }
