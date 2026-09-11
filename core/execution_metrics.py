#!/usr/bin/env python3
"""
Execution-based metrics for StructEval (open-source subset).

This module provides utilities to:
- locate and open SQLite databases,
- execute SQL queries with a soft timeout,
- normalize query results for equality comparison.
"""

from __future__ import annotations

import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple


def get_db_path(db_root: Path, db_id: str, db_ext: str = ".sqlite") -> Path:
    """
    Spider-style DB path:
      {db_root}/{db_id}/{db_id}{db_ext}
    """
    if not db_id or Path(db_id).name != db_id or db_id in {".", ".."}:
        raise ValueError("db_id must be a single directory name")
    candidate = (db_root / db_id / f"{db_id}{db_ext}").resolve()
    if not candidate.is_relative_to(db_root.resolve()):
        raise ValueError("Database must remain within db_root")
    if not candidate.exists():
        raise FileNotFoundError(
            f"Database file not found for db_id={db_id}: {candidate} "
            "(adjust db_root / db_ext if your layout differs)."
        )
    return candidate


def normalize_result_for_compare(
    rows: Sequence[Sequence[Any]],
) -> Sequence[Tuple[Any, ...]]:
    """
    Normalize query results into a deterministic representation for equality.
    """
    tuples = [tuple(r) for r in rows]
    try:
        return tuple(sorted(tuples))
    except TypeError:
        return tuple(tuples)


def execute_sql(
    conn: sqlite3.Connection,
    sql: str,
    max_rows: int = 10_000,
    timeout_seconds: float = 10.0,
) -> Tuple[bool, Optional[Sequence[Tuple[Any, ...]]], Optional[str]]:
    """
    Execute a single SQL statement with a soft timeout.

    If execution exceeds `timeout_seconds`, the query is interrupted via
    sqlite3's progress handler and treated as a failure with a timeout error
    message.
    """
    start_time = time.time()
    timed_out = False

    def progress_handler() -> int:
        nonlocal timed_out
        if time.time() - start_time > timeout_seconds:
            timed_out = True
            return 1
        return 0

    conn.set_progress_handler(progress_handler, 1000)

    try:
        cur = conn.cursor()
        cur.execute(sql)
        rows = cur.fetchmany(max_rows + 1)
        if len(rows) > max_rows:
            return False, None, "ROW_LIMIT: result exceeds max_rows"
        normalized = normalize_result_for_compare(rows)
        if timed_out:
            return False, None, "TIMEOUT: query exceeded limit and was interrupted"
        return True, normalized, None
    except Exception as e:
        if timed_out:
            return False, None, "TIMEOUT: query exceeded limit and was interrupted"
        return False, None, str(e)
    finally:
        conn.set_progress_handler(None, 0)




@contextmanager
def open_readonly_database(path: Path):
    """Open a benchmark database without allowing generated SQL to write or attach."""
    conn = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)
    conn.execute("PRAGMA query_only = ON")
    allowed = {sqlite3.SQLITE_SELECT, sqlite3.SQLITE_READ, sqlite3.SQLITE_FUNCTION, sqlite3.SQLITE_RECURSIVE}
    def authorize(action, arg1, arg2, database, trigger):
        return sqlite3.SQLITE_OK if action in allowed else sqlite3.SQLITE_DENY
    conn.set_authorizer(authorize)
    try:
        yield conn
    finally:
        conn.close()
