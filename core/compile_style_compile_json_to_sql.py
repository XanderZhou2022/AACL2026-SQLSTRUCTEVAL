#!/usr/bin/env python3
"""
Compile-style SQL generator: JSON-to-SQL compiler used in StructEval Experiment 3.

This module is API-free and can be used together with compile-style generation
scripts to turn a JSON query plan into executable SQL.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union


JsonObj = Dict[str, Any]


class CompileError(Exception):
    pass


def _lower(s: str) -> str:
    return s.lower()


@dataclass(frozen=True)
class SchemaIndex:
    tables: Dict[str, Dict[str, str]]

    @staticmethod
    def from_structeval_schema(schema: Mapping[str, Any]) -> "SchemaIndex":
        tables: Dict[str, Dict[str, str]] = {}
        for t in schema.get("tables", []) or []:
            tname = str(t.get("table_name", ""))
            if not tname:
                continue
            col_map: Dict[str, str] = {}
            for c in t.get("columns", []) or []:
                cs = str(c)
                if cs:
                    col_map[_lower(cs)] = cs
            tables[_lower(tname)] = col_map
        return SchemaIndex(tables=tables)

    def has_table(self, table: str) -> bool:
        return _lower(table) in self.tables

    def has_column(self, table: str, column: str) -> bool:
        t = self.tables.get(_lower(table))
        if not t:
            return False
        return _lower(column) in t


def parse_json_strict(text: str) -> JsonObj:
    s = (text or "").strip()
    if not s:
        raise CompileError("empty_output")
    if "```" in s:
        raise CompileError("markdown_fence_detected")
    try:
        obj = json.loads(s)
    except Exception as e:
        raise CompileError(f"json_parse_error: {e}") from e
    if not isinstance(obj, dict):
        raise CompileError("json_root_not_object")
    return obj


def _require_type(obj: Mapping[str, Any], key: str, typ: Union[type, Tuple[type, ...]]) -> Any:
    if key not in obj:
        raise CompileError(f"missing_field:{key}")
    val = obj[key]
    if not isinstance(val, typ):
        raise CompileError(f"type_error:{key}")
    return val


def _optional_type(obj: Mapping[str, Any], key: str, typ: Union[type, Tuple[type, ...]]) -> Any:
    if key not in obj:
        return None
    val = obj[key]
    if val is None:
        return None
    if not isinstance(val, typ):
        raise CompileError(f"type_error:{key}")
    return val


def _sql_ident(name: str) -> str:
    return name


def _sql_literal(val: Any) -> str:
    if isinstance(val, bool):
        return "1" if val else "0"
    if isinstance(val, (int, float)):
        return str(val)
    if isinstance(val, str):
        escaped = val.replace("'", "''")
        return f"'{escaped}'"
    raise CompileError("unsupported_literal_type")


Expr = Dict[str, Any]
Pred = Dict[str, Any]


def _parse_col_ref(col: Any) -> Tuple[str, str]:
    if isinstance(col, list) and len(col) == 2:
        return str(col[0]), str(col[1])
    if isinstance(col, str):
        if col.strip() == "*":
            return "", "*"
        if "." not in col:
            raise CompileError("col_ref_missing_dot")
        t, c = col.split(".", 1)
        return t.strip(), c.strip()
    raise CompileError("col_ref_type_error")


def _build_alias_map(query: Mapping[str, Any]) -> Dict[str, str]:
    alias_to_table: Dict[str, str] = {}

    frm = _require_type(query, "from", dict)
    base_table = str(frm.get("table", "")).strip()
    if not base_table:
        raise CompileError("from_table_empty")
    base_alias = str(frm.get("alias") or base_table).strip()
    alias_to_table[base_alias] = base_table

    joins = query.get("joins") or []
    if joins is None:
        joins = []
    if not isinstance(joins, list):
        raise CompileError("joins_type_error")
    for j in joins:
        if not isinstance(j, dict):
            raise CompileError("join_not_object")
        table = str(j.get("table", "")).strip()
        if not table:
            raise CompileError("join_table_empty")
        alias = str(j.get("alias") or table).strip()
        alias_to_table[alias] = table

    return alias_to_table


def _compile_expr(
    expr: Any,
    alias_to_table: Mapping[str, str],
    schema: SchemaIndex,
) -> str:
    if not isinstance(expr, dict):
        raise CompileError("expr_not_object")

    if expr.get("star") is True:
        return "*"

    if "col" in expr:
        t_or_a, c = _parse_col_ref(expr["col"])
        if c == "*":
            return "*"
        if not t_or_a:
            raise CompileError("col_ref_missing_table")
        table = alias_to_table.get(t_or_a, t_or_a)
        if not schema.has_table(table):
            raise CompileError(f"unknown_table:{table}")
        if not schema.has_column(table, c):
            raise CompileError(f"unknown_column:{table}.{c}")
        return f"{_sql_ident(t_or_a)}.{_sql_ident(c)}"

    if "val" in expr:
        return _sql_literal(expr["val"])

    if "op" in expr:
        op = str(expr.get("op", "")).strip()
        args = expr.get("args")
        if not op or not isinstance(args, list) or len(args) < 1:
            raise CompileError("op_expr_invalid")
        compiled_args = [
            _compile_expr(a, alias_to_table=alias_to_table, schema=schema)
            for a in args
        ]
        if len(compiled_args) == 1:
            return f"({op} {compiled_args[0]})"
        if len(compiled_args) == 2:
            return f"({compiled_args[0]} {op} {compiled_args[1]})"
        cur = compiled_args[0]
        for nxt in compiled_args[1:]:
            cur = f"({cur} {op} {nxt})"
        return cur

    if "func" in expr:
        func = str(expr.get("func", "")).strip()
        args = expr.get("args")
        if not func or not isinstance(args, list) or len(args) < 1:
            raise CompileError("func_expr_invalid")
        compiled_args = [
            _compile_expr(a, alias_to_table=alias_to_table, schema=schema)
            for a in args
        ]
        distinct = expr.get("distinct")
        distinct_sql = "DISTINCT " if distinct is True else ""
        return f"{func.upper()}({distinct_sql}{', '.join(compiled_args)})"

    if "scalar_subquery" in expr:
        sub_q = expr.get("scalar_subquery")
        if not isinstance(sub_q, dict):
            raise CompileError("scalar_subquery_not_object")
        sub_sql = _compile_query_object(sub_q, schema)
        return f"({sub_sql})"

    raise CompileError("expr_unsupported")


def _compile_pred(
    pred: Any,
    alias_to_table: Mapping[str, str],
    schema: SchemaIndex,
) -> str:
    if not isinstance(pred, dict):
        raise CompileError("pred_not_object")

    if "cmp" in pred:
        cmp_op = str(pred.get("cmp", "")).strip()
        left = pred.get("left")
        right = pred.get("right")
        if cmp_op not in {"=", "!=", "<", "<=", ">", ">="}:
            raise CompileError("cmp_op_invalid")
        if left is None or right is None:
            raise CompileError("cmp_missing_operands")
        lsql = _compile_expr(left, alias_to_table=alias_to_table, schema=schema)
        rsql = _compile_expr(right, alias_to_table=alias_to_table, schema=schema)
        return f"({lsql} {cmp_op} {rsql})"

    if "like" in pred:
        left = pred.get("like")
        pattern = pred.get("pattern")
        if left is None or pattern is None:
            raise CompileError("like_missing")
        lsql = _compile_expr(left, alias_to_table=alias_to_table, schema=schema)
        psql = _sql_literal(str(pattern))
        return f"({lsql} LIKE {psql})"

    if "between" in pred:
        target = pred.get("between")
        lo = pred.get("low")
        hi = pred.get("high")
        if target is None or lo is None or hi is None:
            raise CompileError("between_missing")
        tsql = _compile_expr(target, alias_to_table=alias_to_table, schema=schema)
        losql = _compile_expr(lo, alias_to_table=alias_to_table, schema=schema)
        hisql = _compile_expr(hi, alias_to_table=alias_to_table, schema=schema)
        return f"({tsql} BETWEEN {losql} AND {hisql})"

    if "in" in pred:
        target = pred.get("in")
        set_vals = pred.get("set")
        if target is None or not isinstance(set_vals, list) or len(set_vals) == 0:
            raise CompileError("in_missing")
        tsql = _compile_expr(target, alias_to_table=alias_to_table, schema=schema)
        compiled = [
            _compile_expr(v, alias_to_table=alias_to_table, schema=schema)
            for v in set_vals
        ]
        return f"({tsql} IN ({', '.join(compiled)}))"

    if "in_subquery" in pred:
        target = pred.get("in_subquery")
        sub_q = pred.get("subquery")
        if target is None or not isinstance(sub_q, dict):
            raise CompileError("in_subquery_missing")
        tsql = _compile_expr(target, alias_to_table=alias_to_table, schema=schema)
        sub_sql = _compile_query_object(sub_q, schema)
        return f"({tsql} IN ({sub_sql}))"

    if pred.get("is_null") is True:
        target = pred.get("target")
        if target is None:
            raise CompileError("is_null_missing_target")
        tsql = _compile_expr(target, alias_to_table=alias_to_table, schema=schema)
        return f"({tsql} IS NULL)"

    if pred.get("is_not_null") is True:
        target = pred.get("target")
        if target is None:
            raise CompileError("is_not_null_missing_target")
        tsql = _compile_expr(target, alias_to_table=alias_to_table, schema=schema)
        return f"({tsql} IS NOT NULL)"

    if "exists" in pred:
        sub_q = pred.get("exists")
        if not isinstance(sub_q, dict):
            raise CompileError("exists_not_object")
        sub_sql = _compile_query_object(sub_q, schema)
        return f"EXISTS ({sub_sql})"

    if "not_exists" in pred:
        sub_q = pred.get("not_exists")
        if not isinstance(sub_q, dict):
            raise CompileError("not_exists_not_object")
        sub_sql = _compile_query_object(sub_q, schema)
        return f"NOT EXISTS ({sub_sql})"

    raise CompileError("pred_unsupported")


def _compile_query_object(
    query: Mapping[str, Any],
    schema_obj_or_index: Union[Mapping[str, Any], SchemaIndex],
) -> str:
    if isinstance(schema_obj_or_index, SchemaIndex):
        schema = schema_obj_or_index
    else:
        schema = SchemaIndex.from_structeval_schema(schema_obj_or_index)
    alias_to_table = _build_alias_map(query)

    distinct = bool(query.get("distinct", False))
    select_items = _require_type(query, "select", list)
    if not select_items:
        raise CompileError("select_empty")

    select_sql_parts: List[str] = []
    for it in select_items:
        if not isinstance(it, dict):
            raise CompileError("select_item_not_object")
        expr = it.get("expr")
        if expr is None:
            raise CompileError("select_item_missing_expr")
        expr_sql = _compile_expr(expr, alias_to_table=alias_to_table, schema=schema)
        agg = it.get("agg")
        if agg:
            expr_sql = f"{str(agg).upper()}({expr_sql})"
        alias = it.get("alias")
        if alias:
            expr_sql = f"{expr_sql} AS {_sql_ident(str(alias))}"
        select_sql_parts.append(expr_sql)

    frm = _require_type(query, "from", dict)
    base_table = str(frm.get("table"))
    base_alias = str(frm.get("alias") or base_table)
    from_sql = f"FROM {_sql_ident(base_table)}"
    if base_alias and base_alias != base_table:
        from_sql += f" AS {_sql_ident(base_alias)}"

    join_sqls: List[str] = []
    for j in (query.get("joins") or []):
        if not isinstance(j, dict):
            raise CompileError("join_not_object")
        jtype = str(j.get("type", "inner")).lower().strip()
        join_kw = "JOIN" if jtype == "inner" else "LEFT JOIN"
        table = str(j.get("table"))
        alias = str(j.get("alias") or table)
        on_sql = _compile_pred(j["on"], alias_to_table=alias_to_table, schema=schema)
        join_clause = f"{join_kw} {_sql_ident(table)}"
        if alias and alias != table:
            join_clause += f" AS {_sql_ident(alias)}"
        join_clause += f" ON {on_sql}"
        join_sqls.append(join_clause)

    where_preds = query.get("where") or []
    where_sql = ""
    if where_preds:
        parts = [
            _compile_pred(p, alias_to_table=alias_to_table, schema=schema)
            for p in where_preds
        ]
        where_sql = "WHERE " + " AND ".join(parts)

    group_by_exprs = query.get("group_by") or []
    group_sql = ""
    if group_by_exprs:
        parts = [
            _compile_expr(e, alias_to_table=alias_to_table, schema=schema)
            for e in group_by_exprs
        ]
        group_sql = "GROUP BY " + ", ".join(parts)

    having_preds = query.get("having") or []
    having_sql = ""
    if having_preds:
        parts = [
            _compile_pred(p, alias_to_table=alias_to_table, schema=schema)
            for p in having_preds
        ]
        having_sql = "HAVING " + " AND ".join(parts)

    order_by = query.get("order_by") or []
    order_sql = ""
    if order_by:
        parts_sql: List[str] = []
        for ob in order_by:
            expr_sql = _compile_expr(ob["expr"], alias_to_table=alias_to_table, schema=schema)
            direction = str(ob.get("direction", "asc")).upper()
            if direction not in {"ASC", "DESC"}:
                direction = "ASC"
            parts_sql.append(f"{expr_sql} {direction}")
        order_sql = "ORDER BY " + ", ".join(parts_sql)

    limit = query.get("limit", None)
    limit_sql = f"LIMIT {int(limit)}" if isinstance(limit, int) else ""

    sql = "SELECT "
    if distinct:
        sql += "DISTINCT "
    sql += ", ".join(select_sql_parts) + " " + from_sql
    if join_sqls:
        sql += " " + " ".join(join_sqls)
    if where_sql:
        sql += " " + where_sql
    if group_sql:
        sql += " " + group_sql
    if having_sql:
        sql += " " + having_sql
    if order_sql:
        sql += " " + order_sql
    if limit_sql:
        sql += " " + limit_sql
    return sql.strip()


def compile_root_json_to_sql(root: Mapping[str, Any], schema_obj: Mapping[str, Any]) -> str:
    t = str(root.get("type", "query")).lower()
    if t == "query":
        query = _require_type(root, "query", dict)
        return _compile_query_object(query, schema_obj)
    if t == "set_op":
        set_op = _require_type(root, "set_op", dict)
        op = str(set_op.get("op", "")).lower()
        if op not in {"union", "union_all", "intersect", "except"}:
            raise CompileError("set_op_invalid")
        left_q = _require_type(set_op, "left", dict)
        right_q = _require_type(set_op, "right", dict)
        left_sql = _compile_query_object(left_q, schema_obj)
        right_sql = _compile_query_object(right_q, schema_obj)
        if op == "union":
            op_sql = "UNION"
        elif op == "union_all":
            op_sql = "UNION ALL"
        elif op == "intersect":
            op_sql = "INTERSECT"
        else:
            op_sql = "EXCEPT"
        return f"({left_sql}) {op_sql} ({right_sql})"
    raise CompileError("root_type_invalid")
