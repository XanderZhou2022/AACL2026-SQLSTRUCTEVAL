"""Synthetic illustration, no API access or benchmark download required."""
import json
import sqlite3
import tempfile
from pathlib import Path
from core.compile_style_compile_json_to_sql import compile_root_json_to_sql
from experiments.exp2.run_experiment2_exec_vs_structure import analyze_single_example

def main():
    schema = {"tables": [{"table_name": "items", "columns": ["id", "price"]}], "foreign_keys": []}
    plan = {"type": "query", "query": {"select": [{"expr": {"col": ["items", "id"]}}], "from": {"table": "items"}, "order_by": [{"expr": {"col": ["items", "price"]}, "direction": "asc"}], "limit": 1}}
    compiled = compile_root_json_to_sql(plan, schema)
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "demo").mkdir()
        conn = sqlite3.connect(root / "demo/demo.sqlite")
        conn.executescript("CREATE TABLE items (id INTEGER PRIMARY KEY, price INTEGER); INSERT INTO items VALUES (1,10),(2,20);")
        conn.close()
        record = {"example_index": 0, "db_id": "demo", "question": "Which item has the lowest price?", "gold_sql": compiled,
                  "sql_generations": [compiled, "SELECT items.id FROM items WHERE items.price = (SELECT MIN(price) FROM items)"]}
        stats = analyze_single_example(record, root, ".sqlite", 10000)
        print(json.dumps({"compiled_sql": compiled, "execution_accuracy": stats["execution_accuracy_all"], "distinct_correct_structures": stats["correct_distinct_structures"], "majority_correct": stats["ast_similarity_correct"]}, indent=2))

if __name__ == "__main__":
    main()
