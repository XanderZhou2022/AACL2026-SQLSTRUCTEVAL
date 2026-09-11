"""Convert official Spider dev.json and tables.json to the public input schema."""
import argparse
import json
from pathlib import Path

def convert(dev, tables):
    schemas = {}
    for db in tables:
        names = db["table_names_original"]
        columns = db["column_names_original"]
        schemas[db["db_id"]] = {
            "tables": [{"table_name": name, "columns": [column for tid, column in columns if tid == i]} for i, name in enumerate(names)],
            "foreign_keys": [{"source_table": names[columns[a][0]], "source_column": columns[a][1], "target_table": names[columns[b][0]], "target_column": columns[b][1]} for a, b in db.get("foreign_keys", [])]
        }
    return [{"db_id": ex["db_id"], "question": ex["question"], "gold_sql": ex["query"], "schema": schemas[ex["db_id"]]} for ex in dev]

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dev_file", required=True)
    p.add_argument("--tables_file", required=True)
    p.add_argument("--output_file", required=True)
    args = p.parse_args()
    examples = convert(json.loads(Path(args.dev_file).read_text()), json.loads(Path(args.tables_file).read_text()))
    out = Path(args.output_file)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("x", encoding="utf-8") as f:
        json.dump(examples, f, indent=2, ensure_ascii=False)
    print(f"Prepared {len(examples)} examples.")

if __name__ == "__main__":
    main()
