import json
import sqlite3
import tempfile
import unittest
from pathlib import Path
from core.ast_metrics import canonicalize_sql
from core.execution_metrics import open_readonly_database, execute_sql
from experiments.exp3.compile_style_generate_open import try_parse_and_compile
from scripts.prepare_spider import convert

class ReleaseTests(unittest.TestCase):
    def test_alias_and_conjunction_normalization(self):
        a, err = canonicalize_sql("SELECT x.id FROM items AS x WHERE x.id > 0 AND x.price < 20")
        b, err2 = canonicalize_sql("SELECT q.id FROM items AS q WHERE q.price < 20 AND q.id > 0")
        self.assertIsNone(err)
        self.assertIsNone(err2)
        self.assertEqual(a, b)

    def test_database_protection_and_overflow(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "demo.sqlite"
            conn = sqlite3.connect(path)
            conn.executescript("CREATE TABLE items (id INT); INSERT INTO items VALUES(1),(2);")
            conn.close()
            before = path.read_bytes()
            with open_readonly_database(path) as conn:
                self.assertFalse(execute_sql(conn, "DELETE FROM items")[0])
                self.assertFalse(execute_sql(conn, "ATTACH DATABASE ':memory:' AS other")[0])
                self.assertFalse(execute_sql(conn, "PRAGMA query_only=OFF")[0])
                self.assertFalse(execute_sql(conn, "SELECT * FROM items", max_rows=1)[0])
                ok, result, _ = execute_sql(conn, "SELECT COUNT(*) FROM items")
                self.assertTrue(ok)
                self.assertEqual(result, ((2,),))
            self.assertEqual(before, path.read_bytes())

    def test_compile_and_input_conversion(self):
        dev = [{"db_id":"demo", "question":"List ids", "query":"SELECT id FROM items"}]
        tables = [{"db_id":"demo", "table_names_original":["items"], "column_names_original":[[-1,"*"],[0,"id"]], "foreign_keys":[]}]
        examples = convert(dev, tables)
        plan = {"type":"query", "query":{"select":[{"expr":{"col":["items","id"]}}], "from":{"table":"items"}}}
        result = try_parse_and_compile(json.dumps(plan), examples[0]["schema"])
        self.assertTrue(result["compile_ok"])
        self.assertEqual(result["compiled_sql"], "SELECT items.id FROM items")
        self.assertFalse(try_parse_and_compile("not json", examples[0]["schema"])["json_parse_ok"])

if __name__ == "__main__":
    unittest.main()
