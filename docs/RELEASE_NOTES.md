# Public release notes

Prepared from the authors' supplied `StructEval-2435.zip` core archive for the AACL 2026 SQLStructEval project.

## Packaging and runtime changes

- Corrected the compile-style generator's import: the archive referenced a nonexistent `compile_style_v2_compile_json_to_sql` module.
- Fixed table-alias access and column identifier assignment for the pinned sqlglot API. The old access could treat an alias string as an AST node and turn otherwise valid queries into parse failures. This can change structural coverage relative to uncorrected code.
- Added package markers, dependency files, environment configuration example, a Spider input converter and an offline synthetic demo.
- Connected the four generation CLIs to an optional environment-configured OpenAI-compatible adapter. Original provider-neutral function interfaces remain available. Added compact JSON field guidance to the public compile-style prompt.
- Added generation metadata and exclusive output creation. Compilation failures keep their sample slots instead of disappearing from accuracy denominators.
- Mapped the unmodified schema to the robustness analyzer's `original` identifier.

## Database protection

- Evaluation opens benchmark databases read-only and rejects writes, PRAGMA changes and attached databases through an SQLite authorizer.
- Database IDs must remain within the configured root.
- Results exceeding the row cap fail explicitly rather than being judged from a truncated prefix. These cases may receive different execution scores than the archive's behavior.

## Validation and scope

The release was checked with Python 3.12, sqlglot 27.29.0 and the optional OpenAI SDK 2.54.0. Regression checks cover alias normalization, the JSON compiler, input preparation, database write protection and row limits. A synthetic offline example demonstrates identical execution results with two different structures.

No benchmark-wide rerun or live model request was performed. These operational fixes and public wrappers should not be interpreted as a reproduction of the original paper's runtime. Known inherited metric and compiler limits are recorded in [EXPERIMENTS.md](EXPERIMENTS.md).
