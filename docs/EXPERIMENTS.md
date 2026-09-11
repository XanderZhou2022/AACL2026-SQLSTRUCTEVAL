# Running the experiments

Use Python 3.10+ and invoke modules from the repository root. The public release includes operational wrappers around the supplied research functions. No paid model requests were used during release verification.

## Inputs

The dev JSON is a list of objects:

```json
[
  {
    "db_id": "demo",
    "question": "Which item is cheapest?",
    "gold_sql": "SELECT id FROM items ORDER BY price LIMIT 1",
    "schema": {
      "tables": [{"table_name": "items", "columns": ["id", "price"]}],
      "foreign_keys": []
    }
  }
]
```

Use `scripts.prepare_spider` to convert the official Spider `dev.json` and `tables.json`. Database files follow `{db_root}/{db_id}/{db_id}.sqlite`. The preparation script does not download or redistribute a benchmark.

Generation records contain `example_index`, `db_id`, `question`, `gold_sql`, `schema`, and `sql_generations`. Compile-style records additionally contain `compile_records`; failed compilation occupies an empty SQL slot so it remains in the execution-accuracy denominator. API keys and endpoint URLs are never recorded by the adapter.

## Experiment 1: repeated SQL generation

Use the command in the root README. The supplied default selects the first 50 examples; set `--max_examples 1034` for the full Spider development set. This prefix selection is not a reconstruction of the paper's 200-question perturbation subset.

## Experiment 2: execution and structure

Run `experiments.exp2.run_experiment2_exec_vs_structure` on the generation JSONL. The script reports per-question counts, entropy, majority, correct-subset metrics, and thresholded indicators, followed by averages across questions. `scripts.compute_structeval_metrics --help` provides a combined runner that can merge a dev file and saved generations.

Structural counts ignore failed parses. Empty subsets contribute zero to counts, majority and entropy. `ast_similarity_correct` is the same value as `correct_majority_structure_ratio`. Conditional execution accuracy divides correct outputs by error-free executions within each question and then averages across questions. Questions with missing databases or other analysis errors are recorded with an `error` field; inspect these records before interpreting aggregates, because the inherited summarizer counts absent numeric fields as zero.

## Experiment 3: compile-style generation

```bash
python -m experiments.exp3.compile_style_generate_open \
  --dev_file data/dev_structeval.json \
  --output_file outputs/compile.jsonl \
  --model_name gpt-5-mini-2025-08-07 \
  --indices 0-4 --num_generations 10

python -m experiments.exp2.run_experiment2_exec_vs_structure \
  --generations_file outputs/compile.jsonl \
  --db_root data/spider/database \
  --stats_out outputs/compile_stats.jsonl \
  --summary_out outputs/compile_summary.json

python -m experiments.exp3.error_analysis_experiment3_open \
  --baseline_stats outputs/direct_stats.jsonl \
  --compile_stats outputs/compile_stats.jsonl \
  --write_annotation_csv outputs/error_annotation.csv
```

Omit `--indices` to use all dev entries. The compiler implements a limited JSON grammar, not arbitrary SQL. See `core/compile_style_compile_json_to_sql.py` for supported expressions and predicates. No repair loop, full SQL validator or DIN-SQL implementation is bundled. The public generation prompt includes a compact field guide to make the released compiler callable; it is not claimed to reconstruct unpublished original prompts.

## Experiment 4A: paraphrase robustness

Create `data/paraphrases.jsonl` locally, with one record per selected question:

```json
{"example_index": 0, "paraphrases": ["Which item has the lowest price?"]}
```

Paraphrases must be checked for preservation of requested attributes, filters, aggregation, order, limits and schema meaning. The archive does not include the original manually checked paraphrases or their generator.

```bash
python -m experiments.exp4.exp4A_generate_sql_paraphrases_open \
  --dev_file data/dev_structeval.json \
  --paraphrases_file data/paraphrases.jsonl \
  --output_file outputs/paraphrase.jsonl \
  --model_name gpt-5-mini-2025-08-07 --num_generations 10

python -m experiments.exp4.exp4_paraphrase_ast_analysis \
  --generations_file outputs/paraphrase.jsonl \
  --stats_out outputs/paraphrase_stats.jsonl \
  --summary_out outputs/paraphrase_summary.json
```

## Experiment 4B: schema presentation

Create `data/indices.json` with selected dev indices, such as `[0, 1, 2, 3, 4]`.

```bash
python -m experiments.exp4.exp4B_generate_sql_schema_perturb_open \
  --dev_file data/dev_structeval.json \
  --indices_file data/indices.json \
  --output_file outputs/schema.jsonl \
  --model_name gpt-5-mini-2025-08-07 --num_generations 10

python -m experiments.exp4.exp4_paraphrase_ast_analysis \
  --generations_file outputs/schema.jsonl \
  --stats_out outputs/schema_stats.jsonl \
  --summary_out outputs/schema_summary.json
```

The shared analyzer accepts both variant types; output fields retain the historical `paraphrase` names. The schema generator sets `input_id=original` for the unmodified schema, so sensitivity has the correct baseline. Table shuffling uses the example index as its seed; column shuffling uses that index plus 1000.

The inherited robustness analyzer excludes unavailable majorities from pairwise agreement and assigns agreement 1 when fewer than two valid majorities remain. Ties select the first-seen structure. Sensitivity skips invalid variant majorities but uses all variants in its denominator. Inspect `distinct_per_input` and parse coverage before interpreting high agreement. These conventions are retained rather than silently replacing the research metric.

## Reproducibility scope

The source archive did not contain model response dumps, original dependency pins, provider-specific clients, selected subset indices, DIN-SQL, or BIRD / Spider-Syn / Dr.Spider scope-evaluation scripts. This release cannot certify exact reproduction of paper tables. The pinned `sqlglot` version is a release validation environment, not a recovered original version.

The inherited canonicalizer lowercases literal contents and normalizes aliases without full nested-scope analysis. The executor sorts comparable result rows, so it does not preserve order-sensitive correctness; mixed-type rows retain returned order if Python cannot sort them. The compiler does not cover every SQLite construct. Use the official benchmark evaluation when its semantics are required, and report any evaluator changes separately.

API adapter reference: [OpenAI Chat Completions documentation](https://developers.openai.com/api/reference/resources/chat). Credentials come only from the environment. Generation sends question/schema prompts to the configured provider and incurs that provider's normal usage charges when invoked.
