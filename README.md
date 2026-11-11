## Quick Start

1. Install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Set API environment variables (example):

```bash
export LLM_BASE_URL="..."
export LLM_API_KEY="..."
export EMBEDDING_BASE_URL="..."
export EMBEDDING_API_KEY="..."
export RERANK_BASE_URL="..."
export RERANK_API_KEY="..."
```

## Build Knowledge Graph

Script: `testCode/Tools/kg_generate.py`

```bash
python testCode/Tools/kg_generate.py --input-path Input/dataset_for_hipporag2/sample_corpus.json
```

Parameter:

- `--input-path`: corpus JSON path to process.
- If you build KG for a different dataset, update `data_dir` in `src/config.py` first.

## Generate Answers + Evaluate

Script: `testCode/Evaluation/GenerateAnswers.py`

```bash
python testCode/Evaluation/GenerateAnswers.py \
  --method corprorag \
  --eval-count 50 \
  --concurrency 8 \
  --dataset-path Input/dataset_for_hipporag2/sample.json
```

Parameters:

- `--method`: `corprorag` or `SGN` (compatible aliases: `logic_graph`, `beam_search_kg`).
- `--eval-count`: number of samples to evaluate.
- `--concurrency`: async parallel task count.
- `--dataset-path`: QA dataset JSON path(directory ForQA).
- `--start-idx` (optional): start index in dataset.
- `--output-dir` (optional): output directory, default `Output`.
- If you evaluate a different dataset, also update `data_dir` in `src/config.py` to the corresponding database path.

Output files are written to:

- `Output/Ans/*.json`
- `Output/Eval/*.csv`
- `Output/Log/*.json`
