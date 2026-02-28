# Benchmark Suites

Each suite is an independent benchmark focused on a specific document category.

## Structure

```
suites/
├── receipts/          # Store receipts, POS printouts
├── handwriting/       # Handwritten notes, letters
├── tables/            # Structured tables, forms, spec sheets
├── dense-text/        # Legal text, fine print, multi-column
└── multilingual/      # Non-English and mixed-language documents
```

Each suite contains:
- `images/` — Test images (PNG, public domain / CC0)
- `ground-truth/` — Expected text output (`.txt`, UTF-8)

## Composite Scoring

Individual suite scores are combined into a composite score:

```
composite = Σ (suite_score × weight) / Σ weights
```

Default weights (configurable):
| Suite | Weight | Rationale |
|-------|--------|-----------|
| receipts | 1.0 | Common OCR use case |
| handwriting | 1.0 | Differentiates engines |
| tables | 1.0 | Structured extraction |
| dense-text | 1.0 | Baseline readability |
| multilingual | 0.5 | Important but niche |

## Adding a New Suite

1. Create `suites/<name>/images/` and `suites/<name>/ground-truth/`
2. Add test images and matching ground truth files (same basename)
3. Run `python benchmarks/run_benchmark.py --suite <name>`
4. Results appear in `results/<name>/`

## Scoring Metrics

- **Character Error Rate (CER)** — Levenshtein distance / reference length
- **Word Accuracy** — Correct words / total words
- **Latency** — Engine response time in ms
- **Composite** — Weighted average across suites
