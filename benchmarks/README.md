# OCR Engine Benchmarks

Real quality scores from ocr-bench v0.2.0 against all 6 OpenOCR engines.

## Test Images

| Image | Type | Description |
|-------|------|-------------|
| `images/printed_receipt.png` | Printed text | Clean receipt with store name, line items, prices, totals |
| `images/handwritten_letter.png` | Handwritten/scanned | Letter with slight rotation + blur simulating a scanned document |
| `images/noisy_invoice.png` | Noisy/mixed | Invoice with Gaussian noise artifacts overlaid on text |

Images are synthetically generated (Pillow) with known ground-truth text content, ensuring reproducible benchmarks.

## Scoring

Each engine's OCR output is scored by ocr-bench on 5 metrics:

| Metric | Weight | What it measures |
|--------|--------|-----------------|
| `typo_rate` | 30% | Ratio of misspelled words (via wordfreq) |
| `gibberish_ratio` | 30% | Ratio of nonsense character sequences |
| `word_accuracy` | 15% | Detects merged words (>20 chars) and fragmentation |
| `single_letter_ratio` | 15% | Isolated single-character "words" |
| `char_separated_count` | 10% | Mid-word space insertion patterns ("O p e n") |

Composite score: 0–100 (100 = perfect quality).

## Results (2026-02-24)

| Rank | Engine | Avg Score | Avg Latency |
|------|--------|-----------|-------------|
| 1 | Tesseract | 97.3 | 264ms |
| 2 | Surya | 97.1 | 2328ms |
| 3 | OCRmyPDF | 96.9 | 671ms |
| 4 | docTR | 96.2 | 1527ms |
| 5 | EasyOCR | 95.1 | 797ms |
| 6 | RapidOCR | 84.1 | 806ms |

RapidOCR scores lower due to word-merging (missing spaces between words). All other engines score 93+ on every image.

## Re-running

```bash
cd packages/ocr-bench/benchmarks
python3 run_benchmark.py
```

Requirements:
- All 6 engines running on localhost (ports 8100-8106)
- ocr-bench installed (`pip install -e ../` or `from ocr_bench import score`)
- `requests` package

Output: overwrites `engine-scores.json` with fresh results.

## Adding Images

Drop new `.png`/`.jpg` files in `images/`, then add them to the `IMAGES` dict in `run_benchmark.py`. Re-run to include in scores.
