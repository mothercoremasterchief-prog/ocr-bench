# OCR Engine Benchmarks

Reproducible OCR quality comparisons using labeled synthetic documents.

## Minimum viable profile

The smallest validated screening profile uses 8 of the 10 labeled images and
three ground-truth assessments (CER, WER, and bag-of-words WER):

```bash
python3 benchmarks/harness.py --config benchmarks/config-expanded.yaml --profile minimum
python3 benchmarks/analyze_minimum.py --check
```

See [MINIMUM_VIABLE.md](MINIMUM_VIABLE.md) for the selection evidence, exact
scope, protocol, and expansion rule.

## Test Images

| Image | Type | Description |
|-------|------|-------------|
| `images/code_snippet.png` | Technical text | Code, punctuation, quotes, and dark background |
| `images/dense_paragraph.png` | Dense text | Long clean serif paragraph |
| `images/fine_print.png` | Small text | Dense terms and conditions at small font size |
| `images/printed_receipt.png` | Printed text | Clean receipt with store name, line items, prices, totals |
| `images/handwritten_letter.png` | Handwritten/scanned | Letter with slight rotation + blur simulating a scanned document |
| `images/noisy_invoice.png` | Noisy/mixed | Invoice with Gaussian noise artifacts overlaid on text |
| `images/noisy_meeting_notice.png` | Noisy prose | Rotated, blurred notice on a noisy background |
| `images/multi_column.png` | Complex layout | Newspaper-style two-column reading order |
| `images/spec_sheet.png` | Key/value | Mixed typography, labels, values, and units |
| `images/table_form.png` | Table | Four-column table with numeric and license fields |

Images are synthetically generated (Pillow) with known ground-truth text content, ensuring reproducible benchmarks.

## Scoring

Each engine's OCR output retains the legacy score below for compatibility and,
when ground truth is available, standard CER/WER/bWER assessments. New model
comparisons should use the ground-truth measures as primary evidence.

The legacy score uses 5 heuristics:

| Metric | Weight | What it measures |
|--------|--------|-----------------|
| `typo_rate` | 30% | Ratio of misspelled words (via wordfreq) |
| `gibberish_ratio` | 30% | Ratio of nonsense character sequences |
| `word_accuracy` | 15% | Detects merged words (>20 chars) and fragmentation |
| `single_letter_ratio` | 15% | Isolated single-character "words" |
| `char_separated_count` | 10% | Mid-word space insertion patterns ("O p e n") |

Composite score: 0–100 (100 = perfect quality).

## Legacy results (2026-02-24)

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
pip install -e ".[benchmark]"
python3 benchmarks/harness.py --config benchmarks/config-expanded.yaml --profile minimum
```

Requirements:
- All 6 engines running on localhost (ports 8100-8106)
- ocr-bench installed (`pip install -e ../` or `from ocr_bench import score`)
- benchmark dependencies (`pip install -e ".[benchmark]"`)

Output paths are controlled by the selected YAML endpoint configuration.

## Adding Images

Drop new `.png`/`.jpg` files in `images/` and add a same-basename UTF-8
transcript in `ground-truth/`. The unified harness discovers them automatically.
Profile membership is explicit; update the appropriate JSON file in `profiles/`
only after re-running the subset analysis.
