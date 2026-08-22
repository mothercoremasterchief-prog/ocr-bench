# ocr-bench

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-green.svg)](https://www.python.org/downloads/)
[![PyPI](https://img.shields.io/pypi/v/ocr-bench.svg)](https://pypi.org/project/ocr-bench/)

**Engine-agnostic OCR quality scoring, with or without ground truth.**

Most OCR benchmarks require labeled reference text. `ocr-bench` scores raw output quality using empirical heuristics that detect common OCR failure modes. Just pipe in text and get a score.

## Install

```bash
pip install ocr-bench
```

## Quick Start

### Python API

```python
from ocr_bench import score

result = score("The quick brown fox jumps over the lazy dog")
print(result)
# {'score': 96.5, 'typo_rate': 0.0, 'gibberish_ratio': 0.0,
#  'single_letter_ratio': 0.0, 'char_separated_count': 0, 'word_accuracy': 0.0}
```

### CLI

```bash
# Score text directly
ocr-bench score --text "Your OCR output here"

# Score from file
ocr-bench score --file output.txt

# JSON output (for piping)
ocr-bench score --file output.txt --json
```

### Ground-truth evaluation

When a verified transcript exists, use standard OCR error rates instead of
heuristics:

```bash
ocr-bench evaluate \
  --hypothesis ocr-output.txt \
  --reference ground-truth.txt \
  --json
```

This reports CER, sequential WER, bag-of-words WER, and a derived reading-order
error. See the [minimum viable benchmark](benchmarks/MINIMUM_VIABLE.md) for the
evidence-backed eight-image screening profile and its limitations.

### Batch scoring

```python
from ocr_bench import score

engines = {
    "tesseract": "Invoice Number: 12345\nDate: January 15, 2024",
    "easyocr":   "Inoice Numbr: 12345\nDte: Janury 15, 2024",
    "bad_engine": "Inv oic e Num b er : 1 2 3 4 5",
}

for name, text in engines.items():
    result = score(text)
    print(f"{name:12s} → {result['score']:5.1f}/100")
```

## Metrics

| Metric | What it detects | Good OCR | Bad OCR |
|--------|----------------|----------|---------|
| **Typo rate** | Misspelled words (frequency-based) | `"document processing"` → 0.0 | `"documnt procesing"` → 0.3 |
| **Gibberish ratio** | Random character soup | `"Invoice total"` → 0.0 | `"xkjhf qwrtpl"` → 0.9 |
| **Single letter ratio** | Lone characters from bad segmentation | `"Hello world"` → 0.0 | `"H e l l o w o r l d"` → 0.8 |
| **Char-separated count** | Spaces between every letter | `"Hello"` → 0 | `"H e l l o"` → 1 |
| **Word accuracy** | Word splits & merges | `"the document"` → 0.0 | `"thedocument"` or `"docu ment ation"` → high |

### Composite score

The **score** (0–100) is a weighted combination:

| Metric | Weight |
|--------|--------|
| Typo rate | 30% |
| Gibberish ratio | 30% |
| Word accuracy | 15% |
| Single letter ratio | 15% |
| Char-separated count | 10% |

**100** = clean text, **0** = unreadable garbage.

## Why support scoring without ground truth?

Ground truth benchmarks (WER, CER) need reference transcripts for every test image. That's expensive, limits what you can test, and doesn't scale.

These heuristics work on **any raw OCR output**:
- **Compare engines** — run the same image through 5 engines, score each
- **Monitor quality** — track scores over time, alert on regressions
- **Flag bad results** — auto-reject OCR output below a threshold
- **Benchmark at scale** — score thousands of documents without labeling

For model selection, ground truth is preferred when available. The no-reference
score is a scalable diagnostic, not a substitute for CER/WER on labeled pages.

## Examples: Good vs Bad OCR

**Good OCR** (score ~95):
```
Invoice Number: 12345
Date: January 15, 2024
Total Amount: $1,234.56
Thank you for your purchase.
```

**Mediocre OCR** (score ~70):
```
Inoice Numbr: 12345
Dte: Janury 15, 2024
Totl Amunt: $1,234.56
Thnk you for your purchse.
```

**Bad OCR** (score ~30):
```
I n o i c e N u m b e r : 1 2 3 4 5
D t e : J a n u r y 1 5 , 2 0 2 4
xkjhf qwrtpl bvnmcx
```

## Contributing

PRs welcome! To add a new metric:

1. Create `src/ocr_bench/metrics/your_metric.py` with a function returning `float` (0.0–1.0)
2. Export it from `metrics/__init__.py`
3. Add it to the scorer weights in `scorer.py`
4. Add tests in `tests/`
5. Update this README

**Ideas for new metrics:**
- Line break quality (unexpected mid-word breaks)
- Whitespace consistency (irregular spacing)
- Language detection confidence
- Table/column structure preservation
- Number/date format accuracy

### Development

```bash
git clone https://github.com/open-ocr/ocr-bench.git
cd ocr-bench
pip install -e ".[dev]"
pytest
```

## License

[Apache 2.0](LICENSE) — use it anywhere under the license terms.

---

Built by [OpenOCR](https://open-ocr.com). Star ⭐ if you find it useful.
