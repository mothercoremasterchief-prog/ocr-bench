# Minimum viable OCR benchmark

## Decision

For this repository, the minimum viable benchmark is:

- **8 labeled images**;
- **3 ground-truth assessments** per image: character error rate (CER), word
  error rate (WER), and bag-of-words word error rate (bWER); and
- **1 deterministic quality run** per image, for 8 model calls total.

This is an English, single-page, plain-text **screening benchmark**. It is the
smallest subset that faithfully reproduces this repository's full ten-image
benchmark under the criteria below. It is not enough to certify a model for
every language, document domain, camera condition, or structured-output task.

Run it with any existing endpoint configuration:

```bash
python3 benchmarks/harness.py \
  --config benchmarks/config-expanded.yaml \
  --profile minimum
```

Use `--dry-run` to validate the profile without calling an endpoint. The
machine-readable definition is
[`profiles/minimum.json`](profiles/minimum.json).

## The eight images

| Image | Main stressors | Why it remains |
|---|---|---|
| `dense_paragraph` | Clean print, long text | Establishes the clean-text baseline and exposes omissions over a long sequence. |
| `fine_print` | Small glyphs, dense lines | Detects resolution and small-text failures hidden by normal-size print. |
| `handwritten_letter` | Handwriting-like text | Provides the corpus's only handwriting coverage. |
| `multi_column` | Two columns, reading order | Separates recognition from sequencing/layout failure. |
| `noisy_invoice` | Degradation, key/value fields, numbers | Tests scan robustness and business-critical values together. |
| `printed_receipt` | Narrow business layout, prices | Tests short lines and dense numeric/punctuation tokens. |
| `spec_sheet` | Mixed typography, key/value pairs | Tests label/value association and units. |
| `table_form` | Rows, columns, numbers | Provides the only explicit table stress test. |

`code_snippet` and `noisy_meeting_notice` are excluded. Within the declared
business-document scope, their incremental signal did not justify two more
calls: code is outside the scope, and degraded prose is already represented by
the noisy invoice.

## Why eight is the minimum

The selection script exhaustively evaluates every subset of the ten-image
corpus that retains handwriting, reading-order, table, degraded-scan,
small/symbol-text, and baseline-business-text coverage:

```bash
python3 benchmarks/analyze_minimum.py --check
```

Selection used the newest complete saved run for each of 24 engines. The May
2026 rebuild, containing 28 complete engines, was held out until validation. A
candidate had to satisfy **every** limit on both the legacy composite score and
ground-truth similarity, in both splits:

| Fidelity requirement | Limit |
|---|---:|
| Spearman rank correlation with all 10 images | at least 0.95 |
| Mean absolute engine-score error | at most 0.02 |
| 95th-percentile absolute error | at most 0.05 |
| Worst absolute error | at most 0.05 |

The declared eight-image profile produced:

| Legacy measure | Split | Spearman | Mean error | P95 error | Worst error |
|---|---|---:|---:|---:|---:|
| Composite score | selection (24 engines) | 0.9633 | 0.0032 | 0.0138 | 0.0209 |
| Composite score | held out (28 engines) | 0.9980 | 0.0035 | 0.0071 | 0.0116 |
| GT similarity | selection (24 engines) | 0.9680 | 0.0099 | 0.0407 | 0.0453 |
| GT similarity | held out (28 engines) | 0.9983 | 0.0101 | 0.0221 | 0.0452 |

No coverage-valid seven-image subset passed all limits. Three eight-image
subsets passed; this one was chosen because it is already represented by the
repository's four modular suites and has the clearest business-document scope.

This analysis proves fidelity to the existing ten synthetic pages, not to the
unknown population of documents a user may encounter. Failed or incomplete
historical runs are excluded from subset fitting, so endpoint reliability is a
separate hard gate during a new run.

## The three assessments

The harness reports the established edit-distance measures instead of relying
on one blended score:

1. **CER** measures character-level transcription fidelity. It catches small
   spelling, digit, punctuation, and hallucination errors.
2. **WER** measures word errors in sequence, so recognition and reading order
   both affect it.
3. **bWER** measures word errors while ignoring order. The reported
   `reading_order_error = max(0, WER - bWER)` isolates the extra sequential
   penalty; it is derived and is not a fourth assessment.

Text is normalized to Unicode NFC, case-folded, and collapsed to single spaces.
Punctuation and diacritics remain significant. Rates use reference length as
the denominator and can exceed 100% when a model over-generates. Both macro
(each image equal) and micro (each character or word equal) results are saved.

CER and WER follow the definitions in the
[OCR-D quality-assurance specification](https://ocr-d.de/en/spec/ocrd_eval.html).
The WER/bWER pair follows the page-level evaluation result of
[Vidal et al.](https://arxiv.org/abs/2301.05935), which uses their difference
to expose reading-order errors. The repository's dictionary heuristics and
legacy ground-truth similarity remain in output for compatibility, but are
secondary diagnostics rather than benchmark evidence.

## Why CER is the ranking metric

Fidelity and discrimination are different questions, and only the first was
being measured. A subset can reproduce the full ranking almost perfectly and
still be useless for ranking, if the metric ties most engines together.

Measured on the 8-page profile against the May 2026 rebuild (held-out engines,
recomputed from stored transcripts at no API cost):

| Metric | Distinct values | Largest tie | Can it order engines? |
|---|---|---|---|
| `score` (legacy composite) | 14 / 28 | **13** | No |
| `ground_truth_similarity` | 15 / 28 | **13** | No |
| `character_error_rate` | 25 / 30 | 3 | Yes |

Thirteen engines share `ground_truth_similarity` = 0.9567 exactly — among them
Gemini 2.5 Flash, Gemini 3 Pro, Gemini 3.1 Pro and two Qwen3-VL sizes. Their
relative order on that metric is a coin flip, even though its Spearman fidelity
to the ten-page ranking is 0.998. CER separates the same engines
(0.0540 / 0.0584 / 0.0643 / 0.0653 …).

`analyze_minimum.py --check` reports this as "Discriminative power", so a future
metric change is judged on whether it can actually rank, not only on whether it
reproduces the previous ranking. CER is recomputed from stored transcripts, so
any past run can be re-scored without spending anything.

## Run validity and reporting

A valid minimum run must meet all of these conditions:

- all eight images complete successfully; an endpoint error invalidates the
  run rather than silently improving the average by dropping a hard page;
- the same image bytes, prompt, model revision, and decoding settings are used
  for every compared model;
- generative endpoints use temperature 0;
- per-image CER, WER, bWER, and reading-order error are retained; and
- macro CER is the primary ranking measure, with micro CER, WER/bWER,
  worst-page results, error count, and latency shown beside it.

One call per page is sufficient for the deterministic quality screen. Latency
is only descriptive in that run. If latency is a decision criterion, use a
warm-up followed by at least three timed repetitions per page and report the
median and tail, because network and cold-start noise are not OCR quality.

## When eight images are not enough

The profile answers “is this model worth deeper testing on these capabilities?”
It does not estimate production accuracy with a defensible population-level
confidence interval. Before a procurement, release, or broad leaderboard claim:

1. replace or supplement the synthetic anchors with stratified, human-verified
   pages sampled from the target workload;
2. use **30 real pages as a practical floor**, with every important stratum
   represented, then continue sampling until the page-bootstrap 95% confidence
   interval for macro CER has the required precision;
3. compare models with paired page-level differences, not overlapping
   unpaired averages; and
4. add task-specific assessment when output includes coordinates, tables,
   formulas, or key-value JSON. Plain-text CER/WER cannot validate geometry or
   schema correctness.

Thirty is a floor, not a universal theorem. A narrow historical-newspaper study
found that 50 pages sufficed in its own setting, illustrating that sample needs
depend on domain variation ([Ströbel et al., LREC 2020](https://aclanthology.org/2020.lrec-1.436/)).
Add multilingual pages, real handwriting, rotations, low-resolution camera
captures, or formulas whenever those capabilities are in scope.
