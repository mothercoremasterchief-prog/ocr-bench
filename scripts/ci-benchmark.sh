#!/usr/bin/env bash
# CI benchmark — runs ocr-bench against all 6 engines and checks for regressions.
# Fails if any engine score drops >5 points from baseline.
#
# Usage: ./ci-benchmark.sh [--baseline path/to/engine-scores.json]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BENCH_DIR="$SCRIPT_DIR/../benchmarks"
BASELINE="${1:-$BENCH_DIR/engine-scores.json}"
THRESHOLD=5

if [ ! -f "$BASELINE" ]; then
  echo "❌ Baseline not found: $BASELINE"
  echo "   Run benchmarks/run_benchmark.py first to create a baseline."
  exit 1
fi

echo "🔬 Running ocr-bench CI benchmark..."
echo "   Baseline: $BASELINE"
echo "   Regression threshold: ${THRESHOLD} points"
echo ""

# Run benchmark, capture new results to temp file
TMPFILE=$(mktemp /tmp/ocr-bench-ci-XXXXXX.json)
trap "rm -f $TMPFILE" EXIT

cd "$BENCH_DIR"
python3 -c "
import json, sys, os
sys.path.insert(0, os.path.join('$BENCH_DIR', '..'))

# Run the benchmark logic inline (avoid overwriting baseline)
import base64, time, requests
from ocr_bench import score as score_text

ENGINES = {
    'easyocr':   {'port': 8100},
    'tesseract': {'port': 8101},
    'doctr':     {'port': 8102},
    'surya':     {'port': 8103},
    'ocrmypdf':  {'port': 8105},
    'rapidocr':  {'port': 8106},
}

IMAGES = {
    'printed_receipt':    'images/printed_receipt.png',
    'handwritten_letter': 'images/handwritten_letter.png',
    'noisy_invoice':      'images/noisy_invoice.png',
}

results = {}
for img_name, img_path in IMAGES.items():
    with open(img_path, 'rb') as f:
        image_b64 = base64.b64encode(f.read()).decode()
    for eng_name, eng_cfg in ENGINES.items():
        try:
            r = requests.post(f'http://localhost:{eng_cfg[\"port\"]}/v1/ocr',
                json={'input': {'type': 'base64', 'data_base64': image_b64}}, timeout=120)
            if r.status_code != 200:
                score = 0
            else:
                data = r.json()
                text = data.get('text', data.get('result', ''))
                if isinstance(text, list):
                    text = '\n'.join(str(t) for t in text)
                sr = score_text(str(text))
                score = sr['score']
        except Exception:
            score = 0
        results.setdefault(eng_name, {})[img_name] = score

# Compute averages
for eng in results:
    scores = [v for v in results[eng].values() if v > 0]
    results[eng]['average_score'] = round(sum(scores) / len(scores), 1) if scores else 0

with open('$TMPFILE', 'w') as f:
    json.dump(results, f, indent=2)
" 2>&1

if [ ! -s "$TMPFILE" ]; then
  echo "❌ Benchmark produced no output"
  exit 1
fi

# Compare against baseline
python3 -c "
import json, sys

with open('$BASELINE') as f:
    baseline = json.load(f)
with open('$TMPFILE') as f:
    current = json.load(f)

threshold = $THRESHOLD
failures = []

print(f\"{'Engine':12s} {'Baseline':>10s} {'Current':>10s} {'Delta':>8s}  Status\")
print('-' * 52)

for eng in sorted(baseline.keys()):
    base_avg = baseline[eng].get('average_score', 0)
    curr_avg = current.get(eng, {}).get('average_score', 0)
    delta = curr_avg - base_avg
    status = '✅' if delta >= -threshold else '❌ REGRESSION'
    if delta < -threshold:
        failures.append(eng)
    print(f\"{eng:12s} {base_avg:10.1f} {curr_avg:10.1f} {delta:+8.1f}  {status}\")

print()
if failures:
    print(f'❌ FAILED — {len(failures)} engine(s) regressed >{threshold} points: {', '.join(failures)}')
    sys.exit(1)
else:
    print(f'✅ PASSED — all engines within {threshold}-point threshold')
    sys.exit(0)
"
