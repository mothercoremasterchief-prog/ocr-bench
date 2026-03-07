#!/usr/bin/env python3
"""NVIDIA NIM OCR Benchmark - with retries and longer timeouts"""

import base64, json, os, time, urllib.request, urllib.error, sys
from datetime import datetime

API_KEY = os.environ["NVIDIA_NIM_API_KEY"]
API_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
PROMPT = "Extract all text from this image exactly as it appears. Output only the extracted text, nothing else."

MODELS = [
    "nvidia/nemotron-nano-12b-v2-vl",
    "meta/llama-3.2-90b-vision-instruct",
    "microsoft/phi-4-multimodal-instruct",
    "nvidia/llama-3.1-nemotron-nano-vl-8b-v1",
]

DOCUMENTS = [
    ("receipts", "printed_receipt"),
    ("receipts", "noisy_invoice"),
    ("tables", "spec_sheet"),
    ("tables", "table_form"),
    ("dense-text", "dense_paragraph"),
    ("dense-text", "fine_print"),
    ("dense-text", "multi_column"),
    ("handwriting", "handwritten_letter"),
]

BASE = "/home/ben/.openclaw/workspace/projects/ocr-bench/suites"
TIMEOUT = 300  # 5 minutes per request
MAX_RETRIES = 2
RETRY_DELAY = 10


def edit_distance(s1, s2):
    m, n = len(s1), len(s2)
    if m == 0: return n
    if n == 0: return m
    prev = list(range(n + 1))
    for i in range(1, m + 1):
        curr = [i] + [0] * n
        for j in range(1, n + 1):
            if s1[i-1] == s2[j-1]:
                curr[j] = prev[j-1]
            else:
                curr[j] = 1 + min(prev[j], curr[j-1], prev[j-1])
        prev = curr
    return prev[n]


def cer(extracted, ground_truth):
    gt = ground_truth.strip()
    ex = extracted.strip()
    if len(gt) == 0:
        return 0.0
    return edit_distance(ex, gt) / len(gt)


def call_nim(model, image_b64, retries=MAX_RETRIES):
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{image_b64}"
                        }
                    },
                    {
                        "type": "text",
                        "text": PROMPT
                    }
                ]
            }
        ],
        "max_tokens": 2048,
        "temperature": 0,
        "stream": False,
    }
    data = json.dumps(payload).encode("utf-8")

    for attempt in range(retries + 1):
        req = urllib.request.Request(
            API_URL,
            data=data,
            headers={
                "Authorization": f"Bearer {API_KEY}",
                "Content-Type": "application/json",
                "Accept": "application/json"
            },
            method="POST"
        )
        t0 = time.time()
        try:
            with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
                body = json.loads(resp.read().decode("utf-8"))
            latency_ms = int((time.time() - t0) * 1000)
            text = body["choices"][0]["message"]["content"]
            return text, latency_ms, None
        except urllib.error.HTTPError as e:
            latency_ms = int((time.time() - t0) * 1000)
            err_body = e.read().decode("utf-8", errors="replace")
            error = f"HTTP {e.code}: {err_body[:300]}"
            if e.code == 429 or e.code >= 500:
                if attempt < retries:
                    wait = RETRY_DELAY * (attempt + 1)
                    print(f"[retry {attempt+1}/{retries} in {wait}s] ", end="", flush=True)
                    time.sleep(wait)
                    continue
            return None, latency_ms, error
        except Exception as ex:
            latency_ms = int((time.time() - t0) * 1000)
            error = str(ex)
            if attempt < retries:
                wait = RETRY_DELAY * (attempt + 1)
                print(f"[retry {attempt+1}/{retries} in {wait}s: {error[:50]}] ", end="", flush=True)
                time.sleep(wait)
                continue
            return None, latency_ms, error


def main():
    sys.stdout.reconfigure(line_buffering=True)
    results = []

    # Pre-load images and ground truth
    docs = []
    for suite, name in DOCUMENTS:
        img_path = f"{BASE}/{suite}/images/{name}.png"
        gt_path = f"{BASE}/{suite}/ground-truth/{name}.txt"
        with open(img_path, "rb") as f:
            img_b64 = base64.b64encode(f.read()).decode("utf-8")
        with open(gt_path, "r") as f:
            gt_text = f.read()
        docs.append((suite, name, img_b64, gt_text))
        print(f"  Loaded {suite}/{name} (GT: {len(gt_text)} chars, img: {len(img_b64)} b64 chars)")

    print(f"\nRunning benchmark: {len(MODELS)} models × {len(docs)} documents = {len(MODELS)*len(docs)} calls")
    print(f"Timeout: {TIMEOUT}s, Retries: {MAX_RETRIES}\n")

    model_stats = {}

    for mi, model in enumerate(MODELS):
        print(f"\n{'='*60}")
        print(f"[{mi+1}/{len(MODELS)}] Model: {model}")
        print('='*60)
        model_stats[model] = {"cers": [], "latencies": [], "errors": 0}

        for di, (suite, name, img_b64, gt_text) in enumerate(docs):
            doc_id = f"{suite}/{name}"
            print(f"  [{di+1}/{len(docs)}] {doc_id} ... ", end="", flush=True)

            extracted, latency_ms, error = call_nim(model, img_b64)

            if error:
                print(f"ERROR ({latency_ms}ms): {error[:120]}")
                result = {
                    "model": model,
                    "suite": suite,
                    "document": name,
                    "cer": None,
                    "cer_pct": None,
                    "latency_ms": latency_ms,
                    "error": error,
                    "extracted_preview": None,
                }
                model_stats[model]["errors"] += 1
            else:
                c = cer(extracted, gt_text)
                model_stats[model]["cers"].append(c)
                model_stats[model]["latencies"].append(latency_ms)
                print(f"CER={c*100:.1f}% latency={latency_ms}ms ({len(extracted)} chars)")
                result = {
                    "model": model,
                    "suite": suite,
                    "document": name,
                    "cer": round(c, 6),
                    "cer_pct": round(c * 100, 2),
                    "latency_ms": latency_ms,
                    "error": None,
                    "extracted_preview": extracted[:200] if extracted else None,
                }
            results.append(result)

        # Small delay between models to avoid rate limits
        if mi < len(MODELS) - 1:
            print(f"\n  Pausing 5s before next model...")
            time.sleep(5)

    # Compute per-model averages
    summary = []
    for model in MODELS:
        cers = model_stats[model]["cers"]
        lats = model_stats[model]["latencies"]
        avg_cer = sum(cers) / len(cers) if cers else None
        avg_lat = sum(lats) / len(lats) if lats else None
        errors = model_stats[model]["errors"]
        summary.append({
            "model": model,
            "avg_cer": round(avg_cer, 6) if avg_cer is not None else None,
            "avg_cer_pct": round(avg_cer * 100, 2) if avg_cer is not None else None,
            "avg_latency_ms": int(avg_lat) if avg_lat is not None else None,
            "errors": errors,
            "docs_scored": len(cers),
        })

    # Sort by avg CER (ascending = better)
    summary_sorted = sorted(summary, key=lambda x: (x["avg_cer"] is None, x["avg_cer"] or 999))
    for rank, s in enumerate(summary_sorted, 1):
        s["rank"] = rank

    output = {
        "benchmark": "NVIDIA NIM OCR Benchmark",
        "date": "2026-03-07",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "models_tested": MODELS,
        "documents_tested": [f"{s}/{n}" for s, n in DOCUMENTS],
        "summary": summary_sorted,
        "results": results,
    }

    out_path = "/home/ben/.openclaw/workspace/projects/ocr-bench/results/nim_benchmark_2026-03-07.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n✅ Results saved to {out_path}")

    # Print markdown table
    print("\n\n## NIM OCR Benchmark Results (2026-03-07)\n")
    print("| Model | Document | CER% | Latency(ms) |")
    print("|-------|----------|------|-------------|")
    for r in results:
        model_short = r["model"].split("/")[-1]
        doc = f"{r['suite']}/{r['document']}"
        cer_str = f"{r['cer_pct']:.1f}" if r.get("cer_pct") is not None else "ERR"
        lat_str = str(r["latency_ms"])
        print(f"| {model_short} | {doc} | {cer_str} | {lat_str} |")

    print("\n### Average CER per Model (ranked, lower = better)\n")
    print("| Rank | Model | Avg CER% | Avg Latency(ms) | Docs Scored | Errors |")
    print("|------|-------|----------|-----------------|-------------|--------|")
    for s in summary_sorted:
        model_short = s["model"].split("/")[-1]
        cer_str = f"{s['avg_cer_pct']:.2f}" if s["avg_cer_pct"] is not None else "N/A"
        lat_str = str(s["avg_latency_ms"]) if s["avg_latency_ms"] is not None else "N/A"
        print(f"| {s['rank']} | {model_short} | {cer_str} | {lat_str} | {s['docs_scored']}/8 | {s['errors']} |")

    if summary_sorted and summary_sorted[0]["avg_cer"] is not None:
        print(f"\n🏆 Winner: {summary_sorted[0]['model']} (avg CER: {summary_sorted[0]['avg_cer_pct']}%)")
    else:
        print("\n⚠️  No model completed all tests successfully")


if __name__ == "__main__":
    main()
