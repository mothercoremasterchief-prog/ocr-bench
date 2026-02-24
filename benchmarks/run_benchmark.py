#!/usr/bin/env python3
"""Benchmark all 6 local OCR engines with ocr-bench scoring (including ground-truth similarity)."""
import base64
import json
import sys
import time
import requests
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from ocr_bench import score as score_text

ENGINES = {
    "easyocr":   {"port": 8100},
    "tesseract": {"port": 8101},
    "doctr":     {"port": 8102},
    "surya":     {"port": 8103},
    "ocrmypdf":  {"port": 8105},
    "rapidocr":  {"port": 8106},
}

IMAGES = {
    "printed_receipt":      "images/printed_receipt.png",
    "handwritten_letter":   "images/handwritten_letter.png",
    "noisy_invoice":        "images/noisy_invoice.png",
    "dense_paragraph":      "images/dense_paragraph.png",
    "multi_column":         "images/multi_column.png",
    "table_form":           "images/table_form.png",
    "noisy_meeting_notice": "images/noisy_meeting_notice.png",
    "code_snippet":         "images/code_snippet.png",
    "fine_print":           "images/fine_print.png",
    "spec_sheet":           "images/spec_sheet.png",
}

GROUND_TRUTH = {
    "printed_receipt":      "ground-truth/printed_receipt.txt",
    "handwritten_letter":   "ground-truth/handwritten_letter.txt",
    "noisy_invoice":        "ground-truth/noisy_invoice.txt",
    "dense_paragraph":      "ground-truth/dense_paragraph.txt",
    "multi_column":         "ground-truth/multi_column.txt",
    "table_form":           "ground-truth/table_form.txt",
    "noisy_meeting_notice": "ground-truth/noisy_meeting_notice.txt",
    "code_snippet":         "ground-truth/code_snippet.txt",
    "fine_print":           "ground-truth/fine_print.txt",
    "spec_sheet":           "ground-truth/spec_sheet.txt",
}


def ocr_engine(port: int, image_b64: str) -> tuple[str, float]:
    """Send image to engine, return (text, latency_ms)."""
    url = f"http://localhost:{port}/v1/ocr"
    payload = {"input": {"type": "base64", "data_base64": image_b64}}
    t0 = time.time()
    try:
        r = requests.post(url, json=payload, timeout=120)
        latency = (time.time() - t0) * 1000
        if r.status_code != 200:
            return f"ERROR: HTTP {r.status_code}", latency
        data = r.json()
        text = data.get("text", data.get("result", ""))
        if isinstance(text, list):
            text = "\n".join(str(t) for t in text)
        return str(text), latency
    except Exception as e:
        return f"ERROR: {e}", (time.time() - t0) * 1000


def load_ground_truth(base_dir: str) -> dict[str, str]:
    """Load ground-truth transcripts."""
    gt = {}
    for img_name, gt_path in GROUND_TRUTH.items():
        full_path = os.path.join(base_dir, gt_path)
        if os.path.exists(full_path):
            with open(full_path, "r") as f:
                gt[img_name] = f.read()
    return gt


def main():
    base_dir = os.path.dirname(__file__)
    gt_texts = load_ground_truth(base_dir)
    results = {}

    for img_name, img_path in IMAGES.items():
        full_path = os.path.join(base_dir, img_path)
        with open(full_path, "rb") as f:
            image_b64 = base64.b64encode(f.read()).decode()
        gt = gt_texts.get(img_name)
        gt_label = " (with ground truth)" if gt else ""
        print(f"\n{'='*60}")
        print(f"Image: {img_name}{gt_label}")
        print(f"{'='*60}")

        for eng_name, eng_cfg in ENGINES.items():
            print(f"  {eng_name:12s} ... ", end="", flush=True)
            text, latency_ms = ocr_engine(eng_cfg["port"], image_b64)

            if text.startswith("ERROR"):
                print(f"FAILED ({text})")
                if eng_name not in results:
                    results[eng_name] = {}
                results[eng_name][img_name] = {
                    "score": 0, "metrics": {}, "latency_ms": round(latency_ms),
                    "error": text, "text_preview": "",
                }
                continue

            score_result = score_text(text, ground_truth=gt)
            score_val = score_result["score"]
            metrics = {k: v for k, v in score_result.items() if k != "score"}
            gts = score_result.get("ground_truth_similarity")
            gts_str = f"  gt={gts:.3f}" if gts is not None else ""
            preview = text[:80].replace("\n", " ")
            print(f"score={score_val:5.1f}{gts_str}  latency={latency_ms:7.0f}ms  [{preview}...]")

            if eng_name not in results:
                results[eng_name] = {}
            results[eng_name][img_name] = {
                "score": score_val,
                "metrics": metrics,
                "latency_ms": round(latency_ms),
                "text_preview": text[:200],
            }

    # Compute averages
    for eng_name in results:
        scores = [v["score"] for v in results[eng_name].values() if isinstance(v, dict) and "error" not in v]
        latencies = [v["latency_ms"] for v in results[eng_name].values() if isinstance(v, dict) and "error" not in v]
        gt_sims = [
            v["metrics"]["ground_truth_similarity"]
            for v in results[eng_name].values()
            if isinstance(v, dict) and "error" not in v and v.get("metrics", {}).get("ground_truth_similarity") is not None
        ]
        results[eng_name]["average_score"] = round(sum(scores) / len(scores), 1) if scores else 0
        results[eng_name]["average_latency_ms"] = round(sum(latencies) / len(latencies)) if latencies else 0
        results[eng_name]["average_gt_similarity"] = round(sum(gt_sims) / len(gt_sims), 4) if gt_sims else None

    out_path = os.path.join(base_dir, "engine-scores.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n✅ Results written to {out_path}")

    # Summary table
    print(f"\n{'Engine':12s} {'Avg Score':>10s} {'GT Match':>10s} {'Avg Latency':>12s}")
    print("-" * 48)
    for eng in sorted(results, key=lambda e: results[e].get("average_score", 0), reverse=True):
        avg = results[eng].get("average_score", 0)
        gt_avg = results[eng].get("average_gt_similarity")
        gt_str = f"{gt_avg*100:8.1f}%" if gt_avg is not None else "     N/A"
        lat = results[eng].get("average_latency_ms", 0)
        print(f"{eng:12s} {avg:10.1f} {gt_str} {lat:10d}ms")


if __name__ == "__main__":
    main()
