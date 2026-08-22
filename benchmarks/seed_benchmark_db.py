#!/usr/bin/env python3
"""
OCR-357: Seed benchmark_results DB with ground truth and OCR output text.

Runs tesseract, easyocr, doctr against the standard corpus images,
then upserts ground_truth + ocr_output into the DB for existing benchmark rows.

Usage:
  cd /home/ben/.openclaw/workspace/projects/ocr-bench/benchmarks
  SUPABASE_URL=... SUPABASE_SERVICE_KEY=... python seed_benchmark_db.py [--dry-run]
"""
import base64
import json
import os
import sys
import time
import requests

SUPABASE_URL = os.environ.get("SUPABASE_URL", "https://dzeokqqozcnzbpxvsniv.supabase.co")
SERVICE_KEY = os.environ["SUPABASE_SERVICE_KEY"]  # no literal fallback; push-protection caught one here
DRY_RUN = "--dry-run" in sys.argv

BASE_DIR = os.path.dirname(__file__)

ENGINES = {
    "openocr/tesseract": 8101,
    "openocr/easyocr":   8100,
    "openocr/doctr":     8102,
}

IMAGES = {
    "code_snippet":         "images/code_snippet.png",
    "dense_paragraph":      "images/dense_paragraph.png",
    "fine_print":           "images/fine_print.png",
    "handwritten_letter":   "images/handwritten_letter.png",
    "multi_column":         "images/multi_column.png",
    "noisy_invoice":        "images/noisy_invoice.png",
    "noisy_meeting_notice": "images/noisy_meeting_notice.png",
    "printed_receipt":      "images/printed_receipt.png",
    "spec_sheet":           "images/spec_sheet.png",
    "table_form":           "images/table_form.png",
}

GROUND_TRUTH_FILES = {
    "code_snippet":         "ground-truth/code_snippet.txt",
    "dense_paragraph":      "ground-truth/dense_paragraph.txt",
    "fine_print":           "ground-truth/fine_print.txt",
    "handwritten_letter":   "ground-truth/handwritten_letter.txt",
    "multi_column":         "ground-truth/multi_column.txt",
    "noisy_invoice":        "ground-truth/noisy_invoice.txt",
    "noisy_meeting_notice": "ground-truth/noisy_meeting_notice.txt",
    "printed_receipt":      "ground-truth/printed_receipt.txt",
    "spec_sheet":           "ground-truth/spec_sheet.txt",
    "table_form":           "ground-truth/table_form.txt",
}


def sb_headers():
    return {
        "apikey": SERVICE_KEY,
        "Authorization": f"Bearer {SERVICE_KEY}",
        "Content-Type": "application/json",
        "Prefer": "return=representation",
    }


def get_benchmark_runs():
    """Get latest benchmark run id per engine."""
    url = f"{SUPABASE_URL}/rest/v1/benchmark_runs?select=id,engine_id,run_at&order=run_at.desc"
    r = requests.get(url, headers=sb_headers())
    rows = r.json()
    # Pick latest run per engine
    runs = {}
    for row in rows:
        if row["engine_id"] not in runs:
            runs[row["engine_id"]] = row["id"]
    return runs


def get_benchmark_results(run_id: str):
    """Get all benchmark_results rows for a run."""
    url = f"{SUPABASE_URL}/rest/v1/benchmark_results?run_id=eq.{run_id}&select=id,category"
    r = requests.get(url, headers=sb_headers())
    return {row["category"]: row["id"] for row in r.json()}


def load_ground_truth():
    gt = {}
    for img_name, gt_path in GROUND_TRUTH_FILES.items():
        full = os.path.join(BASE_DIR, gt_path)
        if os.path.exists(full):
            with open(full) as f:
                gt[img_name] = f.read().strip()
        else:
            print(f"  WARNING: no ground truth file for {img_name}")
    return gt


def ocr_engine(port: int, img_b64: str) -> str | None:
    url = f"http://localhost:{port}/v1/ocr"
    payload = {"input": {"type": "base64", "data_base64": img_b64}}
    try:
        r = requests.post(url, json=payload, timeout=60)
        if r.status_code != 200:
            return None
        data = r.json()
        text = data.get("text", data.get("result", ""))
        if isinstance(text, list):
            text = "\n".join(str(t) for t in text)
        return str(text).strip()
    except Exception as e:
        print(f"    OCR error: {e}")
        return None


def update_result(result_id: str, ground_truth: str, ocr_output: str | None):
    url = f"{SUPABASE_URL}/rest/v1/benchmark_results?id=eq.{result_id}"
    payload = {"ground_truth": ground_truth}
    if ocr_output:
        payload["ocr_output"] = ocr_output
    r = requests.patch(url, json=payload, headers=sb_headers())
    return r.status_code in (200, 204)


def main():
    print(f"{'=== DRY RUN ===' if DRY_RUN else '=== LIVE RUN ==='}")

    ground_truth = load_ground_truth()
    print(f"Loaded {len(ground_truth)} ground truth files")

    runs = get_benchmark_runs()
    print(f"Found {len(runs)} benchmark runs: {list(runs.keys())}")

    for engine_id, port in ENGINES.items():
        if engine_id not in runs:
            print(f"\nSkipping {engine_id} — no benchmark run in DB")
            continue

        run_id = runs[engine_id]
        results_map = get_benchmark_results(run_id)
        print(f"\n{engine_id} (run {run_id}): {len(results_map)} result rows")

        # Check if engine is up
        try:
            r = requests.get(f"http://localhost:{port}/health", timeout=2)
            engine_up = r.status_code == 200
        except Exception:
            engine_up = False
        print(f"  Engine on :{port}: {'UP' if engine_up else 'DOWN'}")

        for img_name, img_path in IMAGES.items():
            full_img = os.path.join(BASE_DIR, img_path)
            if not os.path.exists(full_img):
                print(f"  SKIP {img_name}: image not found")
                continue

            gt = ground_truth.get(img_name, "")
            if not gt:
                print(f"  SKIP {img_name}: no ground truth")
                continue

            # The DB categories may differ from img_name — try both dash and underscore
            result_id = results_map.get(img_name)
            if not result_id:
                # Try dash version
                dash_name = img_name.replace("_", "-")
                result_id = results_map.get(dash_name)
            if not result_id:
                print(f"  SKIP {img_name}: no DB row (categories: {list(results_map.keys())[:3]}...)")
                continue

            # Get OCR output if engine is up
            ocr_output = None
            if engine_up:
                with open(full_img, "rb") as f:
                    img_b64 = base64.b64encode(f.read()).decode()
                ocr_output = ocr_engine(port, img_b64)
                print(f"  {img_name}: gt={len(gt)}chars, ocr={len(ocr_output) if ocr_output else 0}chars")
            else:
                print(f"  {img_name}: gt={len(gt)}chars, ocr=SKIPPED (engine down)")

            if not DRY_RUN:
                ok = update_result(result_id, gt, ocr_output)
                if not ok:
                    print(f"    FAILED to update {result_id}")

    print("\nDone.")


if __name__ == "__main__":
    main()
