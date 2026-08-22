#!/usr/bin/env python3
"""Seed combined-rankings.json into Supabase benchmark_runs + benchmark_results tables."""
import json
import os
import sys
import uuid
from datetime import datetime, timezone

import requests

SUPABASE_URL = os.environ.get("SUPABASE_URL", "https://dzeokqqozcnzbpxvsniv.supabase.co")
SERVICE_KEY = os.environ["SUPABASE_SERVICE_KEY"]  # no literal fallback; push-protection caught one here
DRY_RUN = "--dry-run" in sys.argv

def sb_headers():
    return {
        "apikey": SERVICE_KEY,
        "Authorization": f"Bearer {SERVICE_KEY}",
        "Content-Type": "application/json",
        "Prefer": "return=representation",
    }

def get_existing_engines():
    """Get engines that already have benchmark runs."""
    url = f"{SUPABASE_URL}/rest/v1/benchmark_runs?select=engine_id"
    r = requests.get(url, headers=sb_headers())
    return set(row["engine_id"] for row in r.json())

def main():
    combined = json.load(open("results/combined-rankings.json"))
    existing = get_existing_engines()
    print(f"Existing engines in DB: {len(existing)}")
    print(f"Engines in combined: {len(combined)}")

    now = datetime.now(timezone.utc).isoformat()
    seeded = 0

    for engine_id, data in combined.items():
        # Skip engines with 0 score (broken)
        if data["avg_score"] <= 0:
            print(f"  SKIP {engine_id} (score=0)")
            continue

        if engine_id in existing:
            print(f"  SKIP {engine_id} (already in DB)")
            continue

        # Create benchmark run
        run_id = str(uuid.uuid4())
        # Normalize score to 0-1 range (harness uses 0-100)
        overall_score = data["avg_score"] / 100.0

        run_row = {
            "id": run_id,
            "engine_id": engine_id,
            "run_at": now,
            "overall_score": round(overall_score, 4),
            "avg_latency_ms": data["avg_latency_ms"],
            "harness_version": "2.0.0",
        }

        if DRY_RUN:
            print(f"  DRY RUN: would insert run for {engine_id} (score={data['avg_score']})")
            continue

        # Insert run
        r = requests.post(
            f"{SUPABASE_URL}/rest/v1/benchmark_runs",
            headers=sb_headers(),
            json=run_row,
        )
        if r.status_code not in (200, 201):
            print(f"  ERROR inserting run for {engine_id}: {r.status_code} {r.text[:200]}")
            continue

        # Insert per-image results
        result_rows = []
        for img_name, img_data in data.get("scores_by_image", {}).items():
            if not isinstance(img_data, dict):
                continue
            score = img_data.get("score", 0)
            if score is None or score <= 0:
                continue  # Skip errors
            result_rows.append({
                "id": str(uuid.uuid4()),
                "run_id": run_id,
                "engine_id": engine_id,
                "category": img_name,
                "score": round(score / 100.0, 4),
                "latency_ms": img_data.get("latency_ms", 0),
            })

        if result_rows:
            r = requests.post(
                f"{SUPABASE_URL}/rest/v1/benchmark_results",
                headers=sb_headers(),
                json=result_rows,
            )
            if r.status_code not in (200, 201):
                print(f"  ERROR inserting results for {engine_id}: {r.status_code} {r.text[:200]}")
                continue

        print(f"  SEEDED {engine_id}: score={data['avg_score']}, {len(result_rows)} image results")
        seeded += 1

    print(f"\nDone. Seeded {seeded} new engines.")

if __name__ == "__main__":
    main()
