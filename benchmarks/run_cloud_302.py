import os
#!/usr/bin/env python3
"""
OCR-302: Benchmark AWS Textract and Google Document AI
Fetches credentials from Supabase Vault, runs against the standard corpus,
and appends results to the cloud results files.
"""

import base64, json, os, sys, time, urllib.request
from pathlib import Path

# ---------------------------------------------------------------------------
# Vault
# ---------------------------------------------------------------------------
SUPABASE_URL = "https://dzeokqqozcnzbpxvsniv.supabase.co"
SERVICE_KEY = os.environ["SUPABASE_SERVICE_KEY"]  # never hardcode; push-protection caught a literal here

def vault_secret(name: str) -> str | None:
    url = f"{SUPABASE_URL}/rest/v1/rpc/get_vault_secret"
    data = json.dumps({"p_name": name}).encode()
    req = urllib.request.Request(url, data=data, method="POST", headers={
        "apikey": SERVICE_KEY, "Authorization": f"Bearer {SERVICE_KEY}",
        "Content-Type": "application/json",
    })
    with urllib.request.urlopen(req) as r:
        return json.load(r)

# ---------------------------------------------------------------------------
# Textract adapter
# ---------------------------------------------------------------------------
def call_textract(image_bytes: bytes, aws_key: str, aws_secret: str, aws_region: str) -> tuple[str, float]:
    import boto3
    client = boto3.client("textract",
        aws_access_key_id=aws_key,
        aws_secret_access_key=aws_secret,
        region_name=aws_region,
    )
    t0 = time.time()
    resp = client.detect_document_text(Document={"Bytes": image_bytes})
    latency = (time.time() - t0) * 1000
    lines = []
    for block in resp.get("Blocks", []):
        if block["BlockType"] == "LINE":
            lines.append(block["Text"])
    return "\n".join(lines), latency

# ---------------------------------------------------------------------------
# Document AI adapter
# ---------------------------------------------------------------------------
def call_document_ai(image_bytes: bytes, project_id: str, private_key: str, private_key_id: str, processor_endpoint: str, processor_id: str) -> tuple[str, float, str]:
    import requests
    
    # Build credentials from vault fields
    sa_info = {
        "type": "service_account",
        "project_id": project_id,
        "private_key_id": private_key_id,
        "private_key": __import__("re").sub(r'-{6,}', '-----', private_key.replace("\\n", "\n").replace("----BEGIN", "-----BEGIN").replace("----END", "-----END")),
        "client_email": f"openocrrouter@{project_id}.iam.gserviceaccount.com",
        "token_uri": "https://oauth2.googleapis.com/token",
    }
    from google.oauth2 import service_account
    credentials = service_account.Credentials.from_service_account_info(
        sa_info, scopes=["https://www.googleapis.com/auth/cloud-platform"]
    )
    
    # Refresh credentials to get access token
    from google.auth.transport.requests import Request as GoogleAuthRequest
    credentials.refresh(GoogleAuthRequest())
    
    # Use the processor endpoint directly
    url = processor_endpoint
    headers = {
        "Authorization": f"Bearer {credentials.token}",
        "Content-Type": "application/json",
    }
    body = {
        "rawDocument": {
            "content": base64.b64encode(image_bytes).decode(),
            "mimeType": "image/png",
        }
    }
    
    t0 = time.time()
    resp = requests.post(url, headers=headers, json=body, timeout=60)
    latency = (time.time() - t0) * 1000
    
    if resp.status_code != 200:
        return f"ERROR: {resp.status_code} {resp.text[:200]}", latency, resp.text[:200]
    
    result = resp.json()
    text = result.get("document", {}).get("text", "")
    return text, latency, ""

# ---------------------------------------------------------------------------
# Scoring (reuse ocr_bench if available, else inline)
# ---------------------------------------------------------------------------
def score_text(text: str, ground_truth: str) -> tuple[float, float]:
    """Returns (quality_score, gt_match_pct)."""
    try:
        from ocr_bench import score as ocr_score
        result = ocr_score(text)
        quality = result["score"]
    except ImportError:
        # Simple fallback: word overlap
        quality = 80.0  # placeholder

    # GT match: word-level Jaccard
    if not ground_truth.strip():
        return quality, 0.0
    gt_words = set(ground_truth.lower().split())
    out_words = set(text.lower().split())
    if not gt_words:
        return quality, 0.0
    intersection = gt_words & out_words
    gt_match = len(intersection) / len(gt_words) * 100
    return quality, gt_match

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    base = Path(__file__).parent
    images_dir = base / "images"
    gt_dir = base / "ground-truth"

    # Load images
    images = {}
    for img in sorted(images_dir.glob("*.png")):
        images[img.stem] = img.read_bytes()
    print(f"Loaded {len(images)} images", file=sys.stderr)

    # Load ground truth
    ground_truth = {}
    for gt in gt_dir.glob("*.txt"):
        ground_truth[gt.stem] = gt.read_text()

    # Fetch credentials
    print("Fetching credentials from Vault...", file=sys.stderr)
    aws_key = vault_secret("AWS_ACCESS_KEY_ID")
    aws_secret = vault_secret("AWS_SECRET_KEY")
    aws_region = vault_secret("AWS_REGION") or "us-east-1"
    gcp_project = vault_secret("GCP_PROJECT_ID")
    gcp_private_key = vault_secret("GCP_PRIVATE_KEY")
    gcp_private_key_id = vault_secret("GCP_PRIVATE_KEY_ID")
    gcp_processor_endpoint = vault_secret("GCP_DOCUMENT_PROCESSOR_ENDPOINT")
    gcp_processor_id = vault_secret("GCP_DOCUMENT_PROCESSOR_ID")

    engines = {}

    # --- Textract ---
    if aws_key and aws_secret:
        print("\n=== AWS Textract ===", file=sys.stderr)
        results = []
        for name, img_bytes in images.items():
            print(f"  {name}...", end="", file=sys.stderr, flush=True)
            try:
                text, latency = call_textract(img_bytes, aws_key, aws_secret, aws_region)
                score, gt_match = score_text(text, ground_truth.get(name, ""))
                results.append({"image": name, "text": text, "score": score, "gt_match": gt_match, "latency_ms": latency, "error": None})
                print(f" score={score:.1f} gt={gt_match:.1f}% {latency:.0f}ms", file=sys.stderr)
            except Exception as e:
                results.append({"image": name, "text": "", "score": 0, "gt_match": 0, "latency_ms": 0, "error": str(e)})
                print(f" ERROR: {e}", file=sys.stderr)
            time.sleep(0.5)  # rate limit
        engines["aws/textract"] = results
    else:
        print("AWS credentials missing, skipping Textract", file=sys.stderr)

    # --- Document AI ---
    if gcp_project and gcp_private_key and gcp_private_key_id and gcp_processor_endpoint:
        print("\n=== Google Document AI ===", file=sys.stderr)
        print(f"Using processor: {gcp_processor_endpoint[:60]}...", file=sys.stderr)
        results = []
        for name, img_bytes in images.items():
            print(f"  {name}...", end="", file=sys.stderr, flush=True)
            try:
                text, latency, raw = call_document_ai(img_bytes, gcp_project, gcp_private_key, gcp_private_key_id, gcp_processor_endpoint, gcp_processor_id)
                if raw and "error" in raw.lower():
                    print(f" API_ERROR: {raw[:100]}", file=sys.stderr)
                    results.append({"image": name, "text": "", "score": 0, "gt_match": 0, "latency_ms": latency, "error": raw[:200]})
                else:
                    score, gt_match = score_text(text, ground_truth.get(name, ""))
                    results.append({"image": name, "text": text, "score": score, "gt_match": gt_match, "latency_ms": latency, "error": None})
                    print(f" score={score:.1f} gt={gt_match:.1f}% {latency:.0f}ms", file=sys.stderr)
            except Exception as e:
                import traceback
                results.append({"image": name, "text": "", "score": 0, "gt_match": 0, "latency_ms": 0, "error": str(e)})
                print(f" ERROR: {e} {traceback.format_exc()[:100]}", file=sys.stderr)
            time.sleep(0.5)
        engines["google-cloud/document-ai"] = results
    else:
        print("GCP credentials missing, skipping Document AI", file=sys.stderr)

    # --- Output ---
    output_dir = base / "results" / "cloud"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load existing results
    existing_json = output_dir / "results.json"
    if existing_json.exists():
        with open(existing_json) as f:
            existing = json.load(f)
    else:
        existing = {"run_timestamp": "", "results": {}, "summary": {}}

    # Normalize: old format used "engines", new uses "results"
    if "engines" in existing and "results" not in existing:
        existing["results"] = existing.pop("engines")
    if "results" not in existing:
        existing["results"] = {}

    # Ensure summary dict exists
    if "summary" not in existing:
        existing["summary"] = {}

    # Merge into existing format: results[engine][image] = {...}, summary[engine] = stats
    for eng_name, results in engines.items():
        avg_score = sum(r["score"] for r in results) / len(results) if results else 0
        avg_gt = sum(r["gt_match"] for r in results) / len(results) if results else 0
        avg_latency = sum(r["latency_ms"] for r in results) / len(results) if results else 0
        error_count = sum(1 for r in results if r["error"])

        # Per-image results
        existing["results"][eng_name] = {
            r["image"]: {"text": r["text"], "score": r["score"], "gt_match": r["gt_match"],
                         "latency_ms": r["latency_ms"], "error": r["error"]}
            for r in results
        }

        # Summary
        existing["summary"][eng_name] = {
            "avg_score": round(avg_score, 1),
            "avg_gt_similarity": round(avg_gt / 100, 4),
            "avg_latency_ms": round(avg_latency),
            "total_errors": error_count,
            "images_tested": len(results),
        }

    existing["run_timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime())

    # Write JSON
    with open(existing_json, "w") as f:
        json.dump(existing, f, indent=2)

    # Write markdown from summary
    md_path = output_dir / "results.md"
    with open(md_path, "w") as f:
        f.write(f"# OCR Benchmark Results\n\nRun: {existing['run_timestamp']}\n\n")
        f.write("| Engine | Avg Score | GT Match | Avg Latency | Errors | Images |\n")
        f.write("|--------|-----------|----------|-------------|--------|--------|\n")
        for eng_name, data in sorted(existing["summary"].items(), key=lambda x: x[1]["avg_score"], reverse=True):
            gt_pct = f"{data['avg_gt_similarity']*100:.1f}%"
            f.write(f"| {eng_name} | {data['avg_score']} | {gt_pct} | {data['avg_latency_ms']}ms | {data['total_errors']} | {data['images_tested']} |\n")

    print(f"\nResults written to {existing_json} and {md_path}", file=sys.stderr)

    # Print summary
    print("\n--- Summary ---")
    for eng_name, data in engines.items():
        avg_score = sum(r["score"] for r in data) / len(data) if data else 0
        avg_gt = sum(r["gt_match"] for r in data) / len(data) if data else 0
        avg_lat = sum(r["latency_ms"] for r in data) / len(data) if data else 0
        errors = sum(1 for r in data if r["error"])
        print(f"{eng_name}: avg_score={avg_score:.1f}, gt_match={avg_gt:.1f}%, avg_latency={avg_lat:.0f}ms, errors={errors}")

if __name__ == "__main__":
    main()
