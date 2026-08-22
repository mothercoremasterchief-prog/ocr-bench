#!/usr/bin/env python3
"""Unified OCR benchmark harness — multi-endpoint, multi-format, ground-truth comparison.

Usage:
    python benchmarks/harness.py                          # default config
    python benchmarks/harness.py --config path/to.yaml
    python benchmarks/harness.py --engines gpt-4o,tesseract-local
    python benchmarks/harness.py --images printed_receipt,noisy_invoice
    python benchmarks/harness.py --profile minimum
    python benchmarks/harness.py --dry-run
"""
from __future__ import annotations

import argparse
import base64
import glob
import hashlib
import json
import os
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests
import yaml

# Prefer the working tree so benchmark runs assess the code in this checkout.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from ocr_bench import evaluate as evaluate_text
from ocr_bench import score as score_text


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

OcrResult = dict[str, Any]

PROFILE_DIR = Path(__file__).resolve().parent / "profiles"
OCR_PROMPT = (
    "Extract all text from this image exactly as it appears. "
    "Output only the extracted text, nothing else."
)


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_config(path: str) -> dict[str, Any]:
    """Load and validate YAML config file."""
    with open(path) as f:
        cfg = yaml.safe_load(f)
    if not cfg or "endpoints" not in cfg:
        raise ValueError(f"Config {path} must contain 'endpoints' list")
    # Defaults
    corpus = cfg.setdefault("corpus", {})
    corpus.setdefault("images_dir", "./images")
    corpus.setdefault("ground_truth_dir", "./ground-truth")
    output = cfg.setdefault("output", {})
    output.setdefault("json_path", "./results.json")
    output.setdefault("markdown_path", "./results.md")
    for ep in cfg["endpoints"]:
        ep.setdefault("timeout", 120)
        ep.setdefault("max_retries", 2)
        ep.setdefault("retry_delay", 10)
    return cfg


def load_profile(name_or_path: str) -> dict[str, Any]:
    """Load a benchmark profile by built-in name or JSON path."""

    path = (
        PROFILE_DIR / f"{name_or_path}.json"
        if name_or_path == "minimum"
        else Path(name_or_path)
    )
    with open(path) as f:
        profile = json.load(f)
    selected = profile.get("selected_images")
    if not isinstance(selected, list) or not selected:
        raise ValueError(
            f"Profile {path} must contain a non-empty selected_images list"
        )
    if len(selected) != len(set(selected)):
        raise ValueError(f"Profile {path} contains duplicate selected_images")
    profile["_path"] = str(path)
    return profile


def validate_profile_assets(
    profile: dict[str, Any],
    images: dict[str, bytes],
    ground_truth: dict[str, str],
) -> None:
    """Reject silent changes to assets frozen by a benchmark profile."""

    expected = profile.get("asset_sha256", {})
    actual_groups = {
        "images": {
            name: hashlib.sha256(content).hexdigest()
            for name, content in images.items()
        },
        "ground_truth": {
            name: hashlib.sha256(text.encode("utf-8")).hexdigest()
            for name, text in ground_truth.items()
        },
    }
    for group, expected_hashes in expected.items():
        for name, expected_hash in expected_hashes.items():
            actual_hash = actual_groups.get(group, {}).get(name)
            if actual_hash != expected_hash:
                raise ValueError(
                    f"Profile asset checksum mismatch: {group}/{name}"
                )


# ---------------------------------------------------------------------------
# Image collection
# ---------------------------------------------------------------------------

def collect_images(cfg: dict[str, Any], base_dir: str) -> dict[str, bytes]:
    """Collect images from images_dir and optionally pdf_dir. Returns {name: bytes}."""
    images: dict[str, bytes] = {}

    # Static images
    img_dir = os.path.join(base_dir, cfg["corpus"]["images_dir"])
    if os.path.isdir(img_dir):
        for ext in ("*.png", "*.jpg", "*.jpeg"):
            for path in sorted(glob.glob(os.path.join(img_dir, ext))):
                name = Path(path).stem
                with open(path, "rb") as f:
                    images[name] = f.read()

    # PDF conversion
    pdf_dir = cfg["corpus"].get("pdf_dir")
    if pdf_dir and os.path.isdir(pdf_dir):
        try:
            from pdf2image import convert_from_path
        except ImportError:
            _log("WARNING: pdf2image not installed — skipping PDF conversion. pip install pdf2image")
            return images

        tmp_dir = tempfile.mkdtemp(prefix="ocr_bench_pdf_")
        _log(f"Converting PDFs from {pdf_dir} → {tmp_dir}")
        for pdf_path in sorted(glob.glob(os.path.join(pdf_dir, "*.pdf"))):
            stem = Path(pdf_path).stem
            try:
                pages = convert_from_path(pdf_path, dpi=200, fmt="png")
                for i, page_img in enumerate(pages, 1):
                    name = f"{stem}_page{i}"
                    out_path = os.path.join(tmp_dir, f"{name}.png")
                    page_img.save(out_path, "PNG")
                    with open(out_path, "rb") as f:
                        images[name] = f.read()
            except Exception as e:
                _log(f"WARNING: Failed to convert {pdf_path}: {e}")

    return images


def load_ground_truth(cfg: dict[str, Any], base_dir: str) -> dict[str, str]:
    """Load ground truth text files. Returns {image_name: text}."""
    gt: dict[str, str] = {}
    gt_dir = os.path.join(base_dir, cfg["corpus"]["ground_truth_dir"])
    if not os.path.isdir(gt_dir):
        return gt
    for path in glob.glob(os.path.join(gt_dir, "*.txt")):
        name = Path(path).stem
        with open(path, encoding="utf-8") as f:
            gt[name] = f.read()
    return gt


# ---------------------------------------------------------------------------
# Endpoint adapters
# ---------------------------------------------------------------------------

def _call_local(url: str, image_b64: str, timeout: int, **_: Any) -> tuple[str, dict | None]:
    """Call local OCR engine (base64 JSON format). Returns (text, token_usage)."""
    payload = {"input": {"type": "base64", "data_base64": image_b64}}
    r = requests.post(url, json=payload, timeout=timeout)
    r.raise_for_status()
    data = r.json()
    text = data.get("text", data.get("result", ""))
    if isinstance(text, list):
        text = "\n".join(str(t) for t in text)
    return str(text), None


def _call_openai_vision(url: str, image_b64: str, timeout: int,
                        api_key: str | None = None, model: str = "gpt-4o",
                        temperature: float = 0.0,
                        **_: Any) -> tuple[str, dict | None]:
    """Call OpenAI-compatible vision API. Returns (text, token_usage)."""
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": OCR_PROMPT},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}},
                ],
            }
        ],
        "max_tokens": 4096,
        "temperature": temperature,
    }
    r = requests.post(url, json=payload, headers=headers, timeout=timeout)
    r.raise_for_status()
    data = r.json()
    text = data["choices"][0]["message"]["content"]
    usage = data.get("usage")
    token_usage = None
    if usage:
        token_usage = {
            "prompt": usage.get("prompt_tokens", 0),
            "completion": usage.get("completion_tokens", 0),
        }
    return text, token_usage


def _call_anthropic_vision(url: str, image_b64: str, timeout: int,
                           api_key: str | None = None, model: str = "claude-3-5-sonnet-20241022",
                           temperature: float = 0.0,
                           **_: Any) -> tuple[str, dict | None]:
    """Call Anthropic Messages API with vision. Returns (text, token_usage)."""
    headers = {
        "Content-Type": "application/json",
        "x-api-key": api_key or "",
        "anthropic-version": "2023-06-01",
    }
    payload = {
        "model": model,
        "max_tokens": 4096,
        "temperature": temperature,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": image_b64}},
                    {"type": "text", "text": OCR_PROMPT},
                ],
            }
        ],
    }
    r = requests.post(url, json=payload, headers=headers, timeout=timeout)
    r.raise_for_status()
    data = r.json()
    text = data["content"][0]["text"]
    usage = data.get("usage")
    token_usage = None
    if usage:
        token_usage = {
            "prompt": usage.get("input_tokens", 0),
            "completion": usage.get("output_tokens", 0),
        }
    return text, token_usage


def _call_aws_textract(url: str, image_b64: str, timeout: int,
                       **_: Any) -> tuple[str, dict | None]:
    """Call AWS Textract DetectDocumentText. Returns (text, None).

    Uses boto3 — ignores url param. Credentials from env vars:
    AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION.
    """
    import boto3

    region = os.environ.get("AWS_REGION", "us-east-1")
    client = boto3.client("textract", region_name=region)

    image_bytes = base64.b64decode(image_b64)
    response = client.detect_document_text(
        Document={"Bytes": image_bytes}
    )

    # Extract LINE blocks in reading order
    lines = []
    for block in response.get("Blocks", []):
        if block["BlockType"] == "LINE":
            lines.append(block.get("Text", ""))

    return "\n".join(lines), None


def _call_gcp_document_ai(url: str, image_b64: str, timeout: int,
                          **_: Any) -> tuple[str, dict | None]:
    """Call Google Cloud Document AI processor. Returns (text, None)."""
    from google.api_core.client_options import ClientOptions
    from google.cloud import documentai

    processor_name = os.environ.get("GCP_DOCUMENT_PROCESSOR_ENDPOINT")
    if not processor_name:
        raise ValueError("Missing env var: GCP_DOCUMENT_PROCESSOR_ENDPOINT")

    # Credentials are loaded by the GCP client library from this env var when set.
    os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")

    image_bytes = base64.b64decode(image_b64)
    client_options = None
    if "/locations/" in processor_name:
        location = processor_name.split("/locations/", 1)[1].split("/", 1)[0]
        if location and location != "us":
            client_options = ClientOptions(
                api_endpoint=f"{location}-documentai.googleapis.com"
            )

    client = documentai.DocumentProcessorServiceClient(client_options=client_options)
    raw_document = documentai.RawDocument(
        content=image_bytes,
        mime_type="image/png",
    )
    request = documentai.ProcessRequest(
        name=processor_name,
        raw_document=raw_document,
    )
    response = client.process_document(request=request, timeout=timeout)
    return response.document.text or "", None


ADAPTERS = {
    "local": _call_local,
    "openocr_router": _call_local,  # same format
    "openai_vision": _call_openai_vision,
    "nim": _call_openai_vision,  # NIM uses OpenAI-compatible format
    "anthropic_vision": _call_anthropic_vision,
    "aws_textract": _call_aws_textract,
    "gcp_document_ai": _call_gcp_document_ai,
}


def call_endpoint(endpoint: dict[str, Any], image_b64: str) -> tuple[str, float, dict | None, str | None]:
    """Call an OCR endpoint with retries. Returns (text, latency_ms, token_usage, error)."""
    ep_type = endpoint["type"]
    adapter = ADAPTERS.get(ep_type)
    if not adapter:
        return "", 0.0, None, f"Unknown endpoint type: {ep_type}"

    api_key = None
    if endpoint.get("api_key_env"):
        api_key = os.environ.get(endpoint["api_key_env"])
        if not api_key:
            return "", 0.0, None, f"Missing env var: {endpoint['api_key_env']}"

    max_retries = endpoint.get("max_retries", 2)
    retry_delay = endpoint.get("retry_delay", 10)

    last_error = None
    for attempt in range(max_retries + 1):
        t0 = time.time()
        try:
            text, token_usage = adapter(
                url=endpoint["url"],
                image_b64=image_b64,
                timeout=endpoint["timeout"],
                api_key=api_key,
                model=endpoint.get("model", ""),
                temperature=endpoint.get("temperature", 0.0),
            )
            latency_ms = (time.time() - t0) * 1000
            return text, latency_ms, token_usage, None
        except requests.exceptions.HTTPError as e:
            latency_ms = (time.time() - t0) * 1000
            status = e.response.status_code if e.response is not None else 0
            last_error = f"HTTP {status}: {e}"
            if status in (429, 500, 502, 503, 504) and attempt < max_retries:
                wait = retry_delay * (attempt + 1)
                _log(f"  Retry {attempt + 1}/{max_retries} for {endpoint['name']} (HTTP {status}), waiting {wait}s")
                time.sleep(wait)
                continue
            break
        except requests.exceptions.Timeout:
            latency_ms = (time.time() - t0) * 1000
            last_error = f"Timeout after {endpoint['timeout']}s"
            if attempt < max_retries:
                _log(f"  Retry {attempt + 1}/{max_retries} for {endpoint['name']} (timeout)")
                time.sleep(retry_delay)
                continue
            break
        except Exception as e:
            latency_ms = (time.time() - t0) * 1000
            last_error = str(e)
            break

    return "", latency_ms, None, last_error


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def _log(msg: str) -> None:
    """Log to stderr."""
    print(msg, file=sys.stderr, flush=True)


def run_benchmark(cfg: dict[str, Any], engine_filter: list[str] | None = None,
                  image_filter: list[str] | None = None, dry_run: bool = False,
                  profile: dict[str, Any] | None = None) -> dict[str, Any]:
    """Run the full benchmark. Returns results dict."""
    base_dir = str(Path(__file__).resolve().parent)

    # Collect inputs
    images = collect_images(cfg, base_dir)
    ground_truth = load_ground_truth(cfg, base_dir)

    # Filter
    endpoints = cfg["endpoints"]
    if engine_filter:
        endpoints = [ep for ep in endpoints if ep["name"] in engine_filter]
    if image_filter:
        missing = sorted(set(image_filter) - set(images))
        if missing:
            raise ValueError(f"Requested images were not found: {', '.join(missing)}")
        images = {k: v for k, v in images.items() if k in image_filter}
    if profile:
        missing_ground_truth = sorted(set(images) - set(ground_truth))
        if missing_ground_truth:
            raise ValueError(
                "Profile images missing ground truth: "
                + ", ".join(missing_ground_truth)
            )
    ground_truth = {name: text for name, text in ground_truth.items() if name in images}
    if profile:
        validate_profile_assets(profile, images, ground_truth)

    _log(f"Engines: {len(endpoints)} | Images: {len(images)} | Ground truth: {len(ground_truth)}")

    if dry_run:
        _log("\n--- DRY RUN ---")
        if profile:
            _log(f"Profile: {profile.get('id', profile.get('_path', 'unknown'))}")
        _log("Endpoints:")
        for ep in endpoints:
            key_status = ""
            if ep.get("api_key_env"):
                has_key = bool(os.environ.get(ep["api_key_env"]))
                key_status = f" [key: {'✓' if has_key else '✗ ' + ep['api_key_env']}]"
            _log(f"  {ep['name']:20s} type={ep['type']:15s} url={ep['url']}{key_status}")
        _log("Images:")
        for name in sorted(images):
            gt_flag = " (has ground truth)" if name in ground_truth else ""
            _log(f"  {name}{gt_flag}")
        return {}

    # Run
    results: dict[str, dict[str, OcrResult]] = {}
    total = len(endpoints) * len(images)
    done = 0

    for ep in endpoints:
        ep_name = ep["name"]
        results[ep_name] = {}
        _log(f"\n{'='*60}")
        _log(f"Engine: {ep_name} ({ep['type']})")
        _log(f"{'='*60}")

        for img_name in sorted(images):
            done += 1
            _log(f"  [{done}/{total}] {img_name:30s} ... ", )

            image_b64 = base64.b64encode(images[img_name]).decode()
            gt = ground_truth.get(img_name)

            text, latency_ms, token_usage, error = call_endpoint(ep, image_b64)

            if error:
                _log(f"FAILED ({error})")
                results[ep_name][img_name] = {
                    "score": 0, "ground_truth_similarity": None,
                    "evaluation": None,
                    "latency_ms": round(latency_ms), "token_usage": token_usage,
                    "error": error, "text_preview": "",
                }
                continue

            score_result = score_text(text, ground_truth=gt)
            evaluation = evaluate_text(text, gt) if gt is not None else None
            gts = score_result.get("ground_truth_similarity")
            gts_str = f"  gt={gts:.3f}" if gts is not None else ""
            preview = text[:80].replace("\n", " ")
            _log(f"score={score_result['score']:5.1f}{gts_str}  latency={latency_ms:7.0f}ms  [{preview}...]")

            results[ep_name][img_name] = {
                "score": score_result["score"],
                "ground_truth_similarity": gts,
                "evaluation": evaluation,
                "latency_ms": round(latency_ms),
                "token_usage": token_usage,
                "error": None,
                "text_preview": text[:200],
                "full_text": text,
                "ground_truth": gt,
            }

    # Build summary
    summary: dict[str, dict[str, Any]] = {}
    for ep_name, img_results in results.items():
        scores = [r["score"] for r in img_results.values() if not r.get("error")]
        latencies = [r["latency_ms"] for r in img_results.values() if not r.get("error")]
        gt_sims = [r["ground_truth_similarity"] for r in img_results.values()
                   if not r.get("error") and r.get("ground_truth_similarity") is not None]
        evaluations = [r["evaluation"] for r in img_results.values()
                       if not r.get("error") and r.get("evaluation") is not None]
        errors = sum(1 for r in img_results.values() if r.get("error"))

        def average_metric(name: str) -> float | None:
            if not evaluations:
                return None
            return round(sum(e[name] for e in evaluations) / len(evaluations), 6)

        reference_characters = sum(e["reference_characters"] for e in evaluations)
        reference_words = sum(e["reference_words"] for e in evaluations)
        character_errors = sum(e["character_errors"] for e in evaluations)
        word_errors = sum(e["word_errors"] for e in evaluations)
        bag_errors = sum(e["bag_of_words_errors"] for e in evaluations)

        summary[ep_name] = {
            "avg_score": round(sum(scores) / len(scores), 1) if scores else 0,
            "avg_gt_similarity": round(sum(gt_sims) / len(gt_sims), 4) if gt_sims else None,
            "macro_cer": average_metric("character_error_rate"),
            "macro_wer": average_metric("word_error_rate"),
            "macro_bag_wer": average_metric("bag_of_words_error_rate"),
            "macro_reading_order_error": average_metric("reading_order_error"),
            "worst_cer": round(
                max(e["character_error_rate"] for e in evaluations), 6
            ) if evaluations else None,
            "worst_wer": round(
                max(e["word_error_rate"] for e in evaluations), 6
            ) if evaluations else None,
            "micro_cer": round(character_errors / reference_characters, 6)
            if reference_characters else None,
            "micro_wer": round(word_errors / reference_words, 6)
            if reference_words else None,
            "micro_bag_wer": round(bag_errors / reference_words, 6)
            if reference_words else None,
            "avg_latency_ms": round(sum(latencies) / len(latencies)) if latencies else 0,
            "total_errors": errors,
            "images_tested": len(img_results),
            "valid_benchmark": errors == 0 and len(evaluations) == len(img_results),
        }

    output = {
        "run_timestamp": datetime.now(timezone.utc).isoformat(),
        "config": str(Path(cfg.get("_config_path", "config.yaml"))),
        "profile": {
            "id": profile.get("id"),
            "version": profile.get("version"),
            "scope": profile.get("scope"),
            "path": profile.get("_path"),
            "selected_images": profile.get("selected_images"),
            "assessments": profile.get("assessments"),
        } if profile else None,
        "protocol": {
            "prompt": OCR_PROMPT,
            "temperature": 0.0,
            "evaluation_normalization": "Unicode NFC, case-folded, whitespace collapsed",
            "engines": [
                {
                    "name": endpoint.get("name"),
                    "type": endpoint.get("type"),
                    "model": endpoint.get("model"),
                    "temperature": endpoint.get("temperature", 0.0),
                }
                for endpoint in endpoints
            ],
        },
        "asset_sha256": {
            "images": {
                name: hashlib.sha256(content).hexdigest()
                for name, content in images.items()
            },
            "ground_truth": {
                name: hashlib.sha256(text.encode("utf-8")).hexdigest()
                for name, text in ground_truth.items()
            },
        },
        "results": results,
        "summary": summary,
    }

    return output


def write_to_supabase(output: dict[str, Any]) -> None:
    """Upsert benchmark results into Supabase benchmark_results table."""
    supabase_url = os.environ.get("NEXT_PUBLIC_SUPABASE_URL") or os.environ.get("SUPABASE_URL")
    supabase_key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
    if not supabase_url or not supabase_key:
        _log("⚠️  Supabase credentials not found — skipping DB write")
        return

    results = output.get("results", {})
    rows = []
    ts = output.get("run_timestamp", datetime.now(timezone.utc).isoformat())

    for engine_id, images in results.items():
        if not isinstance(images, dict):
            continue
        for image_name, data in images.items():
            if not isinstance(data, dict) or data.get("error"):
                continue
            rows.append({
                "engine_id": engine_id,
                "image_name": image_name,
                "score": data.get("score", 0),
                "ground_truth_similarity": data.get("ground_truth_similarity"),
                "latency_ms": data.get("latency_ms", 0),
                "run_timestamp": ts,
            })

    if not rows:
        _log("⚠️  No valid results to write to Supabase")
        return

    # Upsert via PostgREST (on conflict engine_id + image_name)
    headers = {
        "apikey": supabase_key,
        "Authorization": f"Bearer {supabase_key}",
        "Content-Type": "application/json",
        "Prefer": "resolution=merge-duplicates",
    }
    resp = requests.post(
        f"{supabase_url}/rest/v1/benchmark_results",
        json=rows,
        headers=headers,
        timeout=30,
    )
    if resp.ok:
        _log(f"✅ {len(rows)} benchmark results written to Supabase")
    else:
        _log(f"⚠️  Supabase write failed: {resp.status_code} {resp.text[:200]}")


def write_json(output: dict[str, Any], path: str) -> None:
    """Write results as JSON."""
    with open(path, "w") as f:
        json.dump(output, f, indent=2)
    _log(f"\n✅ JSON results written to {path}")


def _ranked_summary_names(summary: dict[str, dict[str, Any]]) -> list[str]:
    """Rank by macro CER when available, then fall back to the legacy score."""

    def ranking_key(name: str) -> tuple[float, ...]:
        values = summary[name]
        if values.get("macro_cer") is not None:
            return (0, values["macro_cer"], values.get("macro_wer") or 0)
        return (1, -values["avg_score"], 0)

    return sorted(summary, key=ranking_key)


def write_markdown(output: dict[str, Any], path: str) -> None:
    """Write results as a markdown table."""
    summary = output.get("summary", {})
    if not summary:
        return

    lines = [
        "# OCR Benchmark Results",
        "",
        f"Run: {output.get('run_timestamp', 'N/A')}",
        "",
        "| Engine | CER | Worst CER | WER | Order Error | Avg Score | "
        "GT Match | Avg Latency | Errors | Images | Valid |",
        "|--------|-----|-----------|-----|-------------|-----------|"
        "----------|-------------|--------|--------|-------|",
    ]

    for name in _ranked_summary_names(summary):
        s = summary[name]
        gt_str = f"{s['avg_gt_similarity']*100:.1f}%" if s["avg_gt_similarity"] is not None else "N/A"
        cer_str = f"{s['macro_cer']*100:.1f}%" if s.get("macro_cer") is not None else "N/A"
        worst_cer_str = f"{s['worst_cer']*100:.1f}%" if s.get("worst_cer") is not None else "N/A"
        wer_str = f"{s['macro_wer']*100:.1f}%" if s.get("macro_wer") is not None else "N/A"
        order_str = (
            f"{s['macro_reading_order_error']*100:.1f}%"
            if s.get("macro_reading_order_error") is not None else "N/A"
        )
        valid_str = "yes" if s.get("valid_benchmark") else "no"
        lines.append(
            f"| {name} | {cer_str} | {worst_cer_str} | {wer_str} | "
            f"{order_str} | "
            f"{s['avg_score']:.1f} | {gt_str} | {s['avg_latency_ms']}ms | "
            f"{s['total_errors']} | {s['images_tested']} | {valid_str} |"
        )

    lines.append("")
    with open(path, "w") as f:
        f.write("\n".join(lines))
    _log(f"✅ Markdown results written to {path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description="OCR Benchmark Harness")
    parser.add_argument("--config", default=os.path.join(os.path.dirname(__file__), "config.yaml"),
                        help="Path to config YAML (default: benchmarks/config.yaml)")
    parser.add_argument("--engines", help="Comma-separated engine names to run (default: all)")
    image_selection = parser.add_mutually_exclusive_group()
    image_selection.add_argument(
        "--images", help="Comma-separated image names to test (default: all)"
    )
    image_selection.add_argument(
        "--profile",
        help="Benchmark profile name ('minimum') or path to a profile JSON file",
    )
    parser.add_argument("--dry-run", action="store_true", help="Validate config and list what would run")
    args = parser.parse_args()

    cfg = load_config(args.config)
    cfg["_config_path"] = args.config

    engine_filter = args.engines.split(",") if args.engines else None
    profile = load_profile(args.profile) if args.profile else None
    image_filter = (
        profile["selected_images"]
        if profile
        else args.images.split(",") if args.images else None
    )

    output = run_benchmark(
        cfg,
        engine_filter=engine_filter,
        image_filter=image_filter,
        dry_run=args.dry_run,
        profile=profile,
    )

    if args.dry_run or not output:
        return

    base_dir = str(Path(__file__).resolve().parent)
    json_path = os.path.join(base_dir, cfg["output"]["json_path"])
    md_path = os.path.join(base_dir, cfg["output"]["markdown_path"])

    write_json(output, json_path)
    write_markdown(output, md_path)
    write_to_supabase(output)

    # Print summary table to stderr
    summary = output.get("summary", {})
    _log(
        f"\n{'Engine':20s} {'CER':>8s} {'WER':>8s} {'Order':>8s} "
        f"{'Avg Score':>10s} {'GT Match':>10s} {'Avg Latency':>12s} "
        f"{'Errors':>7s}"
    )
    _log("-" * 95)
    for name in _ranked_summary_names(summary):
        s = summary[name]
        gt_str = f"{s['avg_gt_similarity']*100:.1f}%" if s["avg_gt_similarity"] is not None else "    N/A"
        cer_str = f"{s['macro_cer']*100:.1f}%" if s.get("macro_cer") is not None else "N/A"
        wer_str = f"{s['macro_wer']*100:.1f}%" if s.get("macro_wer") is not None else "N/A"
        order_str = (
            f"{s['macro_reading_order_error']*100:.1f}%"
            if s.get("macro_reading_order_error") is not None else "N/A"
        )
        _log(
            f"{name:20s} {cer_str:>8s} {wer_str:>8s} {order_str:>8s} "
            f"{s['avg_score']:10.1f} {gt_str:>10s} "
            f"{s['avg_latency_ms']:10d}ms {s['total_errors']:7d}"
        )


if __name__ == "__main__":
    main()
