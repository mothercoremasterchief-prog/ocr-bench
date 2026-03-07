#!/usr/bin/env python3
"""Unified OCR benchmark harness — multi-endpoint, multi-format, ground-truth comparison.

Usage:
    python benchmarks/harness.py                          # default config
    python benchmarks/harness.py --config path/to.yaml
    python benchmarks/harness.py --engines gpt-4o,tesseract-local
    python benchmarks/harness.py --images printed_receipt,noisy_invoice
    python benchmarks/harness.py --dry-run
"""
from __future__ import annotations

import argparse
import base64
import glob
import json
import os
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import requests
import yaml

# ocr_bench is installed from the repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ocr_bench import score as score_text


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

OcrResult = dict[str, Any]  # score, metrics, latency_ms, token_usage, error, text_preview


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
        with open(path) as f:
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
                    {"type": "text", "text": "Extract all text from this image exactly as it appears. Output only the extracted text, nothing else."},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}},
                ],
            }
        ],
        "max_tokens": 4096,
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


ADAPTERS = {
    "local": _call_local,
    "openocr_router": _call_local,  # same format
    "openai_vision": _call_openai_vision,
    "nim": _call_openai_vision,  # NIM uses OpenAI-compatible format
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
                  image_filter: list[str] | None = None, dry_run: bool = False) -> dict[str, Any]:
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
        images = {k: v for k, v in images.items() if k in image_filter}

    _log(f"Engines: {len(endpoints)} | Images: {len(images)} | Ground truth: {len(ground_truth)}")

    if dry_run:
        _log("\n--- DRY RUN ---")
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
                    "latency_ms": round(latency_ms), "token_usage": token_usage,
                    "error": error, "text_preview": "",
                }
                continue

            score_result = score_text(text, ground_truth=gt)
            gts = score_result.get("ground_truth_similarity")
            gts_str = f"  gt={gts:.3f}" if gts is not None else ""
            preview = text[:80].replace("\n", " ")
            _log(f"score={score_result['score']:5.1f}{gts_str}  latency={latency_ms:7.0f}ms  [{preview}...]")

            results[ep_name][img_name] = {
                "score": score_result["score"],
                "ground_truth_similarity": gts,
                "latency_ms": round(latency_ms),
                "token_usage": token_usage,
                "error": None,
                "text_preview": text[:200],
            }

    # Build summary
    summary: dict[str, dict[str, Any]] = {}
    for ep_name, img_results in results.items():
        scores = [r["score"] for r in img_results.values() if not r.get("error")]
        latencies = [r["latency_ms"] for r in img_results.values() if not r.get("error")]
        gt_sims = [r["ground_truth_similarity"] for r in img_results.values()
                   if not r.get("error") and r.get("ground_truth_similarity") is not None]
        errors = sum(1 for r in img_results.values() if r.get("error"))

        summary[ep_name] = {
            "avg_score": round(sum(scores) / len(scores), 1) if scores else 0,
            "avg_gt_similarity": round(sum(gt_sims) / len(gt_sims), 4) if gt_sims else None,
            "avg_latency_ms": round(sum(latencies) / len(latencies)) if latencies else 0,
            "total_errors": errors,
            "images_tested": len(img_results),
        }

    output = {
        "run_timestamp": datetime.now(timezone.utc).isoformat(),
        "config": str(Path(cfg.get("_config_path", "config.yaml"))),
        "results": results,
        "summary": summary,
    }

    return output


def write_json(output: dict[str, Any], path: str) -> None:
    """Write results as JSON."""
    with open(path, "w") as f:
        json.dump(output, f, indent=2)
    _log(f"\n✅ JSON results written to {path}")


def write_markdown(output: dict[str, Any], path: str) -> None:
    """Write results as a markdown table."""
    summary = output.get("summary", {})
    if not summary:
        return

    lines = [
        f"# OCR Benchmark Results",
        f"",
        f"Run: {output.get('run_timestamp', 'N/A')}",
        f"",
        f"| Engine | Avg Score | GT Match | Avg Latency | Errors | Images |",
        f"|--------|-----------|----------|-------------|--------|--------|",
    ]

    for name in sorted(summary, key=lambda n: summary[n]["avg_score"], reverse=True):
        s = summary[name]
        gt_str = f"{s['avg_gt_similarity']*100:.1f}%" if s["avg_gt_similarity"] is not None else "N/A"
        lines.append(
            f"| {name} | {s['avg_score']:.1f} | {gt_str} | {s['avg_latency_ms']}ms | {s['total_errors']} | {s['images_tested']} |"
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
    parser.add_argument("--images", help="Comma-separated image names to test (default: all)")
    parser.add_argument("--dry-run", action="store_true", help="Validate config and list what would run")
    args = parser.parse_args()

    cfg = load_config(args.config)
    cfg["_config_path"] = args.config

    engine_filter = args.engines.split(",") if args.engines else None
    image_filter = args.images.split(",") if args.images else None

    output = run_benchmark(cfg, engine_filter=engine_filter, image_filter=image_filter, dry_run=args.dry_run)

    if args.dry_run or not output:
        return

    base_dir = str(Path(__file__).resolve().parent)
    json_path = os.path.join(base_dir, cfg["output"]["json_path"])
    md_path = os.path.join(base_dir, cfg["output"]["markdown_path"])

    write_json(output, json_path)
    write_markdown(output, md_path)

    # Print summary table to stderr
    summary = output.get("summary", {})
    _log(f"\n{'Engine':20s} {'Avg Score':>10s} {'GT Match':>10s} {'Avg Latency':>12s} {'Errors':>7s}")
    _log("-" * 65)
    for name in sorted(summary, key=lambda n: summary[n]["avg_score"], reverse=True):
        s = summary[name]
        gt_str = f"{s['avg_gt_similarity']*100:.1f}%" if s["avg_gt_similarity"] is not None else "    N/A"
        _log(f"{name:20s} {s['avg_score']:10.1f} {gt_str:>10s} {s['avg_latency_ms']:10d}ms {s['total_errors']:7d}")


if __name__ == "__main__":
    main()
