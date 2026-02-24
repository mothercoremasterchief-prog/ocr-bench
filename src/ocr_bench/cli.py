"""CLI entry point for ocr-bench."""

from __future__ import annotations

import argparse
import json
import sys

from ocr_bench.scorer import score


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="ocr-bench",
        description="Score OCR output quality without ground truth",
    )
    sub = parser.add_subparsers(dest="command")

    score_p = sub.add_parser("score", help="Score OCR text quality")
    group = score_p.add_mutually_exclusive_group(required=True)
    group.add_argument("--text", "-t", help="OCR text to score")
    group.add_argument("--file", "-f", help="File containing OCR text")
    score_p.add_argument("--json", action="store_true", help="Output as JSON")

    args = parser.parse_args()

    if args.command != "score":
        parser.print_help()
        sys.exit(1)

    if args.text:
        text = args.text
    else:
        with open(args.file, "r") as fh:
            text = fh.read()

    result = score(text)

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(f"Score:    {result['score']}/100")
        print(f"  Typo rate:          {result['typo_rate']:.2%}")
        print(f"  Gibberish ratio:    {result['gibberish_ratio']:.2%}")
        print(f"  Single letter ratio:{result['single_letter_ratio']:.2%}")
        print(f"  Char-separated:     {result['char_separated_count']}")
        print(f"  Word accuracy:      {result['word_accuracy']:.2%}")


if __name__ == "__main__":
    main()
