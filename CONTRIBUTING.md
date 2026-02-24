# Contributing to ocr-bench

Thanks for your interest in improving OCR quality measurement! Here's how to contribute.

## Development Setup

```bash
git clone https://github.com/open-ocr/ocr-bench.git
cd ocr-bench
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
make test
```

## Running Tests

```bash
make test       # pytest with coverage
make lint       # ruff check + mypy
make benchmark  # run full engine benchmark (requires running OCR engines)
```

## Adding a Metric

1. Add your metric function to `src/ocr_bench/metrics/` (or inline in `scorer.py`)
2. Register it in `scorer.py` with a weight
3. Add tests in `tests/`
4. Update README.md metric table
5. Run `make test` to verify

## Adding Benchmark Images

1. Place the image in `benchmarks/images/`
2. Write a ground-truth transcript in `benchmarks/ground-truth/` (same base name, `.txt` extension)
3. Add the image to the `IMAGES` dict in `benchmarks/run_benchmark.py`
4. Run `make benchmark` to regenerate scores

## Pull Request Process

1. Fork the repo and create a feature branch
2. Write tests for new functionality
3. Ensure `make test && make lint` passes
4. Open a PR with a clear description of what and why

## Code Style

- Python 3.10+
- Type hints encouraged
- ruff for formatting/linting
- Keep dependencies minimal (wordfreq is the heaviest dep)

## Reporting Issues

Use GitHub Issues. Include:
- Python version
- Input text that produces unexpected scores
- Expected vs actual score
- Full traceback if applicable
