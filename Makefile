.PHONY: test lint benchmark clean install dev

install:
	pip install -e .

dev:
	pip install -e ".[dev]"

test:
	pytest tests/ -v --tb=short

lint:
	ruff check src/ tests/
	@echo "Lint passed ✅"

benchmark:
	cd benchmarks && python3 run_benchmark.py

harness:
	cd benchmarks && python3 harness.py

ci-check:
	cd scripts && bash ci-benchmark.sh

clean:
	rm -rf build/ dist/ *.egg-info src/*.egg-info .pytest_cache __pycache__
	find . -name '__pycache__' -exec rm -rf {} + 2>/dev/null || true
	find . -name '*.pyc' -delete 2>/dev/null || true
