#!/bin/bash
# OCR-369: Run all free-tier benchmark configs sequentially
set -e

cd "$(dirname "$0")"

# API keys live in ./.env (gitignored, mode 600) — they were hardcoded here
# until 2026-08-16, which put six live provider keys in a git working tree one
# `git add -A` away from being committed. Never inline them again.
if [ -f ./.env ]; then
  set -a; . ./.env; set +a
else
  echo "run-all-free.sh: ./.env not found — provider keys unavailable" >&2
  exit 1
fi

# OPENROUTER_API_KEY is expected from the ambient environment (~/.bashrc), not .env
export OPENROUTER_API_KEY="${OPENROUTER_API_KEY:-}"

for _k in SILICONFLOW_API_KEY SAMBANOVA_API_KEY MISTRAL_API_KEY \
          CEREBRAS_API_KEY DEEPINFRA_API_KEY GOOGLE_AI_STUDIO_API_KEY; do
  [ -n "${!_k:-}" ] || { echo "run-all-free.sh: $_k missing from .env" >&2; exit 1; }
done
unset _k

CONFIGS=(
  "config-or-free-vision.yaml"
  "config-sambanova.yaml"
  "config-mistral-free.yaml"
  "config-cerebras.yaml"
  "config-deepinfra.yaml"
  "config-siliconflow.yaml"
)

LOG_FILE="./results/benchmark-run-$(date +%Y%m%d-%H%M%S).log"
mkdir -p ./results

echo "=== OCR-369 Benchmark Run Started: $(date) ===" | tee "$LOG_FILE"

for config in "${CONFIGS[@]}"; do
  echo "" | tee -a "$LOG_FILE"
  echo ">>> Running $config at $(date)" | tee -a "$LOG_FILE"
  if python3 harness.py --config "$config" 2>&1 | tee -a "$LOG_FILE"; then
    echo "<<< $config COMPLETED at $(date)" | tee -a "$LOG_FILE"
  else
    echo "<<< $config FAILED at $(date)" | tee -a "$LOG_FILE"
  fi
  # Rate limit pause between providers
  sleep 5
done

echo "" | tee -a "$LOG_FILE"
echo "=== All benchmarks complete: $(date) ===" | tee -a "$LOG_FILE"

# List result files
echo "" | tee -a "$LOG_FILE"
echo "Result files:" | tee -a "$LOG_FILE"
find ./results -name "results.json" -newer ./run-all-free.sh | tee -a "$LOG_FILE"
