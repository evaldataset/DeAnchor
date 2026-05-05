#!/bin/bash
# GPT-4o Score-blind experiment (n=50)
# Requires: OPENAI_API_KEY environment variable
# Estimated cost: ~$5-10, ~30 minutes

set -e

cd "$(dirname "$0")/.."
source .venv/bin/activate

echo "=== GPT-4o Score-blind Experiment ==="
echo "Model: gpt-4o"
echo "Condition: score-blind"
echo "Input: experiments/results/llm_input_ieee_cis_100.jsonl"
echo "Max samples: 50 (25 FP + 25 TP)"

if [ -z "$OPENAI_API_KEY" ]; then
    echo "ERROR: OPENAI_API_KEY not set"
    echo "Run: export OPENAI_API_KEY=sk-..."
    exit 1
fi

python experiments/run_scoreblind.py \
    --input experiments/results/llm_input_ieee_cis_100.jsonl \
    --model gpt-4o \
    --max_samples 50 \
    --output experiments/results/scoreblind_ieee_cis_gpt4o_50.jsonl

echo ""
echo "=== Running bootstrap analysis ==="
python evaluation/bootstrap_ci.py \
    --input experiments/results/scoreblind_ieee_cis_gpt4o_50.jsonl

echo ""
echo "Done! Results saved to experiments/results/scoreblind_ieee_cis_gpt4o_50.jsonl"
echo "Next: update paper/main.tex Table 2 with GPT-4o score-blind results"
