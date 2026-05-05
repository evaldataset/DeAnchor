#!/bin/bash
# Qwen2.5-7B/14B 로컬 실험 실행 스크립트
# HuggingFace 네트워크 복구 후 사용
#
# 사전 조건:
# 1. HuggingFace 접근 가능
# 2. GPU VRAM 16GB+ (7B-4bit) 또는 32GB+ (14B-4bit)
# 3. pip install bitsandbytes accelerate

set -e
cd "$(dirname "$0")/.."
source .venv/bin/activate

MODEL=${1:-"Qwen/Qwen2.5-7B-Instruct"}
echo "=== Local LLM Experiment: $MODEL ==="

# Step 1: Score-aware experiment
echo "[1/2] Score-aware experiment (100+100)..."
python llm_explainer/pipeline.py \
  --input experiments/results/llm_input_ieee_cis_100.jsonl \
  --backend local --model "$MODEL" \
  --tasks fp_explain --max_samples 200 --threshold 0.8221 \
  --output experiments/results/local_scoreaware_ieee_cis.jsonl

# Step 2: Score-blind experiment
echo "[2/2] Score-blind experiment (100+100)..."
python experiments/run_scoreblind.py \
  --input experiments/results/llm_input_ieee_cis_100.jsonl \
  --output experiments/results/local_scoreblind_ieee_cis.jsonl \
  --model "$MODEL" --max_samples 200

# Step 3: Bootstrap CI analysis
echo "[3/3] Analysis..."
python -c "
from evaluation.bootstrap_ci import analyze_experiment_with_ci
print('=== Local Score-Aware ===')
analyze_experiment_with_ci('experiments/results/local_scoreaware_ieee_cis.jsonl')
print()
print('=== Local Score-Blind ===')
analyze_experiment_with_ci('experiments/results/local_scoreblind_ieee_cis.jsonl')
"

echo "=== Done ==="
