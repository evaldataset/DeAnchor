"""Baseline (non-fine-tuned) Qwen2.5-7B eval on the DISJOINT test set.

Provides a fair comparison point for `run_finetuned_disjoint_eval.py`:
same prompts, same test items, same decoding — only the LoRA adapter differs.

Outputs:
    experiments/results/qwen_base_disjoint_eval.jsonl
    experiments/results/qwen_base_disjoint_summary.json
"""

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from evaluation.bootstrap_ci import two_sample_permutation_test
from llm_explainer.llm_inference import _parse_json_response
from llm_explainer.prompts.fp_explanation import SYSTEM_PROMPT, build_prompt

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = os.environ.get(
    "QWEN_MODEL_DIR",
    str(Path.home() / ".cache/modelscope/Qwen/Qwen2.5-7B-Instruct"),
)
TEST_MANIFEST = BASE_DIR / "data" / "finetune" / "test_manifest.jsonl"
RESULTS_DIR = BASE_DIR / "experiments" / "results"


def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
    quant = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR,
        quantization_config=quant,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    return model, tokenizer


def generate(model, tokenizer, prompt: str, system: str) -> str:
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": prompt},
    ]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.0,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    gen = out[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(gen, skip_special_tokens=True).strip()


def main():
    print(f"Loading base model from {MODEL_DIR}")
    model, tokenizer = load_model()
    print(f"VRAM: {torch.cuda.memory_allocated()/1024**3:.1f} GB")

    with open(TEST_MANIFEST) as f:
        test = [json.loads(l) for l in f]
    print(f"Test set: {len(test)} transactions")

    fp_fl, tp_fl = [], []
    results = []
    start = time.time()
    for i, tx in enumerate(test):
        prompt = build_prompt(tx["text"], tx["fraud_score"], threshold=0.5)
        resp = generate(model, tokenizer, prompt, SYSTEM_PROMPT)
        parsed = _parse_json_response(resp)
        fl = parsed.get("fraud_likelihood") if isinstance(parsed, dict) else None
        if isinstance(fl, (int, float)):
            if tx["category"] == "false_positive":
                fp_fl.append(fl)
            elif tx["category"] == "true_positive":
                tp_fl.append(fl)
        results.append({
            "transaction_id": tx["transaction_id"],
            "category": tx["category"],
            "ml_score": tx["fraud_score"],
            "fraud_likelihood": fl if isinstance(fl, (int, float)) else None,
            "raw_response": resp[:800],
        })
        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{len(test)}] elapsed {time.time()-start:.0f}s")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    raw_out = RESULTS_DIR / "qwen_base_disjoint_eval.jsonl"
    with open(raw_out, "w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    fp = np.array(fp_fl); tp = np.array(tp_fl)
    if len(fp) > 0 and len(tp) > 0:
        delta = float(tp.mean() - fp.mean())
        pooled = np.sqrt((tp.var(ddof=1) + fp.var(ddof=1)) / 2)
        d = float(delta / pooled) if pooled > 0 else 0
        test_result = two_sample_permutation_test(list(tp), list(fp))
        summary = {
            "eval_set": "disjoint_test",
            "model": "Qwen2.5-7B-Instruct (base, no LoRA)",
            "n_fp": len(fp),
            "n_tp": len(tp),
            "n_parse_errors": sum(1 for r in results if r["fraud_likelihood"] is None),
            "fp_mean": round(float(fp.mean()), 4),
            "tp_mean": round(float(tp.mean()), 4),
            "delta": round(delta, 4),
            "cohens_d": round(d, 3),
            "p_value": test_result["p_value"],
            "test_type": "unpaired_permutation",
        }
    else:
        summary = {
            "error": "no valid predictions",
            "n_fp": len(fp), "n_tp": len(tp),
            "n_parse_errors": sum(1 for r in results if r["fraud_likelihood"] is None),
        }

    summary_out = RESULTS_DIR / "qwen_base_disjoint_summary.json"
    with open(summary_out, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults:")
    print(json.dumps(summary, indent=2))
    print(f"Saved to {summary_out}")


if __name__ == "__main__":
    main()
