"""Cross-LLM Evaluation: Generator ≠ Judge.

GPT-4o-mini가 생성한 설명을 다른 모델(Qwen2.5-7B)이 평가.
Self-evaluation bias 해소.
"""

import argparse, json, os, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

RESULTS_DIR = Path(__file__).resolve().parent / "results"

JUDGE_PROMPT = """Rate this fraud explanation. Output ONLY a JSON object, nothing else.

Transaction: {tx_text}

Explanation: {explanation}

Output exactly this format (replace N with 1-5):
{{"coherence": N, "completeness": N, "clarity": N, "actionability": N, "overall": N}}"""


def run(predictions_path, judge_model, n_samples=30):
    # Load generator outputs
    with open(predictions_path) as f:
        preds = [json.loads(line) for line in f][:n_samples]

    # Judge with local Qwen
    if "/" in judge_model:
        # Local model
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        import torch
        tokenizer = AutoTokenizer.from_pretrained(judge_model, trust_remote_code=True)
        quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16, bnb_4bit_quant_type="nf4")
        model = AutoModelForCausalLM.from_pretrained(judge_model, quantization_config=quant, device_map="auto", trust_remote_code=True)
        print(f"Local judge loaded: {judge_model}")

        def generate(prompt):
            messages = [{"role": "user", "content": prompt}]
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(text, return_tensors="pt").to(model.device)
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=200, temperature=0.0, do_sample=False, pad_token_id=tokenizer.eos_token_id)
            return tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    else:
        from llm_explainer.llm_inference import LLMConfig, LLMInference
        config = LLMConfig(model_name=judge_model, backend="openai", temperature=0.0)
        llm = LLMInference(config)
        llm.load()
        generate = lambda prompt: llm.generate(prompt)

    from llm_explainer.llm_inference import _parse_json_response
    import numpy as np

    results = []
    for i, pred in enumerate(preds):
        tx = pred.get("original", {})
        exp = pred.get("fp_explanation") or pred.get("anomaly_explanation", {})
        exp_text = json.dumps(exp, indent=1)[:500] if isinstance(exp, dict) else str(exp)[:500]

        prompt = JUDGE_PROMPT.format(tx_text=tx.get("text", "")[:300], explanation=exp_text)
        raw = generate(prompt)
        try:
            parsed = _parse_json_response(raw)
        except Exception:
            parsed = {"raw_response": raw[:200], "parse_error": True}
        parsed["category"] = tx.get("category", "?")
        results.append(parsed)

        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{n_samples}]")

    # Aggregate
    dims = ["coherence", "completeness", "clarity", "actionability", "overall"]
    print(f"\n=== Cross-LLM Judge Results ({judge_model.split('/')[-1]}) ===")
    for dim in dims:
        scores = [r.get(dim, 0) for r in results if isinstance(r.get(dim), (int, float))]
        if scores:
            print(f"  {dim:15s}: {np.mean(scores):.2f} ± {np.std(scores):.2f} (n={len(scores)})")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "cross_llm_judge.json"
    with open(out_path, "w") as f:
        json.dump({"results": results, "judge_model": judge_model.split("/")[-1]}, f, indent=2, ensure_ascii=False)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument(
        "--judge_model",
        default=os.environ.get(
            "QWEN_MODEL_DIR",
            str(Path.home() / ".cache/modelscope/Qwen/Qwen2.5-7B-Instruct"),
        ),
    )
    parser.add_argument("--n_samples", type=int, default=30)
    args = parser.parse_args()
    run(args.predictions, args.judge_model, args.n_samples)
