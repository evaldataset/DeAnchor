"""Ground-truth-aligned fine-tuning data 생성.

GPT-4o-mini의 anchored 출력 대신, 실제 라벨 기반 설명을 생성.
FP → "이 거래는 정상이다" 라벨로 학습.
"""

import json, os, sys, random
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data" / "finetune"

SYSTEM = "You are a financial fraud analyst providing accurate assessments."

FP_TEMPLATE = """This transaction was flagged by the ML model but is LEGITIMATE. Explain why.

Transaction: {tx_text}
ML Score: {score:.4f}

The ML model was WRONG about this transaction. Provide a balanced assessment that correctly identifies it as likely legitimate.

```json
{{"fraud_likelihood": {target_fl}, "evidence_for_fraud": {evidence_for}, "evidence_against_fraud": {evidence_against}, "recommendation": "RELEASE", "confidence": 0.8}}
```"""

TP_TEMPLATE = """This transaction was correctly flagged as FRAUD. Explain why.

Transaction: {tx_text}
ML Score: {score:.4f}

The ML model was CORRECT. Provide evidence supporting the fraud classification.

```json
{{"fraud_likelihood": {target_fl}, "evidence_for_fraud": {evidence_for}, "evidence_against_fraud": {evidence_against}, "recommendation": "ESCALATE_AS_FRAUD", "confidence": 0.8}}
```"""


def generate_gt_data():
    from llm_explainer.llm_inference import LLMConfig, LLMInference

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("OPENAI_API_KEY needed")
        return

    config = LLMConfig(model_name="gpt-4o-mini", backend="openai", temperature=0.2)
    llm = LLMInference(config)
    llm.load()

    with open("experiments/results/llm_input_ieee_cis_100.jsonl") as f:
        txns = [json.loads(line) for line in f]

    # 200 FP + 200 TP
    fp = [t for t in txns if t["category"] == "false_positive"][:200]
    tp = [t for t in txns if t["category"] == "true_positive"][:200]

    train_data = []
    for tx in fp[:160] + tp[:160]:
        is_fraud = tx["is_fraud"]
        target_fl = 0.3 if is_fraud == 0 else 0.8
        template = FP_TEMPLATE if is_fraud == 0 else TP_TEMPLATE

        prompt = template.format(
            tx_text=tx["text"],
            score=tx["fraud_score"],
            target_fl=target_fl,
            evidence_for='["..."]',
            evidence_against='["..."]',
        )
        response = llm.generate(prompt, SYSTEM)

        train_data.append({
            "messages": [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": f"Assess this transaction.\nTransaction: {tx['text']}\nML Score: {tx['fraud_score']:.4f}"},
                {"role": "assistant", "content": response},
            ],
            "metadata": {"is_fraud": is_fraud, "category": tx["category"]},
        })

        if len(train_data) % 50 == 0:
            print(f"  Generated {len(train_data)} samples")

    random.seed(42)
    random.shuffle(train_data)
    split = int(len(train_data) * 0.8)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path, data in [("gt_train.jsonl", train_data[:split]), ("gt_val.jsonl", train_data[split:])]:
        with open(OUTPUT_DIR / path, "w") as f:
            for d in data:
                f.write(json.dumps(d, ensure_ascii=False) + "\n")

    fp_count = sum(1 for d in train_data if d["metadata"]["is_fraud"] == 0)
    print(f"\nGT-aligned data: {len(train_data)} total ({fp_count} FP + {len(train_data)-fp_count} TP)")
    print(f"Train: {split}, Val: {len(train_data)-split}")


if __name__ == "__main__":
    generate_gt_data()
