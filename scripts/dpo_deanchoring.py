"""DPO De-anchoring: Preference-based learning to reduce score anchoring.

Preference pairs:
  - Chosen: response where fraud_likelihood reflects ground truth
    (FP → low fl, TP → high fl)
  - Rejected: response where fraud_likelihood copies ML score (anchored)

Train with TRL DPOTrainer on these pairs.
"""

import json
import os
import sys
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from trl import DPOConfig, DPOTrainer

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = os.environ.get("QWEN_MODEL_DIR", str(Path.home() / ".cache/modelscope/Qwen/Qwen2.5-7B-Instruct"))
OUTPUT_DIR = str(BASE_DIR / "ml_baseline" / "models" / "qwen_dpo")


def create_dpo_data():
    """기존 score-aware 결과에서 DPO preference pair 생성."""
    # Score-aware 결과 (anchored outputs)
    with open(BASE_DIR / "experiments/results/scoreaware_ieee_cis_200.jsonl") as f:
        aware_data = [json.loads(line) for line in f]

    pairs = []
    for r in aware_data:
        tx = r.get("original", {})
        exp = r.get("fp_explanation", {})
        if exp.get("parse_error") or exp.get("api_error"):
            continue

        is_fraud = tx.get("is_fraud", -1)
        ml_score = r.get("ml_score", 0.5)
        tx_text = tx.get("text", "")

        # Anchored response (rejected): copies ML score
        anchored = json.dumps({
            "fraud_likelihood": round(ml_score, 2),
            "evidence_for_fraud": exp.get("evidence_for_fraud", [])[:2],
            "evidence_against_fraud": exp.get("evidence_against_fraud", [])[:2],
            "recommendation": "ESCALATE_AS_FRAUD" if ml_score > 0.8 else "HOLD_FOR_REVIEW",
        })

        # De-anchored response (chosen): reflects ground truth
        if is_fraud == 0:  # FP → should be low
            target_fl = max(0.15, ml_score * 0.3)  # much lower than ML score
            chosen = json.dumps({
                "fraud_likelihood": round(target_fl, 2),
                "evidence_for_fraud": exp.get("evidence_for_fraud", [])[:1],
                "evidence_against_fraud": exp.get("evidence_against_fraud", [])[:2] + [
                    "The ML score may be inflated due to feature coincidence rather than genuine fraud patterns."
                ],
                "recommendation": "HOLD_FOR_REVIEW",
            })
        else:  # TP → should be high
            target_fl = min(0.95, ml_score * 1.05)
            chosen = json.dumps({
                "fraud_likelihood": round(target_fl, 2),
                "evidence_for_fraud": exp.get("evidence_for_fraud", [])[:2] + [
                    "Multiple indicators converge to suggest genuine fraudulent activity."
                ],
                "evidence_against_fraud": exp.get("evidence_against_fraud", [])[:1],
                "recommendation": "ESCALATE_AS_FRAUD",
            })

        prompt = f"Assess this transaction for fraud risk.\n\nTransaction: {tx_text}\nML Score: {ml_score:.4f}"

        pairs.append({
            "prompt": prompt,
            "chosen": chosen,
            "rejected": anchored,
            "is_fraud": is_fraud,
        })

    return pairs


def main():
    print("Creating DPO preference pairs...")
    pairs = create_dpo_data()
    print(f"  Total pairs: {len(pairs)}")
    fp_count = sum(1 for p in pairs if p["is_fraud"] == 0)
    print(f"  FP pairs: {fp_count}, TP pairs: {len(pairs) - fp_count}")

    # Split
    import random
    random.seed(42)
    random.shuffle(pairs)
    split = int(len(pairs) * 0.8)
    train_pairs = pairs[:split]
    val_pairs = pairs[split:]

    # Convert to Dataset
    train_ds = Dataset.from_list([{
        "prompt": p["prompt"],
        "chosen": p["chosen"],
        "rejected": p["rejected"],
    } for p in train_pairs])
    val_ds = Dataset.from_list([{
        "prompt": p["prompt"],
        "chosen": p["chosen"],
        "rejected": p["rejected"],
    } for p in val_pairs])
    print(f"  Train: {len(train_ds)}, Val: {len(val_ds)}")

    # Load model
    print("\nLoading model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quant = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR, quantization_config=quant, device_map="auto", trust_remote_code=True,
    )
    print(f"  VRAM: {torch.cuda.memory_allocated()/1024**3:.1f} GB")

    # LoRA
    lora = LoraConfig(
        r=16, lora_alpha=32,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=0.05, task_type=TaskType.CAUSAL_LM,
    )

    # DPO config
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    training_args = DPOConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=2,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=5e-5,
        warmup_steps=10,
        logging_steps=10,
        save_steps=100,
        bf16=True,
        report_to="none",
        remove_unused_columns=False,
        max_length=1024,
    )

    print("\nStarting DPO training...")
    trainer = DPOTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        peft_config=lora,
    )

    trainer.train()

    # Save
    final_dir = f"{OUTPUT_DIR}/final"
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"\nDPO model saved to {final_dir}")


if __name__ == "__main__":
    main()
