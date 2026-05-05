"""Qwen2.5-7B LoRA Fine-tuning for fraud explanation.

SFT with LoRA on fraud explanation task.
"""

import json
import os
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, TaskType
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import SFTTrainer

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = os.environ.get("QWEN_MODEL_DIR", str(Path.home() / ".cache/modelscope/Qwen/Qwen2.5-7B-Instruct"))
DATA_DIR = BASE_DIR / "data" / "finetune"
OUTPUT_DIR = BASE_DIR / "ml_baseline" / "models" / "qwen_finetuned"


def load_dataset(path: Path) -> Dataset:
    data = []
    with open(path) as f:
        for line in f:
            item = json.loads(line)
            # Format as single text for SFT
            messages = item["messages"]
            text = ""
            for msg in messages:
                if msg["role"] == "system":
                    text += f"<|im_start|>system\n{msg['content']}<|im_end|>\n"
                elif msg["role"] == "user":
                    text += f"<|im_start|>user\n{msg['content']}<|im_end|>\n"
                elif msg["role"] == "assistant":
                    text += f"<|im_start|>assistant\n{msg['content']}<|im_end|>\n"
            data.append({"text": text})
    return Dataset.from_list(data)


def main():
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    print("Loading model (4-bit)...")
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR,
        quantization_config=quant_config,
        device_map="auto",
        trust_remote_code=True,
    )

    print(f"VRAM: {torch.cuda.memory_allocated()/1024**3:.1f} GB")

    # LoRA config
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=0.05,
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Load data
    print("Loading datasets...")
    train_ds = load_dataset(DATA_DIR / "train.jsonl")
    val_ds = load_dataset(DATA_DIR / "val.jsonl")
    print(f"  Train: {len(train_ds)}, Val: {len(val_ds)}")

    # Training
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        num_train_epochs=3,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=2e-4,
        warmup_steps=10,
        logging_steps=10,
        save_steps=50,
        eval_strategy="steps",
        eval_steps=50,
        bf16=True,
        report_to="none",
        remove_unused_columns=False,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
    )

    print("\nStarting fine-tuning...")
    trainer.train()

    # Save
    model.save_pretrained(str(OUTPUT_DIR / "final"))
    tokenizer.save_pretrained(str(OUTPUT_DIR / "final"))
    print(f"\nFine-tuned model saved to {OUTPUT_DIR / 'final'}")


if __name__ == "__main__":
    main()
