"""Build train/val/test splits for LoRA fine-tuning with DISJOINT transactions.

Fixes the train/eval contamination bug where eval items were drawn from the
same 200-sample source pool used to create the fine-tuning set. New protocol:

1. Sample a large superset (e.g., 400 items) from the ML-flagged region
2. Split into 3 disjoint slices: train, val, test
3. Save manifests with row indices so overlap is auditable

Usage:
    python scripts/prepare_finetune_disjoint.py \
        --dataset ieee_cis --n_train 160 --n_val 40 --n_test 50 --seed 42

Outputs:
    data/finetune/train.jsonl
    data/finetune/val.jsonl
    data/finetune/test.jsonl
    data/finetune/split_manifest.json  (row indices for each split)
"""

import argparse
import hashlib
import json
import random
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
FINETUNE_DIR = BASE / "data" / "finetune"


def text_hash(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()[:16]


def build_disjoint_splits(
    dataset: str, n_train: int, n_val: int, n_test: int, seed: int = 42
):
    """Draw disjoint train/val/test splits from the ML-flagged subset."""
    import pandas as pd

    pred_path = BASE / "ml_baseline" / "results" / f"predictions_{dataset}.csv"
    text_path = BASE / "data" / "processed" / "transactions_text" / f"{dataset}_text.jsonl"
    if not pred_path.exists() or not text_path.exists():
        raise FileNotFoundError(f"Missing predictions or text for {dataset}")

    preds = pd.read_csv(pred_path)
    texts = {}
    with open(text_path) as f:
        for i, line in enumerate(f):
            d = json.loads(line)
            texts[i] = d.get("text", "")

    fp_idx = preds[(preds["predicted"] == 1) & (preds["true_label"] == 0)].index.tolist()
    tp_idx = preds[(preds["predicted"] == 1) & (preds["true_label"] == 1)].index.tolist()

    # Deduplicate by text hash to prevent cross-split text collisions.
    # Keeps the first occurrence of each distinct text.
    def dedup_by_text(idx_list):
        seen_hashes = set()
        out = []
        for idx in idx_list:
            h = text_hash(texts.get(idx, ""))
            if h in seen_hashes:
                continue
            seen_hashes.add(h)
            out.append(idx)
        return out

    fp_idx = dedup_by_text(fp_idx)
    tp_idx = dedup_by_text(tp_idx)

    rng = random.Random(seed)
    rng.shuffle(fp_idx)
    rng.shuffle(tp_idx)

    total_needed = n_train + n_val + n_test
    if len(fp_idx) < total_needed // 2 or len(tp_idx) < total_needed // 2:
        raise ValueError(
            f"Insufficient pool: FP={len(fp_idx)}, TP={len(tp_idx)}, need {total_needed}"
        )

    def take_balanced(n, fp_pool, tp_pool):
        n_each = n // 2
        chosen_fp = fp_pool[:n_each]
        chosen_tp = tp_pool[:n_each]
        remaining_fp = fp_pool[n_each:]
        remaining_tp = tp_pool[n_each:]
        return chosen_fp + chosen_tp, remaining_fp, remaining_tp

    train_idx, fp_idx, tp_idx = take_balanced(n_train, fp_idx, tp_idx)
    val_idx, fp_idx, tp_idx = take_balanced(n_val, fp_idx, tp_idx)
    test_idx, _, _ = take_balanced(n_test, fp_idx, tp_idx)

    all_idx = set(train_idx) | set(val_idx) | set(test_idx)
    assert len(all_idx) == len(train_idx) + len(val_idx) + len(test_idx), (
        "split overlap detected!"
    )

    def to_records(indices):
        out = []
        for idx in indices:
            row = preds.iloc[idx]
            out.append({
                "row_index": int(idx),
                "transaction_id": int(row.get("TransactionID", idx)),
                "text": texts.get(idx, ""),
                "text_hash": text_hash(texts.get(idx, "")),
                "is_fraud": int(row["true_label"]),
                "category": "false_positive" if row["true_label"] == 0 else "true_positive",
                "fraud_score": float(row["fraud_score"]),
            })
        return out

    return {
        "train": to_records(train_idx),
        "val": to_records(val_idx),
        "test": to_records(test_idx),
    }


def verify_no_overlap(splits: dict) -> dict:
    """Verify that train/val/test sets have no overlapping text or transaction ID."""
    stats = {}
    for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
        a_ids = {r["transaction_id"] for r in splits[a]}
        b_ids = {r["transaction_id"] for r in splits[b]}
        a_texts = {r["text_hash"] for r in splits[a]}
        b_texts = {r["text_hash"] for r in splits[b]}
        stats[f"{a}_vs_{b}"] = {
            "id_overlap": len(a_ids & b_ids),
            "text_overlap": len(a_texts & b_texts),
        }
    return stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="ieee_cis")
    parser.add_argument("--n_train", type=int, default=160)
    parser.add_argument("--n_val", type=int, default=40)
    parser.add_argument("--n_test", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    splits = build_disjoint_splits(args.dataset, args.n_train, args.n_val, args.n_test, args.seed)
    overlap = verify_no_overlap(splits)

    for name, records in splits.items():
        print(f"{name}: {len(records)} (FP {sum(1 for r in records if r['is_fraud']==0)}, "
              f"TP {sum(1 for r in records if r['is_fraud']==1)})")
    print("\nOverlap audit (all should be zero):")
    for k, v in overlap.items():
        print(f"  {k}: id_overlap={v['id_overlap']}, text_overlap={v['text_overlap']}")

    assert all(v["id_overlap"] == 0 and v["text_overlap"] == 0
               for v in overlap.values()), "CONTAMINATION DETECTED"
    print("\n✓ All splits are disjoint (zero overlap).")

    FINETUNE_DIR.mkdir(parents=True, exist_ok=True)
    for name, records in splits.items():
        # raw manifest
        out_raw = FINETUNE_DIR / f"{name}_manifest.jsonl"
        with open(out_raw, "w") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"Wrote {out_raw}")

        # SFT-formatted training file (messages)
        out_sft = FINETUNE_DIR / f"{name}.jsonl"
        system_msg = (
            "You are a financial fraud analyst. Assess transactions for fraud risk "
            "by analyzing both suspicious and normal patterns. Provide balanced, "
            "evidence-based assessments."
        )
        with open(out_sft, "w") as f:
            for r in records:
                user_prompt = (
                    f"Assess this transaction for fraud risk.\n\n"
                    f"Transaction: {r['text']}\n"
                    f"ML Fraud Score: {r['fraud_score']:.4f}"
                )
                # Target label: use ground truth to supervise the model.
                if r["is_fraud"] == 0:
                    target = {
                        "fraud_likelihood": 0.2,
                        "recommendation": "RELEASE",
                        "rationale": "ML score appears inflated; feature pattern does not support escalation.",
                    }
                else:
                    target = {
                        "fraud_likelihood": 0.9,
                        "recommendation": "ESCALATE_AS_FRAUD",
                        "rationale": "Multiple indicators consistent with fraud; ML score corroborated.",
                    }
                messages = [
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_prompt},
                    {"role": "assistant", "content": json.dumps(target, ensure_ascii=False)},
                ]
                f.write(json.dumps({"messages": messages}, ensure_ascii=False) + "\n")
        print(f"Wrote {out_sft}")

    manifest = {
        "dataset": args.dataset,
        "seed": args.seed,
        "n_train": args.n_train,
        "n_val": args.n_val,
        "n_test": args.n_test,
        "splits": {name: [r["row_index"] for r in records] for name, records in splits.items()},
        "overlap_audit": overlap,
    }
    (FINETUNE_DIR / "split_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"Wrote {FINETUNE_DIR / 'split_manifest.json'}")


if __name__ == "__main__":
    main()
