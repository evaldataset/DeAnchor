"""Canonical LLM input subset generation with explicit stratified sampling.

Produces `llm_input_<dataset>_<n>.jsonl` with a documented manifest listing
exact transaction IDs so that all downstream experiments are traceable.

Usage:
    python scripts/generate_llm_inputs.py --dataset ieee_cis --n_fp 50 --n_tp 50 --seed 42

Outputs:
    experiments/results/llm_input_<dataset>_<N>.jsonl
    experiments/results/llm_input_<dataset>_<N>.manifest.json
"""

import argparse
import json
import random
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
RESULTS = BASE / "experiments" / "results"


def load_predictions(dataset: str):
    """Load OOF predictions + transaction text for a dataset."""
    pred_path = BASE / "ml_baseline" / "results" / f"predictions_{dataset}.csv"
    text_path = BASE / "data" / "processed" / "transactions_text" / f"{dataset}_text.jsonl"
    if not pred_path.exists() or not text_path.exists():
        raise FileNotFoundError(f"Missing predictions or text for {dataset}")

    import pandas as pd
    preds = pd.read_csv(pred_path)
    texts = {}
    with open(text_path) as f:
        for i, line in enumerate(f):
            d = json.loads(line)
            texts[i] = d.get("text", "")
    return preds, texts


def stratified_sample(dataset: str, n_fp: int, n_tp: int, seed: int = 42):
    """Draw a stratified FP/TP subset from the ML-flagged region."""
    preds, texts = load_predictions(dataset)

    # FP = predicted=1, true=0; TP = predicted=1, true=1
    fp_idx = preds[(preds["predicted"] == 1) & (preds["true_label"] == 0)].index.tolist()
    tp_idx = preds[(preds["predicted"] == 1) & (preds["true_label"] == 1)].index.tolist()

    rng = random.Random(seed)
    if len(fp_idx) < n_fp or len(tp_idx) < n_tp:
        raise ValueError(
            f"Insufficient samples: FP pool={len(fp_idx)}, TP pool={len(tp_idx)}, "
            f"requested {n_fp} FP + {n_tp} TP"
        )

    chosen_fp = rng.sample(fp_idx, n_fp)
    chosen_tp = rng.sample(tp_idx, n_tp)
    combined = [(i, "false_positive") for i in chosen_fp] + [(i, "true_positive") for i in chosen_tp]
    rng.shuffle(combined)

    items = []
    for idx, category in combined:
        row = preds.iloc[idx]
        items.append({
            "transaction_id": int(row.get("TransactionID", idx)),
            "row_index": int(idx),
            "text": texts.get(idx, ""),
            "category": category,
            "is_fraud": int(row["true_label"]),
            "fraud_score": float(row["fraud_score"]),
        })
    return items


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="ieee_cis")
    parser.add_argument("--n_fp", type=int, default=50)
    parser.add_argument("--n_tp", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_tag", default=None,
                        help="optional suffix; default: total count")
    args = parser.parse_args()

    items = stratified_sample(args.dataset, args.n_fp, args.n_tp, args.seed)

    n_total = len(items)
    tag = args.output_tag or f"{n_total}"
    out_jsonl = RESULTS / f"llm_input_{args.dataset}_{tag}.jsonl"
    out_manifest = RESULTS / f"llm_input_{args.dataset}_{tag}.manifest.json"

    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with open(out_jsonl, "w") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    manifest = {
        "dataset": args.dataset,
        "n_fp": args.n_fp,
        "n_tp": args.n_tp,
        "seed": args.seed,
        "sampling": "stratified (FP/TP) without replacement, shuffled",
        "row_indices": [(it["row_index"], it["category"]) for it in items],
        "category_counts": {
            "false_positive": sum(1 for it in items if it["category"] == "false_positive"),
            "true_positive": sum(1 for it in items if it["category"] == "true_positive"),
        },
    }
    with open(out_manifest, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Wrote {n_total} samples to {out_jsonl}")
    print(f"Manifest: {out_manifest}")
    print(f"Balance: {manifest['category_counts']}")


if __name__ == "__main__":
    main()
