"""Benchmark VLM context extraction: baseline and post-training evaluation.

Usage:
  python3 scripts/benchmark_vlm.py --baseline
  python3 scripts/benchmark_vlm.py --adapter vlm_training/vlm-context-lora
"""

import argparse
import json
import sys
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from keysay.llm._patches import apply_transformers_patches
apply_transformers_patches()

DATA_DIR = Path(__file__).resolve().parent.parent / "vlm_training"
DATA_PATH = DATA_DIR / "training_data.json"

EXTRACT_PROMPT = (
    "Focus on the MAIN CONTENT area of the screen (ignore browser tabs, "
    "bookmarks bar, and navigation chrome). Read all text in the main "
    "content: headings, paragraphs, labels, form fields, cards, lists, "
    "sidebar content, and dialog text. "
    "Extract: email addresses, URLs, people names, company names, product "
    "names, medical terms, acronyms, abbreviations, technical jargon, "
    "project names, phone numbers, addresses, and proper nouns. "
    "Include full phrases when they contain specialized vocabulary. "
    "Output ONLY a comma-separated list. Be exhaustive."
)


def normalize_terms(csv_string: str) -> set[str]:
    """Parse comma-separated terms into a normalized set."""
    terms = set()
    for part in csv_string.replace("\n", ",").split(","):
        term = part.strip().strip(".-\"'()[]{}")
        if term and len(term) >= 2:
            terms.add(term.lower())
    return terms


def compute_metrics(predicted: set[str], ground_truth: set[str]) -> dict:
    """Compute precision, recall, F1 between predicted and ground truth term sets."""
    if not predicted and not ground_truth:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    if not predicted:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    if not ground_truth:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    # Fuzzy matching: a predicted term matches if it's a substring of any GT term or vice versa
    matched_pred = set()
    matched_gt = set()
    for p in predicted:
        for g in ground_truth:
            if p in g or g in p or p == g:
                matched_pred.add(p)
                matched_gt.add(g)

    precision = len(matched_pred) / len(predicted) if predicted else 0
    recall = len(matched_gt) / len(ground_truth) if ground_truth else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {"precision": precision, "recall": recall, "f1": f1}


def run_benchmark(model_id: str, adapter_path: str | None = None):
    """Run benchmark on all training images."""
    from mlx_vlm import generate, load
    from mlx_vlm.prompt_utils import apply_chat_template

    print(f"Loading model: {model_id}")
    if adapter_path:
        print(f"With adapter: {adapter_path}")
        model, processor = load(model_id, adapter_path=adapter_path)
    else:
        model, processor = load(model_id)

    with open(DATA_PATH) as f:
        data = json.load(f)

    print(f"Evaluating on {len(data)} images...")

    results_by_category = {}
    all_results = []

    for i, entry in enumerate(data):
        img_path = str(DATA_DIR / entry["image"])

        prompt = apply_chat_template(
            processor,
            config=model.config,
            prompt=EXTRACT_PROMPT,
            num_images=1,
        )

        result = generate(
            model,
            processor,
            prompt=prompt,
            image=img_path,
            max_tokens=500,
            verbose=False,
        )

        raw = result.text if hasattr(result, "text") else str(result)
        predicted = normalize_terms(raw)
        ground_truth = normalize_terms(entry["labels"])
        metrics = compute_metrics(predicted, ground_truth)

        category = entry["category"]
        if category not in results_by_category:
            results_by_category[category] = []
        results_by_category[category].append(metrics)
        all_results.append(metrics)

        if (i + 1) % 10 == 0:
            avg_f1 = sum(r["f1"] for r in all_results) / len(all_results)
            print(f"  [{i + 1}/{len(data)}] Running avg F1: {avg_f1:.3f}")

    # Aggregate metrics
    def avg_metrics(metrics_list):
        n = len(metrics_list)
        return {
            "precision": sum(m["precision"] for m in metrics_list) / n,
            "recall": sum(m["recall"] for m in metrics_list) / n,
            "f1": sum(m["f1"] for m in metrics_list) / n,
            "count": n,
        }

    summary = {
        "model": model_id,
        "adapter": adapter_path,
        "overall": avg_metrics(all_results),
        "by_category": {cat: avg_metrics(ms) for cat, ms in results_by_category.items()},
    }

    # Print results
    print("\n" + "=" * 60)
    print(f"Model: {model_id}")
    if adapter_path:
        print(f"Adapter: {adapter_path}")
    print(f"Overall: P={summary['overall']['precision']:.3f}  "
          f"R={summary['overall']['recall']:.3f}  "
          f"F1={summary['overall']['f1']:.3f}  "
          f"(n={summary['overall']['count']})")
    print("-" * 60)
    for cat, m in sorted(summary["by_category"].items()):
        print(f"  {cat:20s}: P={m['precision']:.3f}  R={m['recall']:.3f}  "
              f"F1={m['f1']:.3f}  (n={m['count']})")
    print("=" * 60)

    # Save results
    suffix = "adapter" if adapter_path else "baseline"
    out_path = DATA_DIR / f"{suffix}_results.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Results saved to: {out_path}")

    return summary


def main():
    parser = argparse.ArgumentParser(description="Benchmark VLM context extraction")
    parser.add_argument("--baseline", action="store_true", help="Run baseline benchmark")
    parser.add_argument("--adapter", type=str, help="Path to LoRA adapter for evaluation")
    parser.add_argument("--model", type=str, default="mlx-community/Qwen3.5-0.8B-8bit",
                        help="Base model ID")
    args = parser.parse_args()

    if not args.baseline and not args.adapter:
        parser.error("Must specify --baseline or --adapter PATH")

    if not DATA_PATH.exists():
        print(f"Error: Training data not found at {DATA_PATH}")
        print("Run generate_vlm_training_data.py first.")
        sys.exit(1)

    run_benchmark(args.model, adapter_path=args.adapter)


if __name__ == "__main__":
    main()
