#!/usr/bin/env python3
"""Format a JSON training dataset into JSONL for mlx_lm LoRA training.

Usage:
    python3 scripts/format_dataset.py --input training_data.json --output dataset_dir
"""

import argparse
import json
import os
import random


SYSTEM_PROMPT = """You clean speech-to-text transcriptions into ready-to-send chat messages.

Remove self-corrections (keep only the final version). Remove filler words.
Never rephrase. Never add words. Keep the original language."""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input JSON file")
    parser.add_argument("--output", required=True, help="Output directory for JSONL files")
    parser.add_argument("--train-ratio", type=float, default=0.85)
    parser.add_argument("--valid-ratio", type=float, default=0.08)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    with open(args.input) as f:
        data = json.load(f)

    examples = []
    for ex in data:
        examples.append({
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": ex["input"]},
                {"role": "assistant", "content": ex["output"]},
            ]
        })

    random.seed(args.seed)
    random.shuffle(examples)

    n = len(examples)
    train_end = int(n * args.train_ratio)
    valid_end = int(n * (args.train_ratio + args.valid_ratio))

    splits = {
        "train": examples[:train_end],
        "valid": examples[train_end:valid_end],
        "test": examples[valid_end:],
    }

    os.makedirs(args.output, exist_ok=True)
    for name, split in splits.items():
        path = os.path.join(args.output, f"{name}.jsonl")
        with open(path, "w") as f:
            for ex in split:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"Formatted {n} examples:")
    for name, split in splits.items():
        print(f"  {name}: {len(split)}")
    print(f"Output: {args.output}/")


if __name__ == "__main__":
    main()
