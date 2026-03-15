#!/usr/bin/env python3
"""Generate text correction training data via Gemini distillation.

Usage:
    export OPENROUTER_KEY="sk-or-v1-..."
    python3 scripts/generate_training_data.py --count 200 --output training_data_v3.json
"""

import argparse
import json
import os
import subprocess
import sys
import time

SYSTEM_PROMPT = """You clean speech-to-text transcriptions into ready-to-send chat messages.

Two operations:

1. Self-corrections: when the speaker says X then corrects to Y (signals: "no", "no no", "perdón", "wait", "actually"), ONLY replace X with Y. Keep everything before and after the correction unchanged.
   - This can cross sentence boundaries: "X. No, Y" or "X. No no, Y" means Y replaces X.

2. Fillers: delete um, uh, eh, vale (at start only), pues, mira, you know, like, basically

"bueno" or "o sea" before a rephrasing = correction signal.

Never rephrase. Never add words. Keep original language.
Output the cleaned message only."""

GENERATION_CATEGORIES = [
    ("filler_removal", "FILLER WORD REMOVAL — Spanish and English fillers at start, middle, end of sentences. Include passthrough cases."),
    ("cross_sentence", "CROSS-SENTENCE CORRECTIONS — corrections that span a period/sentence boundary: 'X. No, Y'"),
    ("nested_corrections", "NESTED AND COMPLEX CORRECTIONS — multiple corrections in one sentence, o sea + no perdón chains"),
    ("realistic_long", "LONG REALISTIC MESSAGES (3-5 sentences) with corrections embedded naturally in medical, business, casual contexts"),
    ("edge_cases", "EDGE CASES — 'no' that isn't a correction, passthrough, very short messages, numbers, names, emotions"),
]


def generate_batch(category: str, description: str, count: int, api_key: str) -> list[dict]:
    prompt = f"""Generate {count} training pairs for a speech transcription cleaner.
Category: {description}
Spanish (70%) and English (30%).
Format each as {{"input": "...", "output": "..."}}"""

    payload = {
        "model": "google/gemini-3-flash-preview",
        "messages": [
            {"role": "system", "content": "Generate training data for NLP models."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 1.0,
        "max_tokens": 16000,
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "data",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "examples": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "input": {"type": "string"},
                                    "output": {"type": "string"},
                                },
                                "required": ["input", "output"],
                                "additionalProperties": False,
                            },
                        }
                    },
                    "required": ["examples"],
                    "additionalProperties": False,
                },
            },
        },
    }

    r = subprocess.run(
        ["curl", "-s", "https://openrouter.ai/api/v1/chat/completions",
         "-H", "Content-Type: application/json",
         "-H", f"Authorization: Bearer {api_key}",
         "-d", json.dumps(payload)],
        capture_output=True, text=True, timeout=120,
    )
    data = json.loads(r.stdout)
    return json.loads(data["choices"][0]["message"]["content"])["examples"]


def validate_example(example: dict, api_key: str) -> dict:
    """Validate an example through Gemini teacher."""
    payload = {
        "model": "google/gemini-3-flash-preview",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": example["input"]},
        ],
        "temperature": 0.3,
        "max_tokens": 2000,
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "c",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {"cleaned_text": {"type": "string"}},
                    "required": ["cleaned_text"],
                    "additionalProperties": False,
                },
            },
        },
    }

    r = subprocess.run(
        ["curl", "-s", "https://openrouter.ai/api/v1/chat/completions",
         "-H", "Content-Type: application/json",
         "-H", f"Authorization: Bearer {api_key}",
         "-d", json.dumps(payload)],
        capture_output=True, text=True, timeout=30,
    )
    resp = json.loads(r.stdout)
    teacher_out = json.loads(resp["choices"][0]["message"]["content"])["cleaned_text"]
    return {"input": example["input"], "output": teacher_out}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=200, help="Total examples to generate")
    parser.add_argument("--output", default="training_data_new.json")
    parser.add_argument("--api-key", default=os.environ.get("OPENROUTER_KEY"))
    parser.add_argument("--merge-with", help="Existing dataset to merge with")
    args = parser.parse_args()

    if not args.api_key:
        print("Error: Set OPENROUTER_KEY or pass --api-key")
        sys.exit(1)

    per_category = args.count // len(GENERATION_CATEGORIES)
    all_examples = []

    for cat_name, cat_desc in GENERATION_CATEGORIES:
        print(f"Generating {per_category} examples for '{cat_name}'...")
        examples = generate_batch(cat_name, cat_desc, per_category, args.api_key)
        print(f"  Got {len(examples)} examples")
        all_examples.extend(examples)
        time.sleep(1)

    print(f"\nValidating {len(all_examples)} examples through Gemini teacher...")
    validated = []
    for i, ex in enumerate(all_examples):
        try:
            validated.append(validate_example(ex, args.api_key))
        except Exception:
            validated.append(ex)
        if (i + 1) % 50 == 0:
            print(f"  Validated {i+1}/{len(all_examples)}")

    # Merge if requested
    if args.merge_with and os.path.exists(args.merge_with):
        with open(args.merge_with) as f:
            prev = json.load(f)
        validated = prev + validated

    # Deduplicate
    seen = set()
    final = []
    for ex in validated:
        key = ex["input"].strip().lower()
        if key not in seen and ex["input"].strip() and ex["output"].strip():
            seen.add(key)
            final.append(ex)

    with open(args.output, "w") as f:
        json.dump(final, f, indent=2, ensure_ascii=False)

    print(f"\nSaved {len(final)} unique examples to {args.output}")


if __name__ == "__main__":
    main()
