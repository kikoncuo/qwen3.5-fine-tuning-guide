"""Format VLM training data and upload to HuggingFace as a dataset.

Usage:
  python3 scripts/format_vlm_dataset.py
"""

import json
import os
from pathlib import Path

from datasets import Dataset, Features, Image, Value
from dotenv import load_dotenv
from huggingface_hub import login
from PIL import Image as PILImage

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

DATA_DIR = Path(__file__).resolve().parent.parent / "vlm_training"
DATA_PATH = DATA_DIR / "training_data.json"

HF_DATASET_ID = "Enriqueag26/keysay-vlm-context-training"

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


def main():
    # Login to HuggingFace
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        login(token=hf_token)
    else:
        print("Warning: HF_TOKEN not set, using cached credentials")

    # Load training data
    with open(DATA_PATH) as f:
        data = json.load(f)

    print(f"Loaded {len(data)} examples")

    # Build dataset columns
    images = []
    questions = []
    answers = []

    for entry in data:
        img_path = DATA_DIR / entry["image"]
        if not img_path.exists():
            print(f"  Warning: Image not found: {img_path}, skipping")
            continue

        images.append(str(img_path))
        questions.append(EXTRACT_PROMPT)
        answers.append(entry["labels"])

    print(f"Valid examples: {len(images)}")

    # Create HuggingFace dataset
    dataset = Dataset.from_dict(
        {
            "image": images,
            "question": questions,
            "answer": answers,
        },
        features=Features({
            "image": Image(),
            "question": Value("string"),
            "answer": Value("string"),
        }),
    )

    print(f"Dataset created: {dataset}")
    print(f"Columns: {dataset.column_names}")
    print(f"First example answer: {dataset[0]['answer'][:100]}...")

    # Upload to HuggingFace
    print(f"\nUploading to {HF_DATASET_ID}...")
    dataset.push_to_hub(HF_DATASET_ID, private=False)
    print(f"Dataset uploaded: https://huggingface.co/datasets/{HF_DATASET_ID}")

    # Also save the custom prompt format file for mlx-vlm training
    prompt_format = {
        "user": [
            {"type": "image", "image": "{image}"},
            {"type": "text", "text": "{question}"},
        ],
        "assistant": [
            {"type": "text", "text": "{answer}"},
        ],
    }
    format_path = DATA_DIR / "prompt_format.json"
    with open(format_path, "w") as f:
        json.dump(prompt_format, f, indent=2)
    print(f"Prompt format saved to: {format_path}")


if __name__ == "__main__":
    main()
