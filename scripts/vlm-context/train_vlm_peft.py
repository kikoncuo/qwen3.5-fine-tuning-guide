"""LoRA fine-tune Qwen3.5-0.8B VLM using transformers + PEFT on Mac (MPS).

After training, merges adapter and converts to MLX format.
"""

import json
import logging
import os
import sys
import time
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, TaskType
from PIL import Image
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    Trainer,
    TrainingArguments,
    BitsAndBytesConfig,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("train_peft")

MODEL_ID = "Qwen/Qwen3.5-0.8B"
HF_DATASET = "Enriqueag26/keysay-vlm-context-training"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "vlm_training" / "peft-lora"
FUSED_DIR = Path(__file__).resolve().parent.parent / "vlm_training" / "peft-fused"

EXTRACT_PROMPT = (
    "You are extracting speech recognition context hints from a screenshot. "
    "These hints help an ASR model recognize uncommon words it might otherwise mishear. "
    "EXTRACT ONLY terms that are clearly legible: people names, company/brand names, "
    "product names, project/repository names, email addresses, URLs, domain-specific jargon, "
    "technical terms, acronyms, specialized vocabulary, and proper nouns. "
    "EXCLUDE: application menus and toolbars (File, Edit, View, Insertar, etc.), "
    "generic UI labels (Inbox, Sent, Drafts, Compose, Search), window controls, "
    "common everyday words, dates, timestamps, and pure numbers. "
    "SKIP any text that appears blurry, garbled, or not clearly readable. "
    "Output ONLY a comma-separated list. No explanations or categories."
)


def build_chat_messages(question: str, answer: str, image: Image.Image):
    """Build chat messages in Qwen VL format."""
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": question},
            ],
        },
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": answer},
            ],
        },
    ]


class VLMDataCollator:
    """Collate function that processes images + text for Qwen3.5 VLM."""

    def __init__(self, processor):
        self.processor = processor

    def __call__(self, examples):
        texts = []
        images_list = []

        for ex in examples:
            messages = build_chat_messages(ex["question"], ex["answer"], ex["image"])

            # Apply chat template to get the full text
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            texts.append(text)
            images_list.append([ex["image"]])

        # Process all examples
        batch = self.processor(
            text=texts,
            images=images_list,
            padding=True,
            truncation=True,
            max_length=2048,
            return_tensors="pt",
        )

        # Create labels (same as input_ids, with padding masked)
        labels = batch["input_ids"].clone()
        labels[labels == self.processor.tokenizer.pad_token_id] = -100

        # Mask everything before the assistant's response
        # Find the assistant token position and mask everything before it
        assistant_token = self.processor.tokenizer.encode(
            "<|im_start|>assistant", add_special_tokens=False
        )
        for i in range(len(labels)):
            ids = batch["input_ids"][i].tolist()
            # Find assistant start
            for j in range(len(ids) - len(assistant_token)):
                if ids[j : j + len(assistant_token)] == assistant_token:
                    # Mask everything up to and including the newline after assistant tag
                    mask_end = j + len(assistant_token) + 1  # +1 for \n
                    labels[i, :mask_end] = -100
                    break

        batch["labels"] = labels
        return batch


def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    log.info("Device: %s", device)
    log.info("Loading model: %s", MODEL_ID)

    # Load processor
    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)

    # Load model in float32 for MPS (no quantization on MPS)
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float32,
        trust_remote_code=True,
    )

    log.info("Model loaded. Params: %.1fM", sum(p.numel() for p in model.parameters()) / 1e6)

    # Configure LoRA - target the language model's attention layers
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        task_type=TaskType.CAUSAL_LM,
        modules_to_save=None,
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Load dataset
    log.info("Loading dataset: %s", HF_DATASET)
    dataset = load_dataset(HF_DATASET, split="train")
    log.info("Dataset size: %d", len(dataset))

    # Training arguments
    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        num_train_epochs=3,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=2e-5,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        logging_steps=10,
        save_steps=50,
        save_total_limit=2,
        fp16=False,
        bf16=False,  # MPS doesn't support bf16
        dataloader_pin_memory=False,  # Required for MPS
        remove_unused_columns=False,
        report_to="none",
        max_grad_norm=1.0,
    )

    # Data collator
    collator = VLMDataCollator(processor)

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collator,
    )

    log.info("Starting training...")
    t0 = time.time()
    trainer.train()
    elapsed = time.time() - t0
    log.info("Training completed in %.0fs (%.1f min)", elapsed, elapsed / 60)

    # Save adapter
    log.info("Saving adapter to %s", OUTPUT_DIR)
    model.save_pretrained(str(OUTPUT_DIR))
    processor.save_pretrained(str(OUTPUT_DIR))

    # Merge adapter into base model
    log.info("Merging adapter into base model...")
    merged_model = model.merge_and_unload()
    FUSED_DIR.mkdir(parents=True, exist_ok=True)
    merged_model.save_pretrained(str(FUSED_DIR))
    processor.save_pretrained(str(FUSED_DIR))
    log.info("Fused model saved to %s", FUSED_DIR)

    print(f"\nDone! Next steps:")
    print(f"1. Convert to MLX: python3 -m mlx_vlm.convert --hf-path {FUSED_DIR} --mlx-path keysay-vlm-context-0.8B-8bit -q 8")
    print(f"2. Test: python3 -m mlx_vlm.generate --model keysay-vlm-context-0.8B-8bit --prompt 'test' --image vlm_training/images/0000.png")
    print(f"3. Upload: huggingface-cli upload Enriqueag26/keysay-vlm-context-0.8B-8bit keysay-vlm-context-0.8B-8bit")


if __name__ == "__main__":
    main()
