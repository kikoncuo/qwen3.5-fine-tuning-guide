# Fine-Tuning Qwen3.5 on Apple Silicon: Text & Vision

## The Problem

Fine-tuning a model usually means renting cloud GPUs, fighting CUDA drivers, and watching your credit card while a training run crawls through epochs on a machine you don't control. You can spend more time debugging SSH tunnels and OOM errors than actually training.

Apple Silicon changes this. The unified memory architecture means your Mac's RAM _is_ your VRAM — there's no data transfer bottleneck between CPU and GPU, no separate GPU memory to overflow. A 16 GB MacBook has 16 GB of usable model memory. With [MLX](https://github.com/ml-explore/mlx), Apple's framework for machine learning on their chips, you can LoRA fine-tune a model in minutes, on your laptop, for free.

## What This Guide Covers

We fine-tuned [Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B) — a vision-language model small enough to run on any Apple Silicon Mac — for two real tasks inside [keysay](https://github.com/kikoncuo/keysay), a macOS press-to-dictate app:

**Text correction** — A user dictates "es a las 5, no perdon, a las 6" and the model needs to understand the self-correction and output "es a las 6". The base model can't do this at all (0/12 test cases). After 5 minutes of LoRA training with `mlx_lm`, the fine-tuned model scores 12/12 — outperforming even the Gemini Flash teacher model that generated the training data.

**Screen context extraction** — While the user dictates, the app screenshots their screen and feeds it to the VLM to extract proper nouns, technical terms, and jargon. These context hints help the ASR model recognize words like "Kubernetes" or "Dr. Martinez" that it would otherwise mishear. We generated 206 synthetic screenshots with Gemini, engineered the extraction prompt through 5 iterations (cutting label noise by 50%), and trained with both `mlx-vlm` LoRA and `transformers` + PEFT on MPS.

## The Experiment

We wanted to answer a simple question: **can a 0.8B model learn a specialized skill that it completely fails at out of the box?**

The approach: use a large model (Gemini Flash) as a teacher to generate and validate training data, then distill that knowledge into the tiny model via LoRA. Keep training cheap and local.

```
Base Qwen3.5-0.8B:  "es a las 5, no perdón, a las 6"  →  "es a las 5, no perdón, a las 6"  (echoes input)
After 5 min LoRA:   "es a las 5, no perdón, a las 6"  →  "es a las 6"                       (correct)
```

| | Base model | Fine-tuned | Gemini Flash (teacher) |
|---|---|---|---|
| Test accuracy | 0/12 | **12/12** | 11/12 |
| Training cost | - | $0 (local) | - |
| Training time | - | 5 minutes | - |
| Peak RAM | 3 GB | 3.6 GB | API |
| Inference speed | ~300 tok/s | ~300 tok/s | ~50 tok/s (API) |

The fine-tuned 0.8B model outperforms its own teacher because knowledge distillation with teacher validation creates a perfectly consistent decision boundary — something the teacher itself doesn't always maintain.

---

## Table of Contents

- [Overview](#overview)
- [Requirements](#requirements)
- [Part 1: Text Correction (mlx_lm LoRA)](#part-1-text-correction-mlx_lm-lora)
  - [The Task](#the-task)
  - [Data Generation via Knowledge Distillation](#data-generation-via-knowledge-distillation)
  - [Dataset Formatting](#dataset-formatting)
  - [LoRA Training with mlx_lm](#lora-training-with-mlx_lm)
  - [Evaluation](#evaluation)
  - [Fusing & Uploading](#fusing--uploading)
  - [Inference](#inference)
- [Part 2: Screen Context Extraction (transformers + PEFT)](#part-2-screen-context-extraction-transformers--peft)
  - [The Task](#the-task-1)
  - [Synthetic Image Generation](#synthetic-image-generation)
  - [Labeling with Gemini](#labeling-with-gemini)
  - [Prompt Engineering](#prompt-engineering)
  - [Baseline Evaluation](#baseline-evaluation)
  - [LoRA Training with PEFT on MPS](#lora-training-with-peft-on-mps)
  - [MLX Conversion](#mlx-conversion)
- [Lessons Learned](#lessons-learned)
- [Models & Datasets](#models--datasets)

---

## Overview

| | Text Correction | Screen Context Extraction |
|---|---|---|
| **Base model** | `Qwen/Qwen3.5-0.8B` (via `mlx-community/Qwen3.5-0.8B-8bit`) | `Qwen/Qwen3.5-0.8B` |
| **Training framework** | `mlx_lm` (Apple MLX) | `transformers` + `peft` (PyTorch MPS) |
| **Task** | Clean ASR output: remove fillers, self-corrections | Extract proper nouns & jargon from screenshots |
| **Data source** | Gemini Flash knowledge distillation | Gemini image generation + Gemini labeling |
| **Dataset size** | 360 examples (v2) | 206 screenshot-label pairs |
| **Training time** | ~5 min on Apple Silicon | ~2 hours on MPS |
| **Result** | 12/12 test accuracy (vs 0/12 base) | In progress |

## Requirements

```bash
# Core
pip install mlx_lm mlx-vlm>=0.4.0

# For PEFT training (Part 2)
pip install torch torchvision peft accelerate

# For data generation
pip install python-dotenv requests datasets huggingface_hub

# API key for data generation
export OPENROUTER_KEY="sk-or-v1-..."
```

**Hardware**: Apple Silicon Mac (M1/M2/M3/M4) with 16+ GB RAM. Training uses 3-4 GB peak memory.

---

## Part 1: Text Correction (mlx_lm LoRA)

### The Task

Speech-to-text output is messy. Users frequently self-correct mid-sentence, use filler words, and make false starts. The correction model cleans this into paste-ready text:

```
Input:  "es a las 5, no perdón, a las 6 de la tarde"
Output: "es a las 6 de la tarde"

Input:  "I think we should um go with the uh first option"
Output: "I think we should go with the first option"

Input:  "the budget is 100, no wait, 200, actually 150 thousand"
Output: "the budget is 150 thousand"
```

**Why fine-tune?** The base Qwen3.5-0.8B model fails completely at this task — it either echoes input unchanged or gets stuck in thinking loops. Larger models (8B+) handle it but are too slow/heavy for real-time dictation on a laptop.

### Data Generation via Knowledge Distillation

We use **Gemini Flash as a teacher** to both generate and validate training examples. This is a two-step process:

#### Step 1: Generate raw examples

```python
# scripts/generate_training_data.py

GENERATION_CATEGORIES = [
    ("filler_removal", "FILLER WORD REMOVAL — Spanish and English fillers"),
    ("cross_sentence", "CROSS-SENTENCE CORRECTIONS — 'X. No, Y'"),
    ("nested_corrections", "NESTED AND COMPLEX CORRECTIONS"),
    ("realistic_long", "LONG REALISTIC MESSAGES (3-5 sentences)"),
    ("edge_cases", "EDGE CASES — 'no' that isn't a correction, passthrough"),
]
```

Each category generates ~40 examples via Gemini with structured JSON output:

```python
payload = {
    "model": "google/gemini-3-flash-preview",
    "messages": [
        {"role": "system", "content": "Generate training data for NLP models."},
        {"role": "user", "content": f"Generate {count} training pairs..."},
    ],
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
                        },
                    }
                },
            },
        },
    },
}
```

#### Step 2: Teacher validation

Every generated example is re-processed through Gemini with the **actual system prompt** used at inference time. The teacher's output replaces the generated output, ensuring the training data matches exactly what we want the model to produce:

```python
def validate_example(example, api_key):
    """Re-run input through Gemini teacher with production system prompt."""
    payload = {
        "model": "google/gemini-3-flash-preview",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": example["input"]},
        ],
        "temperature": 0.3,
    }
    # ... teacher output becomes ground truth
    return {"input": example["input"], "output": teacher_output}
```

**Why teacher validation?** Without it, the generated outputs may be inconsistent with the system prompt's instructions. The teacher validation step ensures a single, consistent decision-making process across all examples.

#### Language mix

- ~70% Spanish, ~30% English
- Mixed-language examples included (code-switching is common in dictation)

```bash
# Generate 200 examples
python3 scripts/generate_training_data.py --count 200 --output training_data_v2.json
```

### Dataset Formatting

Convert the JSON training data to JSONL chat format for `mlx_lm`:

```python
# scripts/format_dataset.py
# Each example becomes a chat conversation:
{
    "messages": [
        {"role": "system", "content": "You clean speech-to-text transcriptions..."},
        {"role": "user", "content": "es a las 5, no perdón, a las 6"},
        {"role": "assistant", "content": "es a las 6"}
    ]
}
```

Split into train/valid/test (85/8/7):

```bash
python3 scripts/format_dataset.py \
    --input training_data_v2.json \
    --output dataset_v2/

# Output:
#   train: 306 examples
#   valid: 28 examples
#   test: 26 examples
```

### LoRA Training with mlx_lm

```bash
python3 -m mlx_lm.lora \
    --model mlx-community/Qwen3.5-0.8B-8bit \
    --data dataset_v2 \
    --train \
    --iters 1000 \
    --batch-size 2 \
    --learning-rate 3e-5 \
    --lora-rank 8 \
    --steps-per-report 50 \
    --steps-per-eval 200 \
    --val-batches 25 \
    --adapter-path transcription-cleaner-lora-v2
```

**Training results:**

```
Iter      1: Val loss 1.727
Iter    200: Val loss 0.136  ← best
Iter    400: Val loss 0.176
Iter    600: Val loss 0.197
Iter    800: Val loss 0.159
Iter   1000: Val loss 0.159

Train loss: 1.727 → 0.005
Training time: ~5 minutes
Peak memory: 3.6 GB
```

**Key observations:**
- Val loss drops dramatically in the first 200 iterations
- The model converges quickly — this is a narrow, well-defined task
- Best checkpoint is at iter 200, though later checkpoints are close
- Overfitting is minimal thanks to the teacher-validated dataset

### Evaluation

We built a test suite of 12 hand-verified cases:

| Test Case | Base 0.8B | Fine-tuned 0.8B | Gemini Flash (teacher) |
|---|---|---|---|
| Simple correction (ES) | Echoes | **Correct** | Correct |
| Simple correction (EN) | Echoes | **Correct** | Correct |
| Cross-sentence correction | Fails | **Correct** | Correct |
| Time expression correction | Fails | **Correct** | Close |
| Nested corrections | Fails | **Correct** | Close |
| Subtle number correction | Fails | **Correct** | Correct |
| Double correction chain | Fails | **Correct** | Fails |
| Passthrough (no correction) | Partial | **Correct** | Correct |
| Filler removal (ES) | Fails | **Correct** | Correct |
| Filler removal (EN) | Fails | **Correct** | Correct |
| Mixed fillers + correction | Fails | **Correct** | Correct |
| False negative ("no" as word) | Fails | **Correct** | Correct |
| **Total** | **0/12** | **12/12** | **11/12** |

The fine-tuned 0.8B model **outperforms its teacher** (Gemini Flash) on edge cases like double corrections. This happens because the teacher-validated dataset creates a consistent decision boundary that Gemini itself sometimes violates.

### Fusing & Uploading

```bash
# Fuse LoRA adapter into base model
python3 -m mlx_lm.fuse \
    --model mlx-community/Qwen3.5-0.8B-8bit \
    --adapter-path transcription-cleaner-lora-v2 \
    --save-path keysay-transcription-cleaner-0.8B-8bit

# Upload to HuggingFace
huggingface-cli upload \
    Enriqueag26/keysay-transcription-cleaner-0.8B-8bit \
    keysay-transcription-cleaner-0.8B-8bit
```

### Inference

```python
from mlx_lm import load, generate

model, tokenizer = load("Enriqueag26/keysay-transcription-cleaner-0.8B-8bit")

SYSTEM_PROMPT = (
    "You clean speech-to-text transcriptions into ready-to-send chat messages.\n\n"
    "Remove self-corrections (keep only the final version). Remove filler words.\n"
    "Never rephrase. Never add words. Keep the original language."
)

messages = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user", "content": "es a las 5, no perdón, a las 6"},
]
prompt = tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True,
    enable_thinking=False,  # Important: disable thinking for Qwen3.5
)

result = generate(model, tokenizer, prompt=prompt, max_tokens=1024)
print(result)  # "es a las 6"
```

**Performance**: ~300 tokens/sec on Apple Silicon, ~3 GB RAM.

---

## Part 2: Screen Context Extraction (transformers + PEFT)

### The Task

When dictating, the user is looking at their screen. A screenshot can provide **context hints** — proper nouns, technical terms, product names — that help the ASR model recognize words it would otherwise mishear. For example, if the screen shows a Slack conversation about "Kubernetes v1.28", the ASR model is more likely to transcribe "Kubernetes" correctly.

The VLM reads a screenshot and outputs a comma-separated list of context terms.

### Synthetic Image Generation

We generate realistic macOS screenshots using **Gemini 3.1 Flash image preview** via OpenRouter. Each prompt describes a specific screen scenario:

```python
# scripts/generate_vlm_training_data.py

CATEGORIES = {
    "corporate_email": [  # ~30 prompts
        "un correo de Gmail corporativo sobre una guía médica de cardiología...",
        "an Outlook email thread about a patent filing for NeuralSync AI...",
    ],
    "chat_apps": [...],       # Slack, WhatsApp, Teams
    "code_editors": [...],    # VS Code, Xcode, IntelliJ
    "documents": [...],       # Google Docs, Word, PDF, Notion
    "browsers": [...],        # Chrome, Safari — docs, dashboards
    "spreadsheets": [...],    # Excel, Google Sheets
    "mixed": [...],           # Terminal, Finder, Calendar, Settings
}
```

**Parallel generation** with 6 workers via `ThreadPoolExecutor`:

```python
with ThreadPoolExecutor(max_workers=6) as pool:
    futures = {pool.submit(process_one, idx, cat, prompt): idx
               for idx, (cat, prompt) in enumerate(remaining)}
    for fut in as_completed(futures):
        entry = fut.result()
        # ... save incrementally
```

**Image extraction from OpenRouter response:**

```python
# OpenRouter returns generated images in msg["images"] field
message = result["choices"][0]["message"]
images = message.get("images", [])
if images:
    img_data = images[0]
    url = img_data["image_url"]["url"]  # "data:image/png;base64,..."
    b64 = url.split(",", 1)[1]
    img_bytes = base64.b64decode(b64)
```

**Results**: 206/210 images generated successfully, ~21 images/min with 6 workers, ~10 minutes total.

### Labeling with Gemini

Each generated screenshot is labeled by **Gemini 3 Flash** using the extraction prompt. The labels become ground truth for training.

### Prompt Engineering

The extraction prompt went through 5 iterations. The key challenge was getting the labeler to focus on **ASR-relevant terms** and exclude generic UI chrome:

| Version | Avg terms/image | Problem |
|---|---|---|
| v1 | 49.9 | Includes menus (File, Edit, View), common words |
| v2 | 24.9 | Better, but still some UI elements |
| v3 | ~25 | Inconsistent across categories |
| **v4 (final)** | **24.9** | Clean: no menus, focused on domain terms |

**v4 prompt (final):**

```
You are extracting speech recognition context hints from a screenshot.
These hints help an ASR model recognize uncommon words it might otherwise mishear.
EXTRACT ONLY terms that are clearly legible: people names, company/brand names,
product names, project/repository names, email addresses, URLs, domain-specific jargon,
technical terms, acronyms, specialized vocabulary, and proper nouns.
EXCLUDE: application menus and toolbars (File, Edit, View, Insertar, etc.),
generic UI labels (Inbox, Sent, Drafts, Compose, Search), window controls,
common everyday words, dates, timestamps, and pure numbers.
SKIP any text that appears blurry, garbled, or not clearly readable.
Output ONLY a comma-separated list. No explanations or categories.
```

**Impact of prompt improvement:**

| Category | v1 avg terms | v4 avg terms | Reduction |
|---|---|---|---|
| corporate_email | 43.9 | 18.7 | -57% |
| spreadsheets | 66.9 | 26.7 | -60% |
| code_editors | 61.6 | 32.1 | -48% |
| chat_apps | 39.3 | 21.5 | -45% |
| **Overall** | **49.9** | **24.9** | **-50%** |

### Baseline Evaluation

The base model (no fine-tuning) with the v4 prompt achieves:

```
Overall: P=0.677  R=0.366  F1=0.411  (n=206)
---------------------------------------------
  browsers          : P=0.649  R=0.386  F1=0.429
  chat_apps         : P=0.686  R=0.398  F1=0.436
  code_editors      : P=0.760  R=0.231  F1=0.321
  corporate_email   : P=0.730  R=0.421  F1=0.475
  documents         : P=0.631  R=0.413  F1=0.440
  mixed             : P=0.630  R=0.422  F1=0.427
  spreadsheets      : P=0.660  R=0.293  F1=0.349
```

**Key insight**: High precision (0.677) but low recall (0.366) — the base model extracts correct terms but misses many. Fine-tuning should teach it to be more exhaustive.

### LoRA Training with PEFT on MPS

#### Why not mlx_lm/mlx-vlm for VLM training?

We first attempted training with `mlx-vlm`'s built-in LoRA (`python3 -m mlx_vlm.lora`). After 3 attempts with different hyperparameters, all adapters **corrupted model generation** — the model would output only vision tokens (`<|vision_start|><|image_pad|><|vision_end|>`) instead of text. This appears to be a bug with Qwen3.5's DeltaRNN/linear attention architecture in mlx-vlm's LoRA implementation.

#### PEFT approach

We switched to **HuggingFace transformers + PEFT** running on PyTorch with the **MPS backend** (Metal Performance Shaders on Apple Silicon):

```python
# scripts/train_vlm_peft.py

from peft import LoraConfig, get_peft_model, TaskType
from transformers import AutoModelForImageTextToText, Trainer

# Load full-precision model for training
model = AutoModelForImageTextToText.from_pretrained(
    "Qwen/Qwen3.5-0.8B",
    torch_dtype=torch.float32,  # MPS doesn't support bf16
    trust_remote_code=True,
)

# LoRA config targeting language model attention
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    task_type=TaskType.CAUSAL_LM,
)

model = get_peft_model(model, lora_config)
# trainable params: 3,194,880 || all params: 856,180,800 || trainable%: 0.3732
```

**Data collator** handles multimodal image+text input:

```python
class VLMDataCollator:
    def __call__(self, examples):
        for ex in examples:
            messages = [
                {"role": "user", "content": [
                    {"type": "image", "image": ex["image"]},
                    {"type": "text", "text": ex["question"]},
                ]},
                {"role": "assistant", "content": [
                    {"type": "text", "text": ex["answer"]},
                ]},
            ]
            text = self.processor.apply_chat_template(messages, ...)

        # Mask labels: only train on assistant response
        # Find <|im_start|>assistant token and mask everything before it
        labels[i, :assistant_start] = -100
```

**Training arguments:**

```python
TrainingArguments(
    num_train_epochs=3,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=2e-5,
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    fp16=False, bf16=False,  # MPS requires float32
    dataloader_pin_memory=False,  # Required for MPS
)
```

**Training speed**: ~15-50s per step (varies with image size), ~2 hours total for 156 steps (3 epochs).

### MLX Conversion

After training, merge the adapter and convert back to MLX format:

```python
# Merge adapter into base model
merged_model = model.merge_and_unload()
merged_model.save_pretrained("vlm_training/peft-fused")
processor.save_pretrained("vlm_training/peft-fused")
```

```bash
# Convert to MLX format with 8-bit quantization
python3 -m mlx_vlm.convert \
    --hf-path vlm_training/peft-fused \
    --mlx-path keysay-vlm-context-0.8B-8bit \
    -q 8

# Upload to HuggingFace
huggingface-cli upload \
    Enriqueag26/keysay-vlm-context-0.8B-8bit \
    keysay-vlm-context-0.8B-8bit
```

---

## Lessons Learned

### What worked well

1. **Knowledge distillation** — Using Gemini Flash as a teacher to generate AND validate training data produced a highly consistent dataset. The student (0.8B) actually outperformed its teacher on edge cases.

2. **mlx_lm for text-only LoRA** — Training is incredibly fast (~5 min), memory-efficient (3.6 GB peak), and the resulting model runs at ~300 tok/s. Apple's MLX framework is excellent for on-device fine-tuning of text models.

3. **Parallel API calls for data generation** — Using `ThreadPoolExecutor` with 6 workers for image generation + labeling achieved 21 images/min vs 5/min sequential (~4x speedup).

4. **Iterative prompt engineering** — 5 rounds of prompt refinement reduced ground truth noise by 50%, making the training signal much cleaner. This was more impactful than any model change.

### What didn't work

1. **mlx-vlm LoRA on Qwen3.5 VLM** — All 3 attempts with different hyperparameters corrupted model generation. The adapter caused the model to output only vision tokens. This appears to be a bug with Qwen3.5's DeltaRNN architecture in mlx-vlm v0.4.0.

2. **Small `max-seq-length` for VLM training** — Setting `--max-seq-length 512` with mlx-vlm meant image tokens consumed the entire budget, leaving ~23 tokens for the actual answer. The model wasn't learning the task at all.

### Key takeaways

- **Start with the prompt, not the model.** Our v4 prompt improvement alone boosted F1 from 0.350 to 0.411 — a bigger gain than many fine-tuning runs.

- **Teacher validation > raw generation.** Re-processing every example through the teacher with the production prompt eliminates distribution mismatch between training and inference.

- **Small models can outperform large ones on narrow tasks.** The 0.8B model achieves 12/12 accuracy on transcription cleaning where even Gemini Flash gets 11/12 — because the fine-tuned model has a perfectly consistent decision boundary.

- **When one framework fails, try another.** mlx-vlm LoRA broke VLM generation, but transformers + PEFT on MPS works correctly with the same model. The frameworks have different LoRA implementations.

- **Always verify adapter output before benchmarking.** A quick sanity check on 3 images would have saved hours of broken evaluation runs.

---

## Models & Datasets

### Published Models

| Model | Task | HuggingFace |
|---|---|---|
| Transcription cleaner | Text correction | [`Enriqueag26/keysay-transcription-cleaner-0.8B-8bit`](https://huggingface.co/Enriqueag26/keysay-transcription-cleaner-0.8B-8bit) |
| VLM context extractor | Screen extraction | [`Enriqueag26/keysay-vlm-context-0.8B-8bit`](https://huggingface.co/Enriqueag26/keysay-vlm-context-0.8B-8bit) (pending) |

### Datasets

| Dataset | Format | HuggingFace |
|---|---|---|
| VLM training images | image + question + answer | [`Enriqueag26/keysay-vlm-context-training`](https://huggingface.co/datasets/Enriqueag26/keysay-vlm-context-training) |

### Base Model

- **Qwen3.5-0.8B**: [`Qwen/Qwen3.5-0.8B`](https://huggingface.co/Qwen/Qwen3.5-0.8B)
- **MLX 8-bit**: [`mlx-community/Qwen3.5-0.8B-8bit`](https://huggingface.co/mlx-community/Qwen3.5-0.8B-8bit)

---

## License

This guide and associated scripts are released under the MIT License.

The Qwen3.5 model is licensed under [Apache 2.0](https://huggingface.co/Qwen/Qwen3.5-0.8B/blob/main/LICENSE).
