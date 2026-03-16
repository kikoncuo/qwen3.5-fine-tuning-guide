# Fine-Tuning Qwen3.5 on Apple Silicon: Text & Vision

## The Problem

Fine-tuning usually means renting cloud GPUs, fighting CUDA drivers, and watching your credit card. Apple Silicon changes this — your Mac's RAM _is_ your VRAM. No data transfers, no OOM surprises. With [MLX](https://github.com/ml-explore/mlx), LoRA fine-tuning takes minutes, on your laptop, for free.

## What This Guide Covers

We fine-tuned [Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B) for two tasks inside [keysay](https://github.com/kikoncuo/keysay), a macOS dictation app:

**Text correction** — Clean messy ASR output. The base model scores 0/12. After 5 min of LoRA: 12/12.

**Screen context extraction** — Read screenshots, extract terms that help ASR. 206 synthetic images, 5 prompt iterations, two training frameworks compared.

## Results at a Glance

```
Base model:   "es a las 5, no perdón, a las 6"  →  echoes input unchanged
Fine-tuned:   "es a las 5, no perdón, a las 6"  →  "es a las 6"
```

| | Base | Fine-tuned | Gemini Flash (teacher) |
|---|---|---|---|
| Accuracy | 0/12 | **12/12** | 11/12 |
| Cost | - | $0 | API fees |
| Training | - | 5 min | - |
| RAM | 3 GB | 3.6 GB | - |
| Speed | ~300 tok/s | ~300 tok/s | ~50 tok/s |

The student outperforms its teacher. Knowledge distillation with teacher validation creates a consistent decision boundary the teacher itself doesn't maintain.

---

## Requirements

```bash
pip install mlx_lm mlx-vlm>=0.4.0           # MLX training + inference
pip install torch torchvision peft accelerate # PEFT training (Part 2)
pip install python-dotenv requests datasets   # Data generation
export OPENROUTER_KEY="sk-or-v1-..."          # For Gemini API calls
```

Apple Silicon Mac, 16+ GB RAM. Training peaks at 3-4 GB.

---

## Part 1: Text Correction

### The Task

```
"es a las 5, no perdón, a las 6 de la tarde"     →  "es a las 6 de la tarde"
"I think we should um go with the uh first option" →  "I think we should go with the first option"
"the budget is 100, no wait, 200, actually 150k"   →  "the budget is 150k"
```

Remove self-corrections, fillers, false starts. Keep everything else. The base 0.8B model can't do this — it echoes input or loops. Bigger models work but are too heavy for real-time dictation.

### Pipeline

```mermaid
graph LR
    A[Gemini generates<br/>raw examples] --> B[Gemini validates<br/>with prod prompt]
    B --> C[Format to<br/>JSONL chat]
    C --> D[mlx_lm LoRA<br/>5 min training]
    D --> E[Fuse + upload<br/>to HuggingFace]
```

### Data Generation

Two-step knowledge distillation using Gemini Flash:

**Generate** — Gemini creates input/output pairs across 5 categories:

| Category | Examples | Description |
|---|---|---|
| Filler removal | ~40 | Spanish & English fillers |
| Cross-sentence | ~40 | "X. No, Y" corrections |
| Nested | ~40 | Multiple corrections chained |
| Realistic long | ~40 | 3-5 sentence messages |
| Edge cases | ~40 | "no" as a word, passthrough |

**Validate** — Every example re-processed through Gemini with the _production system prompt_. Teacher output replaces the generated output:

```python
# The key insight: validate with the EXACT prompt used at inference
def validate_example(example, api_key):
    payload = {
        "model": "google/gemini-3-flash-preview",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},  # Same prompt as prod
            {"role": "user", "content": example["input"]},
        ],
        "temperature": 0.3,
    }
    return {"input": example["input"], "output": teacher_output}
```

This eliminates distribution mismatch between training and inference. ~70% Spanish, ~30% English.

```bash
python3 scripts/generate_training_data.py --count 200 --output training_data_v2.json
```

### Dataset Format

```python
# Each example → chat conversation in JSONL
{"messages": [
    {"role": "system", "content": "You clean speech-to-text transcriptions..."},
    {"role": "user", "content": "es a las 5, no perdón, a las 6"},
    {"role": "assistant", "content": "es a las 6"}
]}
```

```bash
python3 scripts/format_dataset.py --input training_data_v2.json --output dataset_v2/
# → train: 306 | valid: 28 | test: 26
```

### Training

```bash
python3 -m mlx_lm.lora \
    --model mlx-community/Qwen3.5-0.8B-8bit \
    --data dataset_v2 \
    --train \
    --iters 1000 \
    --batch-size 2 \
    --learning-rate 3e-5 \
    --lora-rank 8 \
    --steps-per-eval 200 \
    --adapter-path transcription-cleaner-lora
```

**Loss curve:**

```
Iter    1 → Val loss 1.727
Iter  200 → Val loss 0.136  ← converges fast
Iter  400 → Val loss 0.176
Iter 1000 → Val loss 0.159

Train loss: 1.727 → 0.005 | 5 min | 3.6 GB peak
```

The model converges in ~200 iters. This is a narrow, well-defined task.

### Evaluation

| Test Case | Base | Fine-tuned | Teacher |
|---|---|---|---|
| Simple correction (ES/EN) | Echoes | **Pass** | Pass |
| Cross-sentence correction | Fail | **Pass** | Pass |
| Nested corrections | Fail | **Pass** | Close |
| Double correction chain | Fail | **Pass** | **Fail** |
| Passthrough (no correction) | Partial | **Pass** | Pass |
| Filler removal (ES/EN) | Fail | **Pass** | Pass |
| False negative ("no" as word) | Fail | **Pass** | Pass |
| **Total** | **0/12** | **12/12** | **11/12** |

The 12 test cases were **hand-written and never seen during training** — they're not in the generated dataset. The fine-tuned model beats its teacher on double corrections because consistent training data creates a cleaner decision boundary than the teacher has.

### Deploy

```bash
# Fuse adapter into base model
python3 -m mlx_lm.fuse \
    --model mlx-community/Qwen3.5-0.8B-8bit \
    --adapter-path transcription-cleaner-lora \
    --save-path keysay-transcription-cleaner-0.8B-8bit

# Upload
huggingface-cli upload Enriqueag26/keysay-transcription-cleaner-0.8B-8bit \
    keysay-transcription-cleaner-0.8B-8bit
```

**Inference:**

```python
from mlx_lm import load, generate

model, tokenizer = load("Enriqueag26/keysay-transcription-cleaner-0.8B-8bit")

messages = [
    {"role": "system", "content": "You clean speech-to-text transcriptions..."},
    {"role": "user", "content": "es a las 5, no perdón, a las 6"},
]
prompt = tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True,
    enable_thinking=False,  # Important for Qwen3.5
)
result = generate(model, tokenizer, prompt=prompt, max_tokens=1024)
# → "es a las 6"
```

~300 tok/s, ~3 GB RAM on Apple Silicon.

---

## Part 2: Screen Context Extraction

### The Task

The user dictates while looking at their screen. A screenshot provides context hints — proper nouns, jargon, product names — that help ASR recognize words like "Kubernetes" or "Dr. Martinez".

```
Screenshot of Slack chat about PostgreSQL migration
    → "Sarah Chen, PostgreSQL 16, pg_dump, pg_restore, David Kim, EKS, Helm"
```

### Pipeline

```mermaid
graph LR
    A[Gemini generates<br/>206 screenshots] --> B[Gemini labels<br/>each image]
    B --> C[Iterate prompt<br/>5 versions]
    C --> D[Upload dataset<br/>to HuggingFace]
    D --> E[PEFT LoRA<br/>on MPS]
    E --> F[Convert to MLX<br/>+ upload]
```

### Image Generation

Synthetic macOS screenshots via **Gemini 3.1 Flash image preview**, 7 categories:

| Category | Count | Examples |
|---|---|---|
| Corporate email | 30 | Gmail/Outlook — medical, legal, finance |
| Chat apps | 30 | Slack/WhatsApp/Teams — projects, jargon |
| Code editors | 30 | VS Code/Xcode — functions, imports |
| Documents | 30 | Docs/Word/PDF — reports, contracts |
| Browsers | 29 | Chrome/Safari — docs, dashboards |
| Spreadsheets | 30 | Excel/Sheets — financial, metrics |
| Mixed | 27 | Terminal, Finder, Calendar |

6 parallel workers, 21 images/min, 206/210 successful, ~10 min total.

**OpenRouter image extraction gotcha** — images come in `msg["images"]`, not `msg["content"]`:

```python
message = result["choices"][0]["message"]
images = message.get("images", [])  # Not in "content"!
url = images[0]["image_url"]["url"]  # "data:image/png;base64,..."
```

### Prompt Engineering

The labeling prompt went through 5 iterations. The problem: Gemini dumps everything including menus.

```mermaid
graph LR
    A[v1: 49.9 terms<br/>includes File/Edit/View] --> B[v2: 24.9 terms<br/>still some UI]
    B --> C[v3: ~25 terms<br/>inconsistent]
    C --> D[v4: 24.9 terms<br/>clean, focused]
```

**Before (v1):** `File, Edit, Selection, View, Go, Run, Terminal, Window, Help, data_analysis_workflow, EXPLORER, import numpy...` (90 terms for a code editor)

**After (v4):** `SQLAlchemy, ORM, User, Column, ForeignKey, relationship, Post, backref` (27 terms)

| Category | v1 | v4 | Change |
|---|---|---|---|
| spreadsheets | 66.9 | 26.7 | **-60%** |
| corporate_email | 43.9 | 18.7 | **-57%** |
| code_editors | 61.6 | 32.1 | **-48%** |
| **Overall** | **49.9** | **24.9** | **-50%** |

The v4 prompt:

```
You are extracting speech recognition context hints from a screenshot.
EXTRACT ONLY: people names, company/brand names, product names,
project names, email addresses, URLs, technical jargon, acronyms, proper nouns.
EXCLUDE: application menus (File, Edit, View), generic UI labels,
window controls, common words, dates, numbers.
SKIP blurry or garbled text.
Comma-separated list only.
```

### Baseline

Base model + v4 prompt, no fine-tuning:

```
Overall:  P=0.677  R=0.366  F1=0.411

Best:     corporate_email  F1=0.475
Worst:    code_editors     F1=0.321 (low recall)
```

High precision, low recall — the model extracts correct terms but misses many.

### Training: mlx-vlm vs PEFT

We tried two frameworks. mlx-vlm had three bugs we had to find and fix. PEFT worked and produced the final model.

#### mlx-vlm LoRA — three bugs found

Every mlx-vlm training run corrupted generation — the model output only vision tokens. We investigated with 4 parallel agent teams and found three independent bugs:

**Bug 1: Images discarded during training.** `VisionDataset.process()` sets `images=None` for Qwen/Gemma/SmolVLM models. Training runs without any image data — only 23 tokens per sample instead of ~1,076.

**Bug 2: LoRA scaling 8x too large.** `LoRaLayer` uses raw `alpha` instead of `alpha/rank`. With alpha=16, rank=8: mlx-vlm applies 16x scaling, PEFT applies 2x. The perturbation is 8x larger than intended.

**Bug 3: LoRA on DeltaRNN gate projections.** Qwen3.5 is 75% GatedDeltaNet layers. The `in_proj_a` (decay gate) feeds into a double exponential: `g = exp(-exp(A_log) * softplus(a))`. Even small LoRA perturbations push the gate from ~0.95 to 1.0 (state explosion) or 0.5 (amnesia). No normalization layer between the projection and the gate. Additionally, `in_proj_a` has output dim=16 — with rank=8, LoRA covers half the output space.

We filed [issue #824](https://github.com/Blaizzy/mlx-vlm/issues/824) and [PR #823](https://github.com/Blaizzy/mlx-vlm/pull/823) with fixes for all three bugs. With all fixes applied, mlx-vlm training produces non-corrupted output for ~1 epoch but degrades beyond that. The DeltaNet recurrence amplifies perturbations across the sequence — a fundamental architectural sensitivity.

#### transformers + PEFT on MPS (final approach)

PEFT gives control over which modules get LoRA. Qwen3.5's hybrid architecture has two layer types with different module names:

| Standard attention (25% of layers) | GatedDeltaNet (75% of layers) |
|---|---|
| `q_proj`, `k_proj`, `v_proj`, `o_proj` | `in_proj_qkv`, `in_proj_z`, `out_proj` |

We target both, plus MLP layers, but exclude the gate projections (`in_proj_a`, `in_proj_b`):

```python
lora_config = LoraConfig(
    r=8, lora_alpha=16, lora_dropout=0.05,
    target_modules=[
        # Standard attention (25% of layers)
        "q_proj", "k_proj", "v_proj", "o_proj",
        # GatedDeltaNet (75% of layers) — exclude in_proj_a, in_proj_b
        "in_proj_qkv", "in_proj_z", "out_proj",
        # MLP (all layers)
        "gate_proj", "up_proj", "down_proj",
    ],
    task_type=TaskType.CAUSAL_LM,
)
# trainable: 5.4M / 856M = 0.63%
```

**MPS-specific settings:**

```python
TrainingArguments(
    fp16=False, bf16=False,       # MPS doesn't support bf16
    dataloader_pin_memory=False,  # Required for MPS
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=2e-5,
    lr_scheduler_type="cosine",
    num_train_epochs=3,
)
```

3 epochs, ~7 hours on MPS. Loss: 1.1 → 0.85.

**Label masking** — only train on the assistant response:

```python
# Find <|im_start|>assistant and mask everything before it
labels[i, :assistant_start] = -100
```

### Results

| Metric | Base model | Fine-tuned | Change |
|---|---|---|---|
| **Precision** | 0.677 | **0.731** | +8% |
| **Recall** | 0.366 | **0.596** | +63% |
| **F1** | 0.411 | **0.609** | **+48%** |

Per-category F1:

| Category | Base | Fine-tuned | Change |
|---|---|---|---|
| corporate_email | 0.475 | **0.693** | +46% |
| chat_apps | 0.436 | **0.659** | +51% |
| mixed | 0.427 | **0.626** | +47% |
| browsers | 0.429 | **0.607** | +41% |
| documents | 0.440 | **0.573** | +30% |
| spreadsheets | 0.349 | **0.566** | +62% |
| code_editors | 0.321 | **0.538** | +68% |

The main gain is recall — the model now captures 60% of ground truth terms instead of 37%. Code editors and spreadsheets saw the biggest improvements.

### Convert to MLX

```python
# Merge adapter
merged = model.merge_and_unload()
merged.save_pretrained("vlm_training/peft-fused")
```

```bash
# Convert + quantize to 8-bit MLX
python3 -m mlx_vlm convert \
    --hf-path vlm_training/peft-fused \
    --mlx-path keysay-vlm-context-0.8B-8bit \
    -q --q-bits 8
```

Final model: 1.2 GB, ~1s per screenshot on Apple Silicon. Same speed as the base model.

---

## Lessons Learned

### What worked

**Knowledge distillation** — Gemini generates + validates data. Student outperforms teacher on edge cases. Consistent training data > smarter teacher.

**mlx_lm for text LoRA** — 5 min training, 3.6 GB peak, 300 tok/s inference. Hard to beat for text-only fine-tuning on Mac.

**PEFT for VLM LoRA** — Full control over target modules. Essential for Qwen3.5's hybrid architecture where some projections are unsafe to adapt.

**Parallel data generation** — 6 workers, 21 img/min vs 5/min sequential. Always parallelize API calls.

**Prompt engineering first** — 5 rounds of prompt refinement: F1 0.350 → 0.411. A bigger gain than many training runs.

**Agent-team debugging** — 4 parallel investigation agents found the root causes in ~5 minutes. One analyzed source code, one searched GitHub issues, one researched DeltaRNN architecture, one inspected adapter weights.

### What didn't work

**mlx-vlm LoRA on Qwen3.5** — Three bugs: images discarded, 8x scaling error, DeltaRNN gate instability. We fixed all three and [submitted upstream](https://github.com/Blaizzy/mlx-vlm/pull/823), but the architecture is fundamentally sensitive to LoRA perturbations beyond 1 epoch.

**Standard LoRA target modules on hybrid architectures** — If you target only `q_proj`/`k_proj`/`v_proj`, you adapt only 25% of Qwen3.5's attention layers. The other 75% (GatedDeltaNet) use `in_proj_qkv`, `in_proj_z`, `out_proj`. And you must exclude the gate projections (`in_proj_a`, `in_proj_b`) or the recurrent state diverges.

### Takeaways

- **Start with the prompt.** Our prompt improvement alone was worth +17% F1.
- **Teacher validation > raw generation.** Same prompt at training and inference = no distribution mismatch.
- **Small models beat large ones on narrow tasks.** 0.8B gets 12/12 where Gemini gets 11/12.
- **Know your architecture.** Qwen3.5's DeltaRNN layers need different LoRA targets than standard transformers. Check `model.named_modules()` before training.
- **When one framework fails, try another.** mlx-vlm broke; PEFT on MPS worked. Same model, same task, different LoRA implementations.
- **Sanity-check adapter output on 3 examples before running full benchmarks.**

---

## Models & Datasets

| Resource | Link |
|---|---|
| Fine-tuned text model | [`Enriqueag26/keysay-transcription-cleaner-0.8B-8bit`](https://huggingface.co/Enriqueag26/keysay-transcription-cleaner-0.8B-8bit) |
| Fine-tuned VLM | [`Enriqueag26/keysay-vlm-context-0.8B-8bit`](https://huggingface.co/Enriqueag26/keysay-vlm-context-0.8B-8bit) |
| VLM training dataset | [`Enriqueag26/keysay-vlm-context-training`](https://huggingface.co/datasets/Enriqueag26/keysay-vlm-context-training) |
| Base model | [`Qwen/Qwen3.5-0.8B`](https://huggingface.co/Qwen/Qwen3.5-0.8B) |
| Base model (MLX 8-bit) | [`mlx-community/Qwen3.5-0.8B-8bit`](https://huggingface.co/mlx-community/Qwen3.5-0.8B-8bit) |
| mlx-vlm bug report | [Issue #824](https://github.com/Blaizzy/mlx-vlm/issues/824) |
| mlx-vlm fixes PR | [PR #823](https://github.com/Blaizzy/mlx-vlm/pull/823) |

---

## License

MIT. Qwen3.5 is [Apache 2.0](https://huggingface.co/Qwen/Qwen3.5-0.8B/blob/main/LICENSE).
