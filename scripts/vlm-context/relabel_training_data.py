"""Re-label all training images with the improved v4 prompt.

Parallel execution, incremental saves, preserves image/category/prompt metadata.
"""

import base64
import json
import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("relabel")

OPENROUTER_KEY = os.environ["OPENROUTER_KEY"]
DATA_DIR = Path(__file__).resolve().parent.parent / "vlm_training"
DATA_PATH = DATA_DIR / "training_data.json"
BACKUP_PATH = DATA_DIR / "training_data_v1_backup.json"

MAX_WORKERS = 8

EXTRACT_PROMPT_V4 = (
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


def label_image(image_path: str) -> str | None:
    with open(image_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode()

    for attempt in range(3):
        try:
            resp = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {OPENROUTER_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": "google/gemini-3-flash-preview",
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image_url",
                                    "image_url": {"url": f"data:image/png;base64,{img_b64}"},
                                },
                                {"type": "text", "text": EXTRACT_PROMPT_V4},
                            ],
                        }
                    ],
                    "max_tokens": 1024,
                },
                timeout=120,
            )
            if resp.status_code == 429:
                wait = 2 ** attempt * 5
                log.warning("Rate limited, waiting %ds (retry %d/3)", wait, attempt + 1)
                time.sleep(wait)
                continue
            if resp.status_code >= 500:
                wait = 2 ** attempt * 3
                log.warning("Server error %d, waiting %ds (retry %d/3)", resp.status_code, wait, attempt + 1)
                time.sleep(wait)
                continue
            resp.raise_for_status()
            text = resp.json()["choices"][0]["message"].get("content", "")
            if isinstance(text, list):
                text = " ".join(
                    p.get("text", "") for p in text
                    if isinstance(p, dict) and p.get("type") == "text"
                )
            return text.strip() if text and text.strip() else None
        except requests.exceptions.Timeout:
            log.warning("Timeout for %s (retry %d/3)", image_path, attempt + 1)
        except requests.exceptions.ConnectionError as e:
            log.warning("Connection error for %s: %s (retry %d/3)", image_path, e, attempt + 1)
            time.sleep(2)
        except Exception as e:
            log.error("Unexpected error for %s: %s", image_path, e)
            return None

    log.error("Failed after 3 retries: %s", image_path)
    return None


_lock = threading.Lock()


def process_entry(entry: dict) -> dict | None:
    idx = entry["index"]
    img_path = str(DATA_DIR / entry["image"])

    if not os.path.exists(img_path):
        log.warning("[%04d] Image missing: %s", idx, img_path)
        return None

    labels = label_image(img_path)
    if labels is None:
        log.warning("[%04d] Labeling failed", idx)
        return None

    terms = [t.strip() for t in labels.replace("\n", ",").split(",") if t.strip()]
    log.info("[%04d|%s] %d terms: %.100s", idx, entry["category"], len(terms), labels)

    return {**entry, "labels": labels}


def main():
    with open(DATA_PATH) as f:
        data = json.load(f)

    # Backup original
    if not BACKUP_PATH.exists():
        import shutil
        shutil.copy2(DATA_PATH, BACKUP_PATH)
        log.info("Backed up original to %s", BACKUP_PATH.name)

    log.info("Re-labeling %d images with v4 prompt, %d workers", len(data), MAX_WORKERS)

    relabeled = []
    failed = 0
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(process_entry, entry): entry["index"] for entry in data}

        for fut in as_completed(futures):
            idx = futures[fut]
            try:
                result = fut.result()
            except Exception as e:
                log.error("[%04d] Unexpected error: %s", idx, e)
                failed += 1
                continue

            if result is None:
                failed += 1
            else:
                with _lock:
                    relabeled.append(result)

                    done = len(relabeled) + failed
                    if done % 10 == 0:
                        elapsed = time.time() - t0
                        rate = done / elapsed * 60
                        log.info(
                            "Progress: %d/%d (ok=%d fail=%d) — %.1f/min",
                            done, len(data), len(relabeled), failed, rate,
                        )

                    # Save every 20
                    if len(relabeled) % 20 == 0:
                        sorted_data = sorted(relabeled, key=lambda x: x["index"])
                        with open(DATA_PATH, "w") as f:
                            json.dump(sorted_data, f, indent=2, ensure_ascii=False)

    # Final save
    relabeled.sort(key=lambda x: x["index"])
    with open(DATA_PATH, "w") as f:
        json.dump(relabeled, f, indent=2, ensure_ascii=False)

    elapsed = time.time() - t0
    log.info(
        "DONE. %d relabeled, %d failed. Time: %.0fs (%.1f/min)",
        len(relabeled), failed, elapsed, len(data) / elapsed * 60,
    )


if __name__ == "__main__":
    main()
