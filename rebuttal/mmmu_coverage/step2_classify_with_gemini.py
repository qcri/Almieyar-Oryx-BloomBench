#!/usr/bin/env python3
"""
Step 2: Use Gemini Flash to classify each MMMU question+image into one
        of our 106 Bloom-bench leaf taxonomy categories.

For each sample Gemini receives:
  - the image
  - the question text
  - a NUMBERED list of 106 leaf categories (with parent path for context)

Gemini must return ONLY the number (1-106) or 0 if nothing fits.
Using numbers avoids truncated leaf-name issues.

Results are saved incrementally so the script can be resumed.
"""

import json, os, sys, time, re, traceback
from pathlib import Path

import google.generativeai as genai
from PIL import Image

BASE = os.path.dirname(os.path.abspath(__file__))
REABUTTLE = os.path.dirname(BASE)  # parent Reabuttle dir

MMMU_JSON   = os.path.join(BASE, "mmmu_sampled_1000.json")
EVAL_JSON   = os.path.join(REABUTTLE, "gemini3_oryx_evaluation_CLEANED.json")
OUTPUT_JSON = os.path.join(BASE, "mmmu_classified.json")

# ── Configure Gemini ──────────────────────────────────────────────────
api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY", "")
if not api_key:
    raise RuntimeError("Set GOOGLE_API_KEY or GEMINI_API_KEY")
genai.configure(api_key=api_key)
model = genai.GenerativeModel("gemini-3-flash-preview")

# ── Build taxonomy ────────────────────────────────────────────────────
def load_taxonomy():
    with open(EVAL_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)
    leaves = []
    for level_key, level_data in data["results_by_level"].items():
        full_path = level_data["level"]
        leaf = level_data["leaf"]
        leaves.append({"full_path": full_path, "leaf": leaf})
    return leaves

TAXONOMY = load_taxonomy()

# Build numbered taxonomy string
TAXONOMY_STR = "\n".join(
    f"{i+1}. {t['leaf']}  [{t['full_path']}]"
    for i, t in enumerate(TAXONOMY)
)

CLASSIFY_PROMPT = """You are an expert in visual question answering taxonomy classification.

I have a benchmark called "Bloom-Bench" with 106 leaf-level categories in a cognitive hierarchy.
Each category is numbered and shown with its full hierarchy path in brackets:

{taxonomy}

Now I will give you a question (with its image) from a DIFFERENT benchmark (MMMU).
Your task: Decide which ONE of the 106 categories above best matches this question+image.

Rules:
- Return ONLY the category NUMBER (1-106).
- If the question clearly does not fit ANY of our categories, return 0.
- Return ONLY a single integer, nothing else. No explanation, no text.

Question: {question}
"""


def classify_sample(sample, img_path):
    """Send image + question to Gemini, get leaf number."""
    prompt = CLASSIFY_PROMPT.format(
        taxonomy=TAXONOMY_STR,
        question=sample["question"][:2000],
    )

    img = Image.open(img_path)

    for attempt in range(4):
        try:
            response = model.generate_content(
                [prompt, img],
                generation_config=genai.types.GenerationConfig(
                    temperature=0.0,
                    max_output_tokens=20,
                ),
            )
            text = response.text.strip()
            # Extract number
            nums = re.findall(r"\d+", text)
            if nums:
                num = int(nums[0])
                if 0 <= num <= len(TAXONOMY):
                    if num == 0:
                        return "NONE", 0
                    return TAXONOMY[num - 1]["leaf"], num
            # If we got text that looks like a leaf name, try to match
            return text, -1
        except Exception as e:
            err_str = str(e).lower()
            if "429" in str(e) or "quota" in err_str or "resource" in err_str:
                wait = 30 * (attempt + 1)
                print(f"    Rate limited, waiting {wait}s …")
                time.sleep(wait)
            elif "block" in err_str or "safety" in err_str:
                print(f"    Blocked by safety filter")
                return "BLOCKED", -2
            else:
                print(f"    Gemini error: {e}")
                time.sleep(5)
    return "ERROR", -3


def main():
    # Load MMMU samples
    with open(MMMU_JSON, "r", encoding="utf-8") as f:
        samples = json.load(f)
    print(f"Loaded {len(samples)} MMMU samples")

    # Load existing results for resume
    if os.path.exists(OUTPUT_JSON):
        with open(OUTPUT_JSON, "r", encoding="utf-8") as f:
            results = json.load(f)
        done_ids = set(r["id"] for r in results)
        print(f"Resuming — {len(done_ids)} already classified")
    else:
        results = []
        done_ids = set()

    total = len(samples)
    for idx, sample in enumerate(samples):
        if sample["id"] in done_ids:
            continue

        # Resolve image path
        img_path = os.path.join(BASE, "mmmu_images", f"{sample['id']}.png")
        if not os.path.exists(img_path):
            img_path = os.path.join(REABUTTLE, sample["image_path"])
        if not os.path.exists(img_path):
            print(f"  [{idx+1}/{total}] SKIP {sample['id']} — image missing")
            results.append({**sample, "predicted_leaf": "IMAGE_MISSING", "predicted_num": -4})
            continue

        print(f"  [{idx+1}/{total}] {sample['id']} ({sample['subject']})", end=" ", flush=True)
        pred_leaf, pred_num = classify_sample(sample, img_path)
        print(f"→ [{pred_num}] {pred_leaf}")

        results.append({
            **sample,
            "predicted_leaf": pred_leaf,
            "predicted_num": pred_num,
        })

        # Save incrementally every 10 items
        if len(results) % 10 == 0:
            with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)

        # Small delay to respect rate limits
        time.sleep(0.3)

    # Final save
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # Quick summary
    from collections import Counter
    preds = Counter(r.get("predicted_leaf", "?") for r in results)
    none_count = sum(1 for r in results if r.get("predicted_leaf") in ("NONE", "BLOCKED", "ERROR", "IMAGE_MISSING"))
    matched_leaves = set(r["predicted_leaf"] for r in results if r.get("predicted_num", 0) > 0)
    print(f"\n✓ Classification complete. {len(results)} results → {OUTPUT_JSON}")
    print(f"  Unique leaves matched: {len(matched_leaves)} / {len(TAXONOMY)}")
    print(f"  NONE/ERROR/BLOCKED: {none_count}")
    print(f"  Top 10 predictions: {preds.most_common(10)}")


if __name__ == "__main__":
    main()
