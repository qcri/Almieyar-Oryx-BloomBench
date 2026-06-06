#!/usr/bin/env python3
"""
Step 1: Download MMMU benchmark and uniformly sample ~1000 questions
        across all 30 subjects.  Saves images to disk so Gemini can read them.
"""

import json, os, random, math
from pathlib import Path
from datasets import load_dataset, get_dataset_config_names

random.seed(42)

BASE = os.path.dirname(os.path.abspath(__file__))
OUT_JSON = os.path.join(BASE, "mmmu_sampled_1000.json")
IMG_DIR  = os.path.join(BASE, "mmmu_images")
os.makedirs(IMG_DIR, exist_ok=True)

TARGET_TOTAL = 1060          # aim for ~1000; 30 subjects → ~35 per subject
SUBJECTS = sorted(get_dataset_config_names("MMMU/MMMU"))
PER_SUBJECT = math.ceil(TARGET_TOTAL / len(SUBJECTS))   # 35-36

print(f"MMMU subjects: {len(SUBJECTS)},  target per subject: {PER_SUBJECT}")

all_samples = []

for subj in SUBJECTS:
    # Use validation (30 per subject) + test to have enough
    ds_val  = load_dataset("MMMU/MMMU", subj, split="validation")
    ds_test = load_dataset("MMMU/MMMU", subj, split="test")

    # Combine
    combined = list(ds_val) + list(ds_test)
    
    # Only keep items that have at least image_1 (PIL image)
    combined = [s for s in combined if s.get("image_1") is not None]

    # Uniform random sample
    n = min(PER_SUBJECT, len(combined))
    chosen = random.sample(combined, n)

    for item in chosen:
        # Save image_1 to disk
        img_fname = f"{item['id']}.png"
        img_path  = os.path.join(IMG_DIR, img_fname)
        if not os.path.exists(img_path):
            item["image_1"].save(img_path)

        rec = {
            "id":             item["id"],
            "subject":        subj,
            "subfield":       item.get("subfield", ""),
            "question":       item["question"],
            "options":        item["options"],
            "answer":         item.get("answer", ""),
            "question_type":  item.get("question_type", ""),
            "img_type":       item.get("img_type", []),
            "topic_difficulty": item.get("topic_difficulty", ""),
            "image_path":     os.path.join("mmmu_images", img_fname),
            # Count how many images the question references
            "num_images":     sum(1 for k in [f"image_{i}" for i in range(1, 8)]
                                  if item.get(k) is not None),
        }
        all_samples.append(rec)

    print(f"  {subj}: combined={len(combined)}, sampled={n}")

random.shuffle(all_samples)

with open(OUT_JSON, "w", encoding="utf-8") as f:
    json.dump(all_samples, f, indent=2, ensure_ascii=False)

print(f"\n✓ Saved {len(all_samples)} MMMU samples → {OUT_JSON}")
print(f"  Images dir: {IMG_DIR}")
