#!/usr/bin/env python3
"""
Step 1: Extract all AGREED multiple-choice QA samples from judge output,
merge with the full QA dataset, and save as a single JSON.
Only keeps items that have MC questions AND valid image files on disk.
"""

import json
import os

BASE = os.path.dirname(os.path.abspath(__file__))
EVAL_PATH = os.path.join(BASE, "gemini3_oryx_evaluation_CLEANED.json")
QA_PATH = os.path.join(BASE, "final_oryx_v2.json")
OUT_PATH = os.path.join(BASE, "agreed_mc_samples.json")


def main():
    # ── Load judge evaluation ─────────────────────────────────────────
    with open(EVAL_PATH, "r", encoding="utf-8") as f:
        eval_data = json.load(f)

    agreed_ids = set()
    for level_key, level_data in eval_data["results_by_level"].items():
        for item in level_data["evaluations"]:
            if item["judgments"]["overall"] == "AGREE":
                agreed_ids.add(item["item_id"])
    print(f"Total AGREE items from judge: {len(agreed_ids)}")

    # ── Load full QA dataset ──────────────────────────────────────────
    with open(QA_PATH, "r", encoding="utf-8") as f:
        qa_data = json.load(f)
    qa_map = {d["question_id"]: d for d in qa_data}

    # ── Filter: must have valid image on disk ─────────────────────────
    agreed_samples = []
    skipped_no_img = 0

    for qid in sorted(agreed_ids):
        item = qa_map[qid]
        
        img_rel_path = item["source_image_file"]
        img_path = os.path.join(BASE, img_rel_path)
        
        # Check if exists, or try fallback image_1.jpg if it was image_0.jpg
        if not os.path.exists(img_path):
            if "image_0.jpg" in img_rel_path:
                fallback = img_rel_path.replace("image_0.jpg", "image_1.jpg")
                if os.path.exists(os.path.join(BASE, fallback)):
                    item["source_image_file"] = fallback
                    img_path = os.path.join(BASE, fallback)
            
        if not os.path.exists(img_path):
            skipped_no_img += 1
            print(f"Skipping {qid}: Image not found at {img_rel_path}")
            continue

        agreed_samples.append(item)

    print(f"Kept valid-image: {len(agreed_samples)} / {len(agreed_ids)}")
    print(f"Skipped (missing image): {skipped_no_img}")
    
    mc_count = sum(1 for it in agreed_samples if it.get("multiple_choice_qa"))
    print(f"Items with MC: {mc_count}")
    print(f"Items without MC: {len(agreed_samples) - mc_count}")

    # ── Save ──────────────────────────────────────────────────────────
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(agreed_samples, f, ensure_ascii=False, indent=2)
    print(f"Saved → {OUT_PATH}")


if __name__ == "__main__":
    main()
