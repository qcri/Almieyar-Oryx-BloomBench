import argparse
import json
import os
import random
import re
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import google.generativeai as genai
from PIL import Image
from tqdm import tqdm

CLEANED_DATASET = Path("/fanar-image-understanding-01/OryxTrain/OryxSegments/fanar3/Omid/Bloom-bench-acl/judge/results/cleaned_VQA_dataset.json")
OUTPUT_FILE = Path("/fanar-image-understanding-01/OryxTrain/OryxSegments/fanar3/Omid/Bloom-bench-acl/judge/results/balanced_VQA_dataset.json")
DATASET_DIR = Path("/fanar-image-understanding-01/OryxTrain/OryxSegments/fanar3/Omid/Bloom-bench-acl/Dataset/bloombench-crawling/Updated-dataset")

TEMP_DIR = "/fanar-image-understanding-01/OryxTrain/OryxSegments/fanar3/tmp"
os.makedirs(TEMP_DIR, exist_ok=True)
os.environ["TMPDIR"] = TEMP_DIR
os.environ["TEMP"] = TEMP_DIR
os.environ["TMP"] = TEMP_DIR

TARGET_REMEMBER = 3500
TARGET_UNDERSTAND = 3500
TARGET_OTHER = 2000
MAX_RETRIES = 4
NUM_THREADS = 12

TARGETS = {
    "Remember": TARGET_REMEMBER,
    "Understand": TARGET_UNDERSTAND,
    "Apply": TARGET_OTHER,
    "Analyze": TARGET_OTHER,
    "Evaluate": TARGET_OTHER,
    "Create": TARGET_OTHER,
}

GENERATABLE_LEVELS = {"Analyze", "Evaluate", "Create"}
LEVELS = ["Remember", "Understand", "Apply", "Analyze", "Evaluate", "Create"]


def parse_level_from_lvl1(lvl1: str) -> str:
    if not lvl1:
        return ""
    match = re.match(r"^\s*(Remember|Understand|Apply|Analyze|Evaluate|Create)\b", str(lvl1))
    return match.group(1) if match else ""


def extract_hierarchy_from_source(path_obj: Path, is_create_folder: bool = False) -> dict:
    folder_name = path_obj.name if is_create_folder else path_obj.parent.name

    if " - " in folder_name:
        idx = folder_name.rfind(" - ")
        lvl1 = folder_name[:idx].strip()
        leaf = folder_name[idx + 3 :].strip()
    else:
        lvl1 = folder_name.strip()
        leaf = folder_name.strip()

    bloom_level = parse_level_from_lvl1(lvl1) or lvl1
    return {
        "lvl1": lvl1,
        "leaf": leaf,
        "bloom_level": bloom_level,
        "path": f"{lvl1} -> {leaf}",
    }


def clean_json_response(text: str) -> str:
    text = (text or "").strip()
    text = text.replace('\\"', "'")

    if "```json" in text:
        text = text.split("```json", 1)[1].split("```", 1)[0]
    elif "```" in text:
        text = text.split("```", 1)[1].split("```", 1)[0]

    try:
        start_idx = text.index("{")
        depth = 0
        in_string = False
        escape_next = False
        end_idx = -1

        for i in range(start_idx, len(text)):
            char = text[i]
            if escape_next:
                escape_next = False
                continue
            if char == "\\":
                escape_next = True
                continue
            if char == '"':
                in_string = not in_string
            if not in_string:
                if char == "{":
                    depth += 1
                elif char == "}":
                    depth -= 1
                    if depth == 0:
                        end_idx = i + 1
                        break

        if end_idx > start_idx:
            text = text[start_idx:end_idx]
    except ValueError:
        pass

    text = re.sub(r",(\s*[}\]])", r"\1", text)

    open_braces = text.count("{") - text.count("}")
    if open_braces > 0:
        text += "}" * open_braces

    text = text.rstrip()
    if text and not text.endswith("}"):
        text = text.rstrip(",") + "}"

    return text.strip()


def get_image_prompt(hierarchy: dict, num_qa: int) -> str:
    bloom_level = hierarchy.get("bloom_level", "Analyze")
    leaf = hierarchy.get("leaf", "visual content")
    path = hierarchy.get("path", "")
    return f"""You are an expert assistant for building VQA benchmark data.
Generate EXACTLY {num_qa} QA pairs from the provided image.

Constraints:
- Bloom level target: {bloom_level}
- Leaf concept: {leaf}
- Hierarchy path: {path}
- Questions/answers must be image-grounded.
- Provide complete English + Arabic fields.
- Provide high-quality multiple choice fields.
- Return ONLY valid JSON.

Output format:
{{
  "qa_pairs": [
    {{
      "qa_index": 1,
      "bloom_level_targeted": "{bloom_level}",
      "question_en": "",
      "answer_en": "",
      "question_ar": "",
      "answer_ar": "",
      "mc_question_en": "",
      "choice_A_en": "",
      "choice_B_en": "",
      "choice_C_en": "",
      "choice_D_en": "",
      "mc_correct_answer": "A",
      "mc_question_ar": "",
      "choice_A_ar": "",
      "choice_B_ar": "",
      "choice_C_ar": "",
      "choice_D_ar": ""
    }}
  ]
}}"""


def get_create_prompt(hierarchy: dict, num_qa: int) -> str:
    leaf = hierarchy.get("leaf", "visual concept")
    path = hierarchy.get("path", "")
    return f"""You are building 'Create' level VQA benchmarks.
Generate EXACTLY {num_qa} QA pairs for creative image generation evaluation.
Use ONLY hierarchy metadata (no source image).

Constraints:
- Bloom level: Create
- Leaf concept: {leaf}
- Hierarchy path: {path}
- Questions must challenge the model to design/generate a strong image.
- Answers must describe the ideal generated image and why.
- Provide complete English + Arabic fields and full MC options.
- Return ONLY valid JSON in the exact schema below.

Output format:
{{
  "qa_pairs": [
    {{
      "qa_index": 1,
      "bloom_level_targeted": "Create",
      "question_en": "",
      "answer_en": "",
      "question_ar": "",
      "answer_ar": "",
      "mc_question_en": "",
      "choice_A_en": "",
      "choice_B_en": "",
      "choice_C_en": "",
      "choice_D_en": "",
      "mc_correct_answer": "A",
      "mc_question_ar": "",
      "choice_A_ar": "",
      "choice_B_ar": "",
      "choice_C_ar": "",
      "choice_D_ar": ""
    }}
  ]
}}"""


def parse_pairs(raw_text: str) -> list:
    cleaned = clean_json_response(raw_text)
    if not cleaned:
        return []

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        return []

    if isinstance(data, dict) and isinstance(data.get("qa_pairs"), list):
        return data["qa_pairs"]
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and data.get("question_en"):
        return [data]
    return []


def create_entry(hierarchy: dict, qa_data: dict, source_image_file: str) -> dict | None:
    question_en = (qa_data.get("question_en") or "").strip()
    answer_en = (qa_data.get("answer_en") or "").strip()
    if not question_en or not answer_en:
        return None

    choice_a_en = (qa_data.get("choice_A_en") or "").strip()
    choice_b_en = (qa_data.get("choice_B_en") or "").strip()
    choice_c_en = (qa_data.get("choice_C_en") or "").strip()
    choice_d_en = (qa_data.get("choice_D_en") or "").strip()

    choice_a_ar = (qa_data.get("choice_A_ar") or "").strip()
    choice_b_ar = (qa_data.get("choice_B_ar") or "").strip()
    choice_c_ar = (qa_data.get("choice_C_ar") or "").strip()
    choice_d_ar = (qa_data.get("choice_D_ar") or "").strip()

    choices = [
        {"label": "A", "en": choice_a_en, "ar": choice_a_ar},
        {"label": "B", "en": choice_b_en, "ar": choice_b_ar},
        {"label": "C", "en": choice_c_en, "ar": choice_c_ar},
        {"label": "D", "en": choice_d_en, "ar": choice_d_ar},
    ]

    correct = str(qa_data.get("mc_correct_answer") or "A").strip().upper()[:1]
    if correct not in {"A", "B", "C", "D"}:
        correct = "A"

    random.shuffle(choices)
    new_correct = "A"
    for idx, ch in enumerate(choices):
        if ch["label"] == correct:
            new_correct = chr(ord("A") + idx)
            break

    return {
        "image_id": uuid.uuid4().hex,
        "question_id": uuid.uuid4().hex,
        "hierarchy": {"lvl1": hierarchy.get("lvl1", ""), "leaf": hierarchy.get("leaf", "")},
        "question_en": question_en,
        "answer_en": answer_en,
        "question_ar": (qa_data.get("question_ar") or "").strip(),
        "answer_ar": (qa_data.get("answer_ar") or "").strip(),
        "source_image_file": source_image_file,
        "multiple_choice_qa": {
            "question_en": (qa_data.get("mc_question_en") or question_en).strip(),
            "question_ar": (qa_data.get("mc_question_ar") or qa_data.get("question_ar") or "").strip(),
            "choice_A_en": choices[0]["en"],
            "choice_A_ar": choices[0]["ar"],
            "choice_B_en": choices[1]["en"],
            "choice_B_ar": choices[1]["ar"],
            "choice_C_en": choices[2]["en"],
            "choice_C_ar": choices[2]["ar"],
            "choice_D_en": choices[3]["en"],
            "choice_D_ar": choices[3]["ar"],
            "answer": new_correct,
        },
    }


def call_gemini_for_task(task: dict, model, max_retries: int) -> list:
    path_obj = task["source"]
    level = task["level"]
    num_qa = task["num_qa"]
    is_create = level == "Create"

    hierarchy = extract_hierarchy_from_source(path_obj, is_create_folder=is_create)
    prompt = get_create_prompt(hierarchy, num_qa) if is_create else get_image_prompt(hierarchy, num_qa)

    last_error = None
    for attempt in range(max_retries):
        try:
            if is_create:
                response = model.generate_content([prompt])
                source_image_file = ""
            else:
                with Image.open(path_obj) as img:
                    response = model.generate_content([img, prompt])
                source_image_file = str(path_obj.relative_to(DATASET_DIR))

            pairs = parse_pairs(getattr(response, "text", ""))
            entries = []
            for qa_data in pairs:
                entry = create_entry(hierarchy, qa_data, source_image_file)
                if entry is not None:
                    entries.append(entry)

            if entries:
                return entries
            last_error = Exception("No valid QA pairs parsed")
        except Exception as exc:
            last_error = exc

        if attempt < max_retries - 1:
            time.sleep(1.2 + random.uniform(0, 0.8))

    print(f"[!] Failed task level={level} source={path_obj.name}: {last_error}")
    return []


def build_existing_buckets(cleaned: list) -> dict:
    buckets = {lvl: [] for lvl in LEVELS}
    skipped = 0

    for item in cleaned:
        if not isinstance(item, dict):
            skipped += 1
            continue
        hierarchy = item.get("hierarchy")
        if not isinstance(hierarchy, dict):
            skipped += 1
            continue

        level = parse_level_from_lvl1(hierarchy.get("lvl1", ""))
        if level not in buckets:
            skipped += 1
            continue

        if not item.get("question_en") or not item.get("answer_en"):
            skipped += 1
            continue

        buckets[level].append(item)

    if skipped:
        print(f"Skipped {skipped} malformed entries from cleaned dataset")

    return buckets


def discover_candidates() -> tuple[dict, list]:
    image_candidates = {"Analyze": [], "Evaluate": []}
    create_folders = []

    for p in DATASET_DIR.iterdir():
        if p.is_dir() and p.name.startswith("Create"):
            create_folders.append(p)

    img_exts = ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]
    for ext in img_exts:
        for img_path in DATASET_DIR.rglob(ext):
            level = parse_level_from_lvl1(extract_hierarchy_from_source(img_path).get("lvl1", ""))
            if level in image_candidates:
                image_candidates[level].append(img_path)

    return image_candidates, create_folders


def make_tasks(needed: dict, image_candidates: dict, create_folders: list) -> list:
    tasks = []

    for level in ["Analyze", "Evaluate"]:
        missing = needed.get(level, 0)
        if missing <= 0:
            continue
        candidates = image_candidates.get(level, [])
        if not candidates:
            print(f"[!] No {level} image candidates found; cannot fill missing={missing}")
            continue

        random.shuffle(candidates)
        idx = 0
        while missing > 0:
            source = candidates[idx % len(candidates)]
            take = min(10, missing)
            tasks.append({"level": level, "source": source, "num_qa": take})
            missing -= take
            idx += 1

    missing_create = needed.get("Create", 0)
    if missing_create > 0:
        if not create_folders:
            print(f"[!] No Create folders found; cannot fill missing={missing_create}")
        else:
            random.shuffle(create_folders)
            idx = 0
            while missing_create > 0:
                source = create_folders[idx % len(create_folders)]
                take = min(10, missing_create)
                tasks.append({"level": "Create", "source": source, "num_qa": take})
                missing_create -= take
                idx += 1

    return tasks


def trim_to_targets(buckets: dict) -> dict:
    trimmed = {}
    for level in LEVELS:
        items = buckets[level][:]
        random.shuffle(items)
        trimmed[level] = items[: TARGETS[level]]
    return trimmed


def summarize_counts(tag: str, buckets: dict):
    print(f"\n{tag}")
    for level in LEVELS:
        print(f"  {level:<10}: {len(buckets[level])}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-retries", type=int, default=MAX_RETRIES)
    parser.add_argument("--threads", type=int, default=NUM_THREADS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test", action="store_true", help="Generate a tiny missing set for smoke testing")
    args = parser.parse_args()

    random.seed(args.seed)

    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise SystemExit("GOOGLE_API_KEY is not set. Export it first.")

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(
        model_name="gemini-3-flash-preview",
        generation_config={
            "temperature": 0.0,
            "max_output_tokens": GEN_MAX_TOKENS,
            "response_mime_type": "application/json",
        },
    )

    print(f"Loading {CLEANED_DATASET}...")
    with open(CLEANED_DATASET, "r", encoding="utf-8") as f:
        cleaned = json.load(f)

    buckets = build_existing_buckets(cleaned)
    summarize_counts("Existing valid counts", buckets)

    buckets = trim_to_targets(buckets)
    summarize_counts("After initial cap", buckets)

    needed = {lvl: max(0, TARGETS[lvl] - len(buckets[lvl])) for lvl in LEVELS}

    if args.test:
        for lvl in GENERATABLE_LEVELS:
            needed[lvl] = min(needed[lvl], 20)

    print("\nMissing to generate")
    for lvl in LEVELS:
        print(f"  {lvl:<10}: {needed[lvl]}")

    image_candidates, create_folders = discover_candidates()
    print(f"\nCandidate sources: Analyze={len(image_candidates['Analyze'])}, Evaluate={len(image_candidates['Evaluate'])}, CreateFolders={len(create_folders)}")

    tasks = make_tasks(needed, image_candidates, create_folders)
    print(f"Total Gemini tasks queued: {len(tasks)}")

    generated_by_level = {lvl: [] for lvl in LEVELS}
    if tasks:
        with ThreadPoolExecutor(max_workers=args.threads) as executor:
            futures = [executor.submit(call_gemini_for_task, task, model, args.max_retries) for task in tasks]
            for fut in tqdm(as_completed(futures), total=len(futures), desc="Generating missing QAs", unit="task"):
                entries = fut.result()
                for entry in entries:
                    lvl = parse_level_from_lvl1(entry.get("hierarchy", {}).get("lvl1", ""))
                    if lvl in generated_by_level:
                        generated_by_level[lvl].append(entry)

    summarize_counts("Generated raw counts", generated_by_level)

    for lvl in GENERATABLE_LEVELS:
        if needed[lvl] > 0:
            random.shuffle(generated_by_level[lvl])
            buckets[lvl].extend(generated_by_level[lvl][: needed[lvl]])

    buckets = trim_to_targets(buckets)
    summarize_counts("Final counts", buckets)

    final_dataset = []
    for lvl in LEVELS:
        final_dataset.extend(buckets[lvl])

    random.shuffle(final_dataset)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(final_dataset, f, ensure_ascii=False, indent=2)

    print(f"\nSaved balanced dataset to: {OUTPUT_FILE}")
    print(f"Total entries: {len(final_dataset)}")


if __name__ == "__main__":
    GEN_MAX_TOKENS = 16384
    main()
