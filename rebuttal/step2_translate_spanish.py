#!/usr/bin/env python3
"""
Step 2: Translate agreed MC QA to Spanish using Gemini 3 Flash.
Translates: question, 4 choices.  The answer letter stays the same.
Uses batching (10 items per prompt) for efficiency, with retry logic.
Saves incrementally so it can be resumed.
"""

import json, os, time, re, copy
import google.generativeai as genai

BASE = os.path.dirname(os.path.abspath(__file__))
INPUT_PATH = os.path.join(BASE, "agreed_mc_samples.json")
OUTPUT_PATH = os.path.join(BASE, "agreed_mc_samples_with_spanish.json")

BATCH_SIZE = 5  # items per Gemini call

# ── Configure Gemini ──────────────────────────────────────────────────
api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY", "")
if not api_key:
    raise RuntimeError("Set GOOGLE_API_KEY or GEMINI_API_KEY environment variable")
genai.configure(api_key=api_key)
model = genai.GenerativeModel("gemini-3-flash-preview")

TRANSLATE_PROMPT = """You are a professional translator. Translate the following JSON array of objects from English to Spanish.
Return ONLY a valid JSON array with the same structure, where every string value is translated to Spanish.
Keep the "id" and "type" keys exactly as-is. Do NOT add any commentary or markdown fences.

Input:
{batch_json}
"""


def translate_batch(items):
    """Send a batch of items to Gemini for translation; returns list of dicts."""
    batch = []
    for it in items:
        obj = {"id": it["question_id"]}
        mc = it.get("multiple_choice_qa")
        if mc:
            obj["type"] = "mc"
            obj["question"] = mc["question_en"]
            obj["choice_A"] = mc["choice_A_en"]
            obj["choice_B"] = mc["choice_B_en"]
            obj["choice_C"] = mc["choice_C_en"]
            obj["choice_D"] = mc["choice_D_en"]
        else:
            obj["type"] = "open"
            obj["question"] = it["question_en"]
            obj["answer"] = it["answer_en"]
        batch.append(obj)
    
    prompt = TRANSLATE_PROMPT.format(batch_json=json.dumps(batch, ensure_ascii=False, indent=2))

    for attempt in range(5):
        try:
            resp = model.generate_content(prompt)
            text = resp.text.strip()
            # Strip markdown code fences if present
            text = re.sub(r"^```(?:json)?\s*", "", text)
            text = re.sub(r"\s*```$", "", text)
            translated = json.loads(text)
            # Ensure it's a list
            if isinstance(translated, dict) and "items" in translated:
                translated = translated["items"]
            return translated
        except Exception as e:
            wait = 2 ** attempt
            print(f"  ⚠ Attempt {attempt+1} failed ({e}), retrying in {wait}s…")
            time.sleep(wait)
    raise RuntimeError("Translation failed after 5 retries")


def main():
    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        samples = json.load(f)
    print(f"Loaded {len(samples)} agreed samples")

    # ── Load partial progress if exists ───────────────────────────────
    if os.path.exists(OUTPUT_PATH):
        with open(OUTPUT_PATH, "r", encoding="utf-8") as f:
            done_samples = json.load(f)
        
        def is_translated(s):
            if s.get("multiple_choice_qa"):
                return "question_es" in s["multiple_choice_qa"]
            return s.get("question_es") is not None

        done_ids = {d["question_id"] for d in done_samples if is_translated(d)}
        print(f"Resuming – {len(done_ids)} already translated")
    else:
        done_samples = []
        done_ids = set()

    # Build a map from question_id -> index in done_samples for updating
    done_map = {d["question_id"]: i for i, d in enumerate(done_samples)}

    # Items still needing translation
    todo = [s for s in samples if s["question_id"] not in done_ids]
    print(f"Items to translate: {len(todo)}")

    # ── Translate in batches ──────────────────────────────────────────
    for batch_start in range(0, len(todo), BATCH_SIZE):
        batch = todo[batch_start : batch_start + BATCH_SIZE]
        print(f"Translating batch {batch_start // BATCH_SIZE + 1} "
              f"({batch_start+1}–{batch_start+len(batch)} of {len(todo)})")

        translated = translate_batch(batch)
        tr_map = {t["id"]: t for t in translated}

        for item in batch:
            qid = item["question_id"]
            tr = tr_map.get(qid, {})
            enriched = copy.deepcopy(item)
            
            if item.get("multiple_choice_qa"):
                enriched["multiple_choice_qa"]["question_es"] = tr.get("question", "")
                enriched["multiple_choice_qa"]["choice_A_es"] = tr.get("choice_A", "")
                enriched["multiple_choice_qa"]["choice_B_es"] = tr.get("choice_B", "")
                enriched["multiple_choice_qa"]["choice_C_es"] = tr.get("choice_C", "")
                enriched["multiple_choice_qa"]["choice_D_es"] = tr.get("choice_D", "")
            else:
                enriched["question_es"] = tr.get("question", "")
                enriched["answer_es"] = tr.get("answer", "")

            if qid in done_map:
                done_samples[done_map[qid]] = enriched
            else:
                done_map[qid] = len(done_samples)
                done_samples.append(enriched)

        # Incremental save
        with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
            json.dump(done_samples, f, ensure_ascii=False, indent=2)

        time.sleep(1)  # rate-limit courtesy

    print(f"\n✓ Translation complete – {len(done_samples)} items saved → {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
