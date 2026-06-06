#!/usr/bin/env python3
"""
Step 3: Run Qwen2.5-VL-7B-Instruct on all agreed MC samples
        in English, Arabic and Spanish.

Produces both:
  - Regex extraction answer  (RAE)
  - Likelihood-based answer  (LBS)

Saves results/<lang>_results.csv  incrementally so runs can be resumed.
"""

import json, os, sys, re
import pandas as pd
import torch
from pathlib import Path
from tqdm import tqdm
from transformers import (
    AutoProcessor,
    AutoTokenizer,
    Qwen2_5_VLForConditionalGeneration,
    BitsAndBytesConfig,
)
from qwen_vl_utils import process_vision_info
import torch.nn.functional as F

BASE = os.path.dirname(os.path.abspath(__file__))
INPUT_PATH = os.path.join(BASE, "agreed_mc_samples_with_spanish.json")
RESULTS_DIR = os.path.join(BASE, "results_rebuttal")
os.makedirs(RESULTS_DIR, exist_ok=True)

MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"

# ── Helpers ───────────────────────────────────────────────────────────

def build_prompt(mc, lang, likelihood=False):
    """Return (prompt_text, gold_answer, [choices if likelihood])."""
    if lang == "en":
        q = mc["question_en"]
        cA, cB, cC, cD = mc["choice_A_en"], mc["choice_B_en"], mc["choice_C_en"], mc["choice_D_en"]
    elif lang == "ar":
        q = mc["question_ar"]
        cA, cB, cC, cD = mc["choice_A_ar"], mc["choice_B_ar"], mc["choice_C_ar"], mc["choice_D_ar"]
    elif lang == "es":
        q = mc["question_es"]
        cA, cB, cC, cD = mc["choice_A_es"], mc["choice_B_es"], mc["choice_C_es"], mc["choice_D_es"]
    else:
        raise ValueError(f"Unknown lang {lang}")

    gold = mc["answer"]  # A/B/C/D

    if likelihood:
        prompt = (
            "You are an expert in visual question answering.\n"
            "You will be given an image and a question about that image.\n"
            "Your task is to answer the question based on the visual content of the image.\n"
            f"Question: {q}\n"
            "Answer: "
        )
        return prompt, gold, [cA, cB, cC, cD]
    else:
        prompt = (
            "You are an expert in visual question answering.\n"
            "You will be given an image and a question about that image.\n"
            "Your task is to answer the question based on the visual content of the image.\n"
            "The question is in multiple choice format, and you need to select the correct answer from the given options.\n"
            f"Question: {q}\n"
            "Options:\n"
            f"1) {cA}\n"
            f"2) {cB}\n"
            f"3) {cC}\n"
            f"4) {cD}\n"
            "Please provide the number of the correct answer (1, 2, 3, or 4) as your response without any additional text.\n"
            "Answer: "
        )
        return prompt, gold, [cA, cB, cC, cD]


def extract_regex_answer(text):
    """Extract the chosen option number → letter."""
    match = re.findall(r"[1-4]", text.strip())
    if len(match) >= 1:
        return {"1": "A", "2": "B", "3": "C", "4": "D"}[match[0]]
    return "random"


def score_candidate(context_inputs, full_inputs, model):
    """Average log-prob of answer tokens beyond context length."""
    context_len = context_inputs["input_ids"].shape[1]
    input_ids = full_inputs["input_ids"]
    with torch.no_grad():
        logits = model(**full_inputs).logits
    answer_logits = logits[:, context_len - 1 : -1, :]
    answer_ids = input_ids[:, context_len:]
    log_probs = F.log_softmax(answer_logits, dim=-1)
    token_log_probs = log_probs.gather(2, answer_ids.unsqueeze(-1)).squeeze(-1)
    return token_log_probs.mean().item()


# ── Main ──────────────────────────────────────────────────────────────

def main():
    # Load samples
    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        samples = json.load(f)
    print(f"Loaded {len(samples)} samples")

    # Load model (use device_map auto to spread across GPUs)
    print(f"Loading {MODEL_ID} …")
    hf_token = os.environ.get("HF_TOKEN")

    qmodel = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        token=hf_token,
    )
    processor = AutoProcessor.from_pretrained(MODEL_ID, token=hf_token)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=hf_token)
    qmodel.eval()
    print("Model loaded ✓")

    # Languages to evaluate
    LANGS = ["en", "ar", "es"]

    for lang in LANGS:
        out_csv = os.path.join(RESULTS_DIR, f"{lang}_results.csv")

        # Resume logic
        if os.path.exists(out_csv):
            cache_df = pd.read_csv(out_csv)
            done_qids = set(cache_df["question_id"].astype(str).tolist())
            rows = cache_df.to_dict("records")
            print(f"[{lang}] Resuming – {len(done_qids)} already done")
        else:
            done_qids, rows = set(), []

        for sample in tqdm(samples, desc=f"Evaluating [{lang}]"):
            qid = sample["question_id"]
            if qid in done_qids:
                continue

            mc = sample.get("multiple_choice_qa")
            if not mc:
                # Skip non-MC items for RAE/LBS evaluation
                continue
                
            img_path = os.path.join(BASE, sample["source_image_file"])

            # Skip if Spanish fields missing
            if lang == "es" and not mc.get("question_es", "").strip():
                continue

            try:
                # ── 1) Regex extraction (generation) ──────────────
                prompt_gen, gold, choices = build_prompt(mc, lang, likelihood=False)
                messages = [
                    {"role": "user", "content": [
                        {"type": "image", "image": img_path},
                        {"type": "text", "text": prompt_gen},
                    ]}
                ]
                text_in = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                image_inputs, video_inputs = process_vision_info(messages)
                inputs = processor(
                    text=[text_in], images=image_inputs, videos=video_inputs,
                    padding=True, return_tensors="pt",
                ).to(qmodel.device)

                with torch.inference_mode():
                    gen_out = qmodel.generate(**inputs, max_new_tokens=128, do_sample=False)
                gen_ids = gen_out[0][len(inputs.input_ids[0]):]
                response_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
                regex_answer = extract_regex_answer(response_text)

                # ── 2) Likelihood scoring ─────────────────────────
                prompt_lh, _, lh_choices = build_prompt(mc, lang, likelihood=True)
                msg_base = [
                    {"role": "user", "content": [
                        {"type": "image", "image": img_path},
                        {"type": "text", "text": prompt_lh},
                    ]}
                ]
                text_base = processor.apply_chat_template(msg_base, tokenize=False, add_generation_prompt=True)
                img_in_base, vid_in_base = process_vision_info(msg_base)
                ctx_inputs = processor(
                    text=[text_base], images=img_in_base, videos=vid_in_base,
                    padding=True, return_tensors="pt",
                ).to(qmodel.device)

                option_scores = []
                for opt in lh_choices:
                    msg_full = [
                        {"role": "user", "content": [
                            {"type": "image", "image": img_path},
                            {"type": "text", "text": prompt_lh + opt},
                        ]}
                    ]
                    text_full = processor.apply_chat_template(msg_full, tokenize=False, add_generation_prompt=True)
                    img_in_f, vid_in_f = process_vision_info(msg_full)
                    full_inputs = processor(
                        text=[text_full], images=img_in_f, videos=vid_in_f,
                        padding=True, return_tensors="pt",
                    ).to(qmodel.device)
                    option_scores.append(score_candidate(ctx_inputs, full_inputs, qmodel))

                abcd = ["A", "B", "C", "D"]
                best_idx = max(range(4), key=lambda i: option_scores[i])
                likelihood_answer = abcd[best_idx]

                # ── Hybrid ────────────────────────────────────────
                final_answer = regex_answer if regex_answer != "random" else likelihood_answer

                row = {
                    "question_id": qid,
                    "lang": lang,
                    "gold": gold,
                    "regex_answer": regex_answer,
                    "likelihood_answer": likelihood_answer,
                    "final_answer": final_answer,
                    "response_text": response_text,
                    "option_scores": json.dumps(dict(zip(abcd, option_scores))),
                    "hierarchy_lvl1": sample["hierarchy"]["lvl1"],
                    "hierarchy_leaf": sample["hierarchy"]["leaf"],
                }

            except Exception as e:
                print(f"  ⚠ Error on {qid}/{lang}: {e}")
                row = {
                    "question_id": qid,
                    "lang": lang,
                    "gold": "ERROR",
                    "regex_answer": "ERROR",
                    "likelihood_answer": "ERROR",
                    "final_answer": "ERROR",
                    "response_text": str(e),
                    "option_scores": "ERROR",
                    "hierarchy_lvl1": sample.get("hierarchy", {}).get("lvl1", ""),
                    "hierarchy_leaf": sample.get("hierarchy", {}).get("leaf", ""),
                }

            rows.append(row)

            # Incremental save every row
            pd.DataFrame([row]).to_csv(
                out_csv, mode="a",
                header=not os.path.exists(out_csv) or os.path.getsize(out_csv) == 0,
                index=False, encoding="utf-8",
            )

        # Full final save
        df = pd.DataFrame(rows)
        df.to_csv(out_csv, index=False, encoding="utf-8")
        print(f"[{lang}] Saved {len(rows)} results → {out_csv}")

    print("\n✓ Inference complete for all languages.")


if __name__ == "__main__":
    main()
