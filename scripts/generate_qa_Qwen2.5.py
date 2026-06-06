
"""Generate QA dataset from images using Qwen2.5-VL-72B-Instruct.

Optimized version: Generates all content (QA + Translations + Multiple Choice) 
in a SINGLE model pass per image to maximize throughput and efficiency.
"""

import os
import json
import uuid
import re
import random
from pathlib import Path

import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from qwen_vl_utils import process_vision_info
from tqdm import tqdm

# Short temp dir to avoid long IPC socket paths
TEMP_DIR = "/fanar-image-understanding-01/OryxTrain/OryxSegments/fanar3/tmp"
os.makedirs(TEMP_DIR, exist_ok=True)
os.environ["TMPDIR"] = TEMP_DIR
os.environ["TEMP"] = TEMP_DIR
os.environ["TMP"] = TEMP_DIR
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

# Dataset and output
DATASET_DIR = Path("/fanar-image-understanding-01/OryxTrain/OryxSegments/fanar3/Omid/Bloom-bench-acl/Dataset/bloombench-crawling/Updated-dataset")
OUTPUT_FILE = Path("/fanar-image-understanding-01/OryxTrain/OryxSegments/fanar3/Omid/Bloom-bench-acl/scripts/QA_Qwen2.5-VL-72B-Instruct_Optimized.json")

# Generation settings
GEN_MAX_TOKENS = 2048  # Increased slightly to accommodate JSON + all translations
BATCH_SIZE = 4  # Reduced slightly because we are generating more text per sample now

def extract_hierarchy_from_path(image_path: str):
    folder_name = Path(image_path).parent.name
    
    if ' - ' in folder_name:
        last_separator_idx = folder_name.rfind(' - ')
        lvl1 = folder_name[:last_separator_idx].strip()
        leaf = folder_name[last_separator_idx + 3:].strip()
    else:
        lvl1 = folder_name.strip()
        leaf = folder_name.strip()
    
    return {"lvl1": lvl1, "leaf": leaf}

def get_comprehensive_prompt():
    """
    Returns a prompt that asks the model to generate everything at once.
    This is 4x more efficient than 4 separate calls.
    """
    return """Analyze this image thoroughly. Your task is to:
1. Create a challenging question in English.
2. Provide the correct answer in English.
3. Translate the question and answer to Arabic.
4. Create a multiple-choice variant in English (4 options A-D) and translate the options to Arabic.

Return the response in valid JSON format ONLY. Do not include markdown code blocks (```json). Ensure the JSON structure matches this exactly:

{
    "question_en": "Your English question here",
    "answer_en": "Your English answer here",
    "question_ar": "سؤالك باللغة العربية هنا",
    "answer_ar": "إجابتك باللغة العربية هنا",
    "mc_question_en": "Your English multiple choice question here",
    "choice_A_en": "Option A (English)",
    "choice_B_en": "Option B (English)",
    "choice_C_en": "Option C (English)",
    "choice_D_en": "Option D (English)",
    "mc_correct_answer": "A",
    "mc_question_ar": "سؤال الاختيار من متعدد باللغة العربية هنا",
    "choice_A_ar": "الخيار أ (عربي)",
    "choice_B_ar": "الخيار ب (عربي)",
    "choice_C_ar": "الخيار ج (عربي)",
    "choice_D_ar": "الخيار د (عربي)"
}"""

def clean_json_response(text: str) -> str:
    """Extracts JSON from text, handling common LLM formatting quirks."""
    text = text.strip()
    # Remove markdown code blocks if present
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0]
    elif "```" in text:
        text = text.split("```")[1].split("```")[0]
    return text.strip()

def run_batch_inference(image_paths_batch, model, processor):
    prompt_text = get_comprehensive_prompt()
    
    messages_list = []
    for image_path in image_paths_batch:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": str(image_path)},
                    {"type": "text", "text": prompt_text},
                ],
            }
        ]
        messages_list.append(messages)
    
    # Process texts for processor
    texts = [
        processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
        for msg in messages_list
    ]
    
    # Process images (using qwen_vl_utils)
    image_inputs, video_inputs = process_vision_info(messages_list)
    
    inputs = processor(
        text=texts,
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=GEN_MAX_TOKENS, do_sample=False, temperature=0.0)
    
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_texts = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return output_texts

def process_batch(image_paths_batch, model, processor):
    outputs = run_batch_inference(image_paths_batch, model, processor)
    
    final_entries = []
    
    for image_path, raw_output in zip(image_paths_batch, outputs):
        cleaned_output = clean_json_response(raw_output)
        hierarchy = extract_hierarchy_from_path(str(image_path))
        
        # Defaults to prevent crashes if parsing fails
        entry = {
            "question_en": "", "answer_en": "",
            "question_ar": "", "answer_ar": "",
            "mc_question_en": "", "mc_correct_answer": "A",
            "mc_question_ar": "",
            "choice_A_en": "", "choice_B_en": "", "choice_C_en": "", "choice_D_en": "",
            "choice_A_ar": "", "choice_B_ar": "", "choice_C_ar": "", "choice_D_ar": ""
        }
        
        try:
            data = json.loads(cleaned_output)
            
            # Simple extraction
            entry["question_en"] = data.get("question_en", "")
            entry["answer_en"] = data.get("answer_en", "")
            entry["question_ar"] = data.get("question_ar", "")
            entry["answer_ar"] = data.get("answer_ar", "")
            
            entry["mc_question_en"] = data.get("mc_question_en", "")
            entry["choice_A_en"] = data.get("choice_A_en", "")
            entry["choice_B_en"] = data.get("choice_B_en", "")
            entry["choice_C_en"] = data.get("choice_C_en", "")
            entry["choice_D_en"] = data.get("choice_D_en", "")
            
            # Normalize answer to single upper case letter
            ans = data.get("mc_correct_answer", "A")
            entry["mc_correct_answer"] = str(ans).strip().upper()[0] if ans else "A"
            
            entry["mc_question_ar"] = data.get("mc_question_ar", "")
            entry["choice_A_ar"] = data.get("choice_A_ar", "")
            entry["choice_B_ar"] = data.get("choice_B_ar", "")
            entry["choice_C_ar"] = data.get("choice_C_ar", "")
            entry["choice_D_ar"] = data.get("choice_D_ar", "")

        except json.JSONDecodeError as e:
            print(f"\n[!] Warning: Failed to parse JSON for {image_path.name}. Using fallback. Error: {e}")
            # Optional: print raw output for debugging
            # print(f"Raw: {raw_output[:200]}")

        multiple_choice_qa = {
            "question_en": entry["mc_question_en"] if entry["mc_question_en"] else entry["question_en"],
            "question_ar": entry["mc_question_ar"] if entry["mc_question_ar"] else entry["question_ar"],
            "choice_A_en": entry["choice_A_en"],
            "choice_A_ar": entry["choice_A_ar"],
            "choice_B_en": entry["choice_B_en"],
            "choice_B_ar": entry["choice_B_ar"],
            "choice_C_en": entry["choice_C_en"],
            "choice_C_ar": entry["choice_C_ar"],
            "choice_D_en": entry["choice_D_en"],
            "choice_D_ar": entry["choice_D_ar"],
            "answer": entry["mc_correct_answer"],
        }

        final_entries.append({
            "image_id": str(uuid.uuid4()).replace("-", ""),
            "question_id": str(uuid.uuid4()).replace("-", ""),
            "hierarchy": hierarchy,
            "question_en": entry["question_en"],
            "answer_en": entry["answer_en"],
            "question_ar": entry["question_ar"],
            "answer_ar": entry["answer_ar"],
            "source_image_file": str(image_path.relative_to(DATASET_DIR)),
            "multiple_choice_qa": multiple_choice_qa,
        })
    
    return final_entries

if __name__ == '__main__':
    img_exts = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
    image_paths = []
    for ext in img_exts:
        image_paths.extend(sorted(DATASET_DIR.rglob(ext)))

    print(f"Found {len(image_paths)} images in {DATASET_DIR}")
    if len(image_paths) == 0:
        raise SystemExit("No images found in dataset directory")

    checkpoint = "Qwen/Qwen2.5-VL-72B-Instruct"
    print(f"Loading model {checkpoint} with 4-bit quantization...")
    print("OPTIMIZATION MODE: Single-pass generation (4x faster).")

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        checkpoint,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        quantization_config=quantization_config,
        trust_remote_code=True
    )

    processor = AutoProcessor.from_pretrained(checkpoint, trust_remote_code=True)
    # Using right padding for generation is standard, left padding is for training/continued pre-training usually
    # However, for batched inference with padding_side='left', we ensure padding tokens are ignored in loss if needed.
    # For simple generation, 'right' is usually safer unless specific requirements exist. 
    # Leaving default or setting to right if not specified in original, but keeping original config if it worked.
    # Original set 'left', let's keep it to be safe.
    processor.tokenizer.padding_side = 'left' 

    qa_dataset = []
    total = len(image_paths)

    with tqdm(total=total, desc="Processing images (Optimized)", unit="image") as pbar:
        for i in range(0, total, BATCH_SIZE):
            batch_paths = image_paths[i : i + BATCH_SIZE]
            
            try:
                batch_entries = process_batch(batch_paths, model, processor)
                qa_dataset.extend(batch_entries)
                pbar.update(len(batch_paths))

                # Save checkpoint
                with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
                    json.dump(qa_dataset, f, ensure_ascii=False, indent=2)
                
                # Print status less frequently to not spam logs
                if (i // BATCH_SIZE) % 10 == 0:
                    print(f"Processed {i + len(batch_paths)}/{total} images...")

            except Exception as e:
                print(f"\n[!] Error processing batch starting at {i}: {e}")
                import traceback
                traceback.print_exc()
                # Clear cache to prevent error cascade
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    print(f"Done. Saved {len(qa_dataset)} entries to {OUTPUT_FILE}")