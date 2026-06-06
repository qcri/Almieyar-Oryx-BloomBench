"""
Generate QA dataset from images using Qwen3-VL-235B-A22B-Instruct.

Optimized version:
- Single-pass generation (QA + Arabic + MCQ)
- Deterministic decoding (benchmark-grade)
"""

import os
import json
import uuid
from pathlib import Path
import torch
from transformers import Qwen3VLMoeForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from tqdm import tqdm

# -----------------------------------------------------------------------------
# TEMP DIR (short path avoids IPC socket issues)
# -----------------------------------------------------------------------------
TEMP_DIR = "/fanar-image-understanding-01/OryxTrain/OryxSegments/fanar3/tmp"
os.makedirs(TEMP_DIR, exist_ok=True)

os.environ["TMPDIR"] = TEMP_DIR
os.environ["TEMP"] = TEMP_DIR
os.environ["TMP"] = TEMP_DIR
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

# -----------------------------------------------------------------------------
# DATASET PATHS
# -----------------------------------------------------------------------------
DATASET_DIR = Path(
    "/fanar-image-understanding-01/OryxTrain/OryxSegments/fanar3/Omid/"
    "Bloom-bench-acl/Dataset/bloombench-crawling/Updated-dataset"
)

OUTPUT_FILE = Path(
    "/fanar-image-understanding-01/OryxTrain/OryxSegments/fanar3/Omid/"
    "Bloom-bench-acl/scripts/QA_Qwen3-VL-235B-A22B-Instruct-FP8.json"
)

# -----------------------------------------------------------------------------
# GENERATION SETTINGS (BEST PRACTICE)
# -----------------------------------------------------------------------------
BATCH_SIZE = 10               
GEN_MAX_TOKENS = 2048         # Safe for full JSON + translations

# -----------------------------------------------------------------------------
# HELPERS
# -----------------------------------------------------------------------------
def extract_hierarchy_from_path(image_path: str):
    folder_name = Path(image_path).parent.name
    if " - " in folder_name:
        idx = folder_name.rfind(" - ")
        return {
            "lvl1": folder_name[:idx].strip(),
            "leaf": folder_name[idx + 3:].strip(),
        }
    return {"lvl1": folder_name, "leaf": folder_name}


def get_comprehensive_prompt(leaf: str, bloom_node: str, path: str):
    return f"""You are an expert assistant for creating Visual Question Answering (VQA) benchmarks grounded in Bloom's Taxonomy.

---
CONTEXT:
We are building a benchmark to evaluate the cognitive abilities of Vision-Language Models (VLLMs). Each VQA pair must target a specific thinking skill (Bloom level) as applied to a core concept (leaf). The questions should be answerable *only* by analyzing the visual content of the image.

---
YOUR ASSIGNMENT:
- **Path**: {path}
- **Leaf (Core Topic)**: {leaf}
- **Bloom Level (Cognitive Skill)**: {bloom_node}

---
BLOOM'S TAXONOMY LEVELS:
1. **Remembering** – Recall facts and basic concepts. (e.g., "What color is the car?")
   - Abilities: Recognizing, Identifying, Listing.
2. **Understanding** – Explain ideas or concepts. (e.g., "Explain what is happening in this scene.")
   - Abilities: Summarizing, Interpreting, Classifying.
3. **Applying** – Use information in new situations. (e.g., "What tool would you use to fix this?")
   - Abilities: Executing, Implementing, Using.
4. **Analyzing** – Draw connections among ideas. Break down information. (e.g., "What is the relationship between the person and the dog?")
   - Abilities: Differentiating, Organizing, Comparing, Attributing.
5. **Evaluating** – Justify a stand or decision. (e.g., "Is this a safe situation? Why or why not?")
   - Abilities: Critiquing, Justifying, Validating.
6. **Creating** – Produce new or original work. (e.g., "What could be a creative caption for this image?")
   - Abilities: Designing, Planning, Producing.

---
GUIDELINES:
1. **Image-Grounded**: Both question and answer MUST be directly derivable from the visual information in the image. Do not invent information.
2. **Bloom Alignment**: The question must genuinely require the cognitive skill of "{bloom_node}". Ensure the question type matches the Bloom level.
3. **Leaf Targeting**: The question should revolve around the "{leaf}" concept.
4. **Clarity and Conciseness**: Questions should be clear and unambiguous. Answers should be direct and concise.
5. **Deterministic Answers**: The answer should be objective and verifiable from the image. Avoid questions that are subjective, speculative, or could have multiple valid answers. The goal is to create reliable, deterministic benchmark items.
6. **Quality over Quantity**: Generate up to 10 high-quality VQA pairs. If the image is not suitable for generating meaningful pairs that align with "{leaf}" and "{bloom_node}", return fewer pairs or an empty list. Do not force bad questions.

---
YOUR TASK:
Analyze this image and generate up to 10 high-quality VQA pairs. For each pair:
1. Create a challenging question in English that targets "{bloom_node}" level thinking about "{leaf}".
2. Provide the deterministic, verifiable answer in English.
3. Translate the question and answer to Arabic.
4. Create a multiple-choice variant in English (4 options A-D) and translate the options to Arabic.

Return the response as a valid JSON array ONLY. Do not include markdown code blocks.
The JSON structure MUST be an array of objects, each matching:

[
  {{
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
"""


def clean_json_response(text: str) -> str:
    text = text.strip()
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0]
    elif "```" in text:
        text = text.split("```")[1].split("```")[0]
    return text.strip()

# -----------------------------------------------------------------------------
# MODEL INFERENCE
# -----------------------------------------------------------------------------
def run_batch_inference(image_paths, model, processor):
    messages_list = []
    
    for p in image_paths:
        hierarchy = extract_hierarchy_from_path(str(p))
        leaf = hierarchy.get("leaf", "Unknown")
        bloom_node = hierarchy.get("lvl1", "Analyzing")
        path = f"{bloom_node} -> {leaf}"
        
        prompt = get_comprehensive_prompt(leaf, bloom_node, path)
        
        messages_list.append([{
            "role": "user",
            "content": [
                {"type": "image", "image": str(p)},
                {"type": "text", "text": prompt},
            ],
        }])
    

    texts = [
        processor.apply_chat_template(
            msg, tokenize=False, add_generation_prompt=True
        )
        for msg in messages_list
    ]

    image_inputs, video_inputs = process_vision_info(messages_list)

    inputs = processor(
        text=texts,
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=GEN_MAX_TOKENS,
            do_sample=False,              # 🔒 Deterministic
            num_beams=1,                  # Fast + stable
            repetition_penalty=1.05,      # Avoid JSON loops
            eos_token_id=processor.tokenizer.eos_token_id,
            pad_token_id=processor.tokenizer.pad_token_id,
        )

    trimmed = [
        out[len(inp):] for inp, out in zip(inputs.input_ids, generated_ids)
    ]

    return processor.batch_decode(
        trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )

# -----------------------------------------------------------------------------
# BATCH PROCESSING
# -----------------------------------------------------------------------------
def process_batch(image_paths, model, processor):
    outputs = run_batch_inference(image_paths, model, processor)
    results = []

    for image_path, raw in zip(image_paths, outputs):
        cleaned = clean_json_response(raw)
        hierarchy = extract_hierarchy_from_path(str(image_path))
        image_id = uuid.uuid4().hex

        try:
            data = json.loads(cleaned)
            
            # Handle both single object and array responses
            qa_list = data if isinstance(data, list) else [data]
            
            # Process up to 10 QA pairs
            for qa_data in qa_list[:10]:
                entry = {
                    "question_en": "", "answer_en": "",
                    "question_ar": "", "answer_ar": "",
                    "mc_question_en": "", "mc_correct_answer": "A",
                    "mc_question_ar": "",
                    "choice_A_en": "", "choice_B_en": "",
                    "choice_C_en": "", "choice_D_en": "",
                    "choice_A_ar": "", "choice_B_ar": "",
                    "choice_C_ar": "", "choice_D_ar": "",
                }
                
                # Extract fields from qa_data
                for k in entry:
                    if k in qa_data:
                        entry[k] = qa_data[k]

                entry["mc_correct_answer"] = (
                    str(entry["mc_correct_answer"]).strip().upper()[:1] or "A"
                )

                results.append({
                    "image_id": image_id,
                    "question_id": uuid.uuid4().hex,
                    "hierarchy": hierarchy,
                    "question_en": entry["question_en"],
                    "answer_en": entry["answer_en"],
                    "question_ar": entry["question_ar"],
                    "answer_ar": entry["answer_ar"],
                    "source_image_file": str(image_path.relative_to(DATASET_DIR)),
                    "multiple_choice_qa": {
                        "question_en": entry["mc_question_en"] or entry["question_en"],
                        "question_ar": entry["mc_question_ar"] or entry["question_ar"],
                        "choice_A_en": entry["choice_A_en"],
                        "choice_A_ar": entry["choice_A_ar"],
                        "choice_B_en": entry["choice_B_en"],
                        "choice_B_ar": entry["choice_B_ar"],
                        "choice_C_en": entry["choice_C_en"],
                        "choice_C_ar": entry["choice_C_ar"],
                        "choice_D_en": entry["choice_D_en"],
                        "choice_D_ar": entry["choice_D_ar"],
                        "answer": entry["mc_correct_answer"],
                    },
                })

        except json.JSONDecodeError as e:
            print(f"[!] JSON parse failed for {image_path.name}: {e}")

    return results

# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    image_paths = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"):
        image_paths.extend(DATASET_DIR.rglob(ext))

    image_paths = sorted(image_paths)
    print(f"Found {len(image_paths)} images")

    checkpoint = "Qwen/Qwen3-VL-235B-A22B-Instruct-FP8"
    print(f"Loading {checkpoint} (single-pass optimized mode)")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
        checkpoint,
        torch_dtype="auto",  # Auto-detect dtype for FP8 quantized models
        device_map="auto",
        trust_remote_code=True,
    )

    processor = AutoProcessor.from_pretrained(checkpoint, trust_remote_code=True)
    processor.padding_side = "left"

    qa_dataset = []

    with tqdm(total=len(image_paths), desc="Processing images", unit="img") as pbar:
        for i in range(0, len(image_paths), BATCH_SIZE):
            batch = image_paths[i:i + BATCH_SIZE]
            try:
                qa_dataset.extend(process_batch(batch, model, processor))
                pbar.update(len(batch))

                with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
                    json.dump(qa_dataset, f, ensure_ascii=False, indent=2)

            except Exception as e:
                print(f"[!] Batch error at index {i}: {e}")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    print(f"Done. Saved {len(qa_dataset)} entries → {OUTPUT_FILE}")
