
"""Generate QA dataset from images using Gemini 3 Flash.

Optimized version: Generates 10 high-quality VQA pairs per image based on 
Bloom's Taxonomy cognitive levels. Includes English and Arabic translations
plus multiple choice variants.
"""

import os
import json
import uuid
import re
import random
import time
from pathlib import Path
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

import google.generativeai as genai
import argparse
from tqdm import tqdm

# Gemini API Key Setup
# Provide credentials via environment variable `GOOGLE_API_KEY` (preferred)
# or via Application Default Credentials (`GOOGLE_APPLICATION_CREDENTIALS`).
genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))

# Short temp dir to avoid long IPC socket paths
TEMP_DIR = "/fanar-image-understanding-01/OryxTrain/OryxSegments/fanar3/tmp"
os.makedirs(TEMP_DIR, exist_ok=True)
os.environ["TMPDIR"] = TEMP_DIR
os.environ["TEMP"] = TEMP_DIR
os.environ["TMP"] = TEMP_DIR

# Dataset and output
DATASET_DIR = Path("/fanar-image-understanding-01/OryxTrain/OryxSegments/fanar3/Omid/Bloom-bench-acl/Dataset/bloombench-crawling/Updated-dataset")
OUTPUT_FILE = Path("/fanar-image-understanding-01/OryxTrain/OryxSegments/fanar3/Omid/Bloom-bench-acl/scripts/QA_Gemini-3-Flash_Dataset-10.json")

# Generation settings
GEN_MAX_TOKENS = 16384  
BATCH_SIZE = 100  # Sequential processing for Gemini API
MAX_RETRIES = 4  # Default number of retries for failed API calls
NUM_THREADS = 12  # Default number of parallel threads for API calls
NUM_QA_PAIRS = 12  # Number of QA pairs to generate per image

def extract_hierarchy_from_path(image_path: str):
    """Extract Bloom's Taxonomy level and leaf concept from folder structure."""
    folder_name = Path(image_path).parent.name
    
    if ' - ' in folder_name:
        last_separator_idx = folder_name.rfind(' - ')
        lvl1 = folder_name[:last_separator_idx].strip()
        leaf = folder_name[last_separator_idx + 3:].strip()
    else:
        lvl1 = folder_name.strip()
        leaf = folder_name.strip()
    
    # Extract Bloom level from lvl1 (e.g., "Apply - Applying a Scientific Concept" -> "Apply")
    bloom_level = lvl1.split(' - ')[0].strip() if ' - ' in lvl1 else lvl1
    
    return {"lvl1": lvl1, "leaf": leaf, "bloom_level": bloom_level, "path": f"{lvl1} -> {leaf}"}

def get_comprehensive_prompt(hierarchy: dict):
    """
    Returns a detailed prompt for generating 10 high-quality VQA pairs 
    based on Bloom's Taxonomy cognitive levels.
    """
    bloom_level = hierarchy.get("bloom_level", "Analyzing")
    leaf = hierarchy.get("leaf", "visual content")
    path = hierarchy.get("path", "")
    
    return f"""You are an expert assistant for creating Visual Question Answering (VQA) benchmarks grounded in Bloom's Taxonomy. Your task is to generate 10 high-quality, insightful question-answer pairs based on the provided image.

---
CONTEXT
We are building a benchmark to evaluate the cognitive abilities of Vision-Language Models (VLLMs). Each VQA pair must target a specific thinking skill (Bloom level) as applied to a core concept (leaf). The questions should be answerable ONLY by analyzing the visual content of the image.

---
KEY CONCEPTS FOR THIS IMAGE

- **Leaf (Core Concept)**: "{leaf}"
- **Bloom Level (Cognitive Skill)**: "{bloom_level}"
- **Full Path**: "{path}"

---
BLOOM'S TAXONOMY LEVELS & ABILITIES (Reference)

1. **Remembering** – Recall facts and basic concepts.
   - Abilities: Recognizing, Identifying, Listing, Naming, Locating
   - Example: "What color is the car?" / "How many objects are visible?"

2. **Understanding** – Explain ideas or concepts.
   - Abilities: Summarizing, Interpreting, Classifying, Explaining, Describing
   - Example: "Explain what is happening in this scene." / "What does this symbol represent?"

3. **Applying** – Use information in new situations.
   - Abilities: Executing, Implementing, Using, Demonstrating, Solving
   - Example: "What tool would you use to fix this?" / "How would you apply this concept?"

4. **Analyzing** – Draw connections among ideas, break down information.
   - Abilities: Differentiating, Organizing, Comparing, Attributing, Deconstructing
   - Example: "What is the relationship between...?" / "Compare and contrast the two objects."

5. **Evaluating** – Justify a stand or decision, make judgments.
   - Abilities: Critiquing, Justifying, Validating, Assessing, Judging
   - Example: "Is this a safe situation? Why or why not?" / "What is the quality of...?"

6. **Creating** – Produce new or original work, synthesize ideas.
   - Abilities: Designing, Planning, Producing, Constructing, Hypothesizing
   - Example: "What could be a creative caption for this image?" / "Design a solution for..."

---
GUIDELINES FOR VQA PAIR GENERATION

1. **Image-Grounded**: Both the question AND answer MUST be directly derivable from the visual information in the image. Do NOT invent information not visible in the image.

2. **Bloom Alignment**: The question must genuinely require the cognitive skill of "{bloom_level}". Ensure the question matches the complexity expected at this level.

3. **Leaf Targeting**: Each question should revolve around the "{leaf}" concept while exploring different aspects.

4. **Clarity and Conciseness**: Questions should be clear, specific, and unambiguous. Answers should be direct, complete, and informative.

5. **Variety**: Generate diverse questions that explore different aspects of the image and concept. Avoid repetitive questions.

6. **Deterministic Answers**: Answers should be objective and verifiable from the image. Avoid subjective or speculative questions.

7. **Complete Translations**: Provide COMPLETE Arabic translations for ALL questions, answers, and multiple choice options. Do NOT leave any Arabic field empty or incomplete.

8. **Multiple Choice Quality**: Create plausible distractors (wrong options) that test understanding, not just random options. Randomize the position of the correct answer (A, B, C, or D) among the choices.

---
CRITICAL INSTRUCTIONS

- Generate EXACTLY 10 complete VQA pairs.
- Each QA pair must have COMPLETE English AND Arabic content.
- Do NOT truncate or leave any field empty.
- Do NOT use double quotes (") inside content values - use single quotes (') instead.
- Do NOT generate backslashes or escape characters.
- Ensure Arabic translations are complete sentences, not fragments.

---
OUTPUT FORMAT

Return ONLY valid JSON (no markdown code blocks). Use this exact structure:

{{
    "qa_pairs": [
        {{
            "qa_index": 1,
            "bloom_level_targeted": "{bloom_level}",
            "question_en": "Your detailed English question here",
            "answer_en": "Your comprehensive English answer here",
            "question_ar": "سؤالك الكامل باللغة العربية هنا",
            "answer_ar": "إجابتك الكاملة باللغة العربية هنا",
            "mc_question_en": "Your English multiple choice question here",
            "choice_A_en": "First option in English",
            "choice_B_en": "Second option in English",
            "choice_C_en": "Third option in English",
            "choice_D_en": "Fourth option in English",
            "mc_correct_answer": "A",
            "mc_question_ar": "سؤال الاختيار من متعدد الكامل باللغة العربية",
            "choice_A_ar": "الخيار الأول بالعربية",
            "choice_B_ar": "الخيار الثاني بالعربية",
            "choice_C_ar": "الخيار الثالث بالعربية",
            "choice_D_ar": "الخيار الرابع بالعربية"
        }},
        ... (repeat for all 10 QA pairs)
    ]
}}

Now analyze the image and generate 10 high-quality VQA pairs following all guidelines above."""

def clean_json_response(text: str) -> str:
    """Extracts JSON from text, handling common LLM formatting quirks."""
    text = text.strip()
    
    # Pre-process: replace escaped quotes with single quotes to avoid backslash issues
    text = text.replace('\\"', "'")
    
    # Remove markdown code blocks if present
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0]
    elif "```" in text:
        text = text.split("```")[1].split("```")[0]
    
    # Find JSON object boundaries (most robust approach)
    # Look for outermost { } pair
    try:
        start_idx = text.index('{')
        # Find matching closing brace by counting depth
        depth = 0
        in_string = False
        escape_next = False
        end_idx = -1
        
        for i in range(start_idx, len(text)):
            char = text[i]
            
            if escape_next:
                escape_next = False
                continue
                
            if char == '\\':
                escape_next = True
                continue
                
            if char == '"' and not escape_next:
                in_string = not in_string
                
            if not in_string:
                if char == '{':
                    depth += 1
                elif char == '}':
                    depth -= 1
                    if depth == 0:
                        end_idx = i + 1
                        break
        
        if end_idx > start_idx:
            text = text[start_idx:end_idx]
    except (ValueError, IndexError):
        pass
    
    # Aggressive JSON repair strategies
    # 1. Remove trailing commas before closing braces/brackets
    text = re.sub(r',(\s*[}\]])', r'\1', text)
    
    # 2. Fix unterminated strings at end of lines
    lines = text.split('\n')
    fixed_lines = []
    for i, line in enumerate(lines):
        line_stripped = line.rstrip()
        # Count quotes, if odd and line doesn't end with quote, add one
        if line_stripped.count('"') % 2 != 0:
            # Check if the unterminated string should be closed
            if ':' in line_stripped and not line_stripped.endswith('"'):
                # Find last quote position
                last_quote = line_stripped.rfind('"')
                # Add closing quote before comma or at end
                if line_stripped.endswith(','):
                    line = line_stripped[:-1] + '",'
                else:
                    line = line_stripped + '"'
        fixed_lines.append(line)
    text = '\n'.join(fixed_lines)
    
    # 3. Ensure JSON ends properly - add missing closing braces if needed
    open_braces = text.count('{') - text.count('}')
    if open_braces > 0:
        text = text + ('}' * open_braces)
    
    # 4. Handle incomplete last field - if JSON ends abruptly, try to close it
    text = text.rstrip()
    if text and not text.endswith('}'):
        # If last char is a quote, comma, or colon, add closing brace
        if text[-1] in ['"', ',', ':']:
            text = text.rstrip(',') + '}'
        # If ends with key but no value, add empty string and closing brace
        elif text[-1] == ':':
            text = text + ' ""'
            text = text + '}'
        # If completely unterminated, try to salvage what we can
        else:
            # Check if we're in the middle of a string value
            last_colon = text.rfind(':')
            last_quote = text.rfind('"')
            if last_colon > last_quote:
                # We have a key but incomplete value, close the string and object
                text = text + '"}'
            else:
                # Just close the object
                text = text + '}'
    
    return text.strip()

def extract_fields_with_regex(text: str) -> list:
    """Fallback: extract QA pairs using regex when JSON parsing fails."""
    qa_pairs = []
    
    # Try to find individual QA pair blocks
    # Look for qa_index patterns to split the content
    qa_blocks = re.split(r'"qa_index"\s*:\s*(\d+)', text)
    
    if len(qa_blocks) > 1:
        # Process pairs of (index, content)
        for i in range(1, len(qa_blocks), 2):
            if i + 1 < len(qa_blocks):
                block_content = qa_blocks[i + 1]
                entry = extract_single_qa_regex(block_content)
                entry['qa_index'] = int(qa_blocks[i])
                if entry.get('question_en'):  # Only add if we got something
                    qa_pairs.append(entry)
    
    # If no qa_index pattern found, try to extract a single QA
    if not qa_pairs:
        entry = extract_single_qa_regex(text)
        if entry.get('question_en'):
            entry['qa_index'] = 1
            qa_pairs.append(entry)
    
    return qa_pairs

def extract_single_qa_regex(text: str) -> dict:
    """Extract a single QA pair fields using regex."""
    entry = {}
    
    patterns = {
        'bloom_level_targeted': r'"bloom_level_targeted"\s*:\s*"([^"]*?)"',
        'question_en': r'"question_en"\s*:\s*"([^"]*?)"',
        'answer_en': r'"answer_en"\s*:\s*"([^"]*?)"',
        'question_ar': r'"question_ar"\s*:\s*"([^"]*?)"',
        'answer_ar': r'"answer_ar"\s*:\s*"([^"]*?)"',
        'mc_question_en': r'"mc_question_en"\s*:\s*"([^"]*?)"',
        'choice_A_en': r'"choice_A_en"\s*:\s*"([^"]*?)"',
        'choice_B_en': r'"choice_B_en"\s*:\s*"([^"]*?)"',
        'choice_C_en': r'"choice_C_en"\s*:\s*"([^"]*?)"',
        'choice_D_en': r'"choice_D_en"\s*:\s*"([^"]*?)"',
        'mc_correct_answer': r'"mc_correct_answer"\s*:\s*"([A-D])"',
        'mc_question_ar': r'"mc_question_ar"\s*:\s*"([^"]*?)"',
        'choice_A_ar': r'"choice_A_ar"\s*:\s*"([^"]*?)"',
        'choice_B_ar': r'"choice_B_ar"\s*:\s*"([^"]*?)"',
        'choice_C_ar': r'"choice_C_ar"\s*:\s*"([^"]*?)"',
        'choice_D_ar': r'"choice_D_ar"\s*:\s*"([^"]*?)"',
    }
    
    for field, pattern in patterns.items():
        match = re.search(pattern, text, re.DOTALL)
        if match:
            value = match.group(1).strip()
            value = value.replace('\\n', ' ').replace('\\', '').strip()
            entry[field] = value
    
    return entry

def run_batch_inference(image_paths_batch, model, processor=None, max_retries=MAX_RETRIES):
    """Legacy function - kept for compatibility."""
    output_texts = []
    
    for image_path in image_paths_batch:
        hierarchy = extract_hierarchy_from_path(str(image_path))
        prompt_text = get_comprehensive_prompt(hierarchy)
        success = False
        last_error = None
        
        for attempt in range(max_retries):
            try:
                img = Image.open(image_path)
                response = model.generate_content([img, prompt_text])
                output_texts.append(response.text)
                success = True
                break
                
            except Exception as e:
                last_error = e
                wait_time = (2 ** attempt) + random.uniform(0, 1)
                
                if attempt < max_retries - 1:
                    print(f"Retry {attempt + 1}/{max_retries} for {image_path.name} after {wait_time:.1f}s (Error: {str(e)[:100]})")
                    time.sleep(wait_time)
                else:
                    print(f"Failed after {max_retries} attempts for {image_path}: {e}")
        
        if not success:
            output_texts.append('{"qa_pairs": []}')
            
    return output_texts

def process_single_image(image_path, model, max_retries=MAX_RETRIES):
    """Process a single image with retry logic - designed for parallel execution."""
    hierarchy = extract_hierarchy_from_path(str(image_path))
    prompt_text = get_comprehensive_prompt(hierarchy)
    
    for attempt in range(max_retries):
        try:
            img = Image.open(image_path)
            response = model.generate_content([img, prompt_text])
            raw_output = response.text
            
            # Validate response has QA content
            if 'question_en' in raw_output and 'answer_en' in raw_output:
                # Check for completeness - look for multiple qa_index entries
                qa_count = raw_output.count('"qa_index"')
                if qa_count >= 5:  # At least half the expected QAs
                    return image_path, raw_output, None
                elif attempt < max_retries - 1:
                    print(f"\n[!] Incomplete response for {image_path.name} (only {qa_count} QAs), retrying...")
                    wait_time = 1.0 + random.uniform(0, 0.5)
                    time.sleep(wait_time)
                    continue
                else:
                    # Accept partial response on last attempt
                    return image_path, raw_output, None
            else:
                if attempt < max_retries - 1:
                    wait_time = 1.0 + random.uniform(0, 0.5)
                    time.sleep(wait_time)
                    continue
                else:
                    return image_path, raw_output, Exception("Incomplete response after retries")
            
        except Exception as e:
            wait_time = (2 ** attempt) + random.uniform(0, 1)
            
            if attempt < max_retries - 1:
                time.sleep(wait_time)
            else:
                return image_path, '{"qa_pairs": []}', e
    
    return image_path, '{"qa_pairs": []}', Exception("Max retries exceeded")

def process_batch(image_paths_batch, model, processor=None, max_retries=MAX_RETRIES, num_threads=NUM_THREADS):
    """Process a batch of images in parallel using ThreadPoolExecutor."""
    
    # Use ThreadPoolExecutor for parallel processing
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        # Submit all tasks
        futures = {executor.submit(process_single_image, img_path, model, max_retries): img_path 
                   for img_path in image_paths_batch}
        
        # Collect results as they complete
        results = {}
        for future in as_completed(futures):
            image_path, raw_output, error = future.result()
            results[image_path] = (raw_output, error)
            if error:
                print(f"\n[!] Error for {image_path.name}: {error}")
    
    # Process results in original order
    final_entries = []
    for image_path in image_paths_batch:
        raw_output, error = results[image_path]
        cleaned_output = clean_json_response(raw_output)
        hierarchy = extract_hierarchy_from_path(str(image_path))
        
        # Default empty QA pair structure
        def get_default_qa_entry():
            return {
                "bloom_level_targeted": hierarchy.get("bloom_level", ""),
                "question_en": "", "answer_en": "",
                "question_ar": "", "answer_ar": "",
                "mc_question_en": "", "mc_correct_answer": "A",
                "mc_question_ar": "",
                "choice_A_en": "", "choice_B_en": "", "choice_C_en": "", "choice_D_en": "",
                "choice_A_ar": "", "choice_B_ar": "", "choice_C_ar": "", "choice_D_ar": ""
            }
        
        qa_pairs_data = []
        
        try:
            data = json.loads(cleaned_output)
            
            # Handle new format with qa_pairs array
            if "qa_pairs" in data and isinstance(data["qa_pairs"], list):
                qa_pairs_data = data["qa_pairs"]
            # Handle case where response is a list directly
            elif isinstance(data, list):
                qa_pairs_data = data
            # Handle old single-QA format for backwards compatibility
            elif "question_en" in data:
                qa_pairs_data = [data]
                
        except json.JSONDecodeError as e:
            print(f"\n[!] Warning: Failed to parse JSON for {image_path.name}. Attempting regex extraction. Error: {e}")
            # Fallback to regex extraction
            qa_pairs_data = extract_fields_with_regex(raw_output)
        
        # Process each QA pair
        if not qa_pairs_data:
            print(f"\n[!] Warning: No QA pairs extracted for {image_path.name}")
            # Create one empty entry to maintain structure
            qa_pairs_data = [get_default_qa_entry()]
        
        for idx, qa_data in enumerate(qa_pairs_data):
            entry = get_default_qa_entry()
            
            # Extract basic fields
            entry["bloom_level_targeted"] = qa_data.get("bloom_level_targeted", hierarchy.get("bloom_level", ""))
            entry["question_en"] = qa_data.get("question_en", "").strip()
            entry["answer_en"] = qa_data.get("answer_en", "").strip()
            entry["question_ar"] = qa_data.get("question_ar", "").strip()
            entry["answer_ar"] = qa_data.get("answer_ar", "").strip()
            entry["mc_question_en"] = qa_data.get("mc_question_en", "").strip()
            entry["mc_question_ar"] = qa_data.get("mc_question_ar", "").strip()

            # Randomize correct answer position by shuffling choices
            # 1. Collect options and their original labels
            orig_choices = [
                {"en": qa_data.get("choice_A_en", "").strip(), "ar": qa_data.get("choice_A_ar", "").strip(), "label": "A"},
                {"en": qa_data.get("choice_B_en", "").strip(), "ar": qa_data.get("choice_B_ar", "").strip(), "label": "B"},
                {"en": qa_data.get("choice_C_en", "").strip(), "ar": qa_data.get("choice_C_ar", "").strip(), "label": "C"},
                {"en": qa_data.get("choice_D_en", "").strip(), "ar": qa_data.get("choice_D_ar", "").strip(), "label": "D"}
            ]
            
            # Get original correct answer label (e.g., "A")
            orig_ans = qa_data.get("mc_correct_answer", "A")
            orig_ans_label = str(orig_ans).strip().upper()[0] if orig_ans else "A"
            if orig_ans_label not in "ABCD": orig_ans_label = "A"

            # 2. Shuffle choices (keeping English and Arabic pairs together)
            shuffled_choices = orig_choices.copy()
            random.shuffle(shuffled_choices)

            # 3. Find the new label for the correct answer
            new_ans_label = "A"
            for i, choice in enumerate(shuffled_choices):
                if choice["label"] == orig_ans_label:
                    new_ans_label = chr(65 + i)  # 0->A, 1->B, 2->C, 3->D
                    break
            
            # 4. Re-assign to entry
            entry["choice_A_en"] = shuffled_choices[0]["en"]
            entry["choice_B_en"] = shuffled_choices[1]["en"]
            entry["choice_C_en"] = shuffled_choices[2]["en"]
            entry["choice_D_en"] = shuffled_choices[3]["en"]
            
            entry["choice_A_ar"] = shuffled_choices[0]["ar"]
            entry["choice_B_ar"] = shuffled_choices[1]["ar"]
            entry["choice_C_ar"] = shuffled_choices[2]["ar"]
            entry["choice_D_ar"] = shuffled_choices[3]["ar"]
            
            entry["mc_correct_answer"] = new_ans_label
            
            # Skip entirely empty entries
            if not entry["question_en"] and not entry["answer_en"]:
                continue
            
            # Warn about incomplete Arabic translations
            if entry["question_en"] and not entry["question_ar"]:
                print(f"\n[!] Warning: Missing Arabic question translation for {image_path.name} QA #{idx+1}")
            if entry["answer_en"] and not entry["answer_ar"]:
                print(f"\n[!] Warning: Missing Arabic answer translation for {image_path.name} QA #{idx+1}")
            
            # Check for empty multiple choice options
            mc_fields = ["choice_A_en", "choice_B_en", "choice_C_en", "choice_D_en"]
            empty_mc = [f for f in mc_fields if not entry[f]]
            if empty_mc and entry["mc_question_en"]:
                print(f"\n[!] Warning: Missing MC options {empty_mc} for {image_path.name} QA #{idx+1}")

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
                "hierarchy": {"lvl1": hierarchy["lvl1"], "leaf": hierarchy["leaf"]},
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

    # CLI args
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true", help="Limit run to 5 folders for testing")
    parser.add_argument("--max-retries", type=int, default=MAX_RETRIES, help=f"Maximum number of retries for failed API calls (default: {MAX_RETRIES})")
    parser.add_argument("--threads", type=int, default=NUM_THREADS, help=f"Number of parallel threads for API calls (default: {NUM_THREADS})")
    parser.add_argument("--num-qa", type=int, default=NUM_QA_PAIRS, help=f"Number of QA pairs to generate per image (default: {NUM_QA_PAIRS})")
    args = parser.parse_args()
    
    max_retries = args.max_retries
    num_threads = args.threads

    print(f"Found {len(image_paths)} images in {DATASET_DIR}")
    if len(image_paths) == 0:
        raise SystemExit("No images found in dataset directory")

    # If test flag set, limit to first 5 unique parent folders (preserve ordering)
    if args.test:
        selected_folders = []
        filtered = []
        for p in image_paths:
            parent = p.parent
            if parent not in selected_folders:
                if len(selected_folders) < 5:
                    selected_folders.append(parent)
                else:
                    # skip images from folders beyond the first 5
                    continue
            if parent in selected_folders:
                filtered.append(p)
        print(f"Test mode: limiting to {len(selected_folders)} folders:")
        for f in selected_folders:
            print(f" - {f}")
        image_paths = filtered

    model_name = "gemini-3-flash-preview"
    print(f"Initializing model {model_name}...")
    print(f"Using {num_threads} parallel threads with {max_retries} max retries per image")
    print(f"Generating {args.num_qa} QA pairs per image based on Bloom's Taxonomy")
    
    generation_config = {
        "temperature": 0.0,
        "max_output_tokens": GEN_MAX_TOKENS,
        "response_mime_type": "application/json",
    }
    
    model = genai.GenerativeModel(
        model_name=model_name,
        generation_config=generation_config
    )

    qa_dataset = []
    total = len(image_paths)
    
    # Process in larger batches for threading efficiency
    batch_size = num_threads * 2

    with tqdm(total=total, desc="Processing images (Gemini)", unit="image") as pbar:
        for i in range(0, total, batch_size):
            batch_paths = image_paths[i : i + batch_size]
            
            try:
                batch_entries = process_batch(batch_paths, model, processor=None, max_retries=max_retries, num_threads=num_threads)
                qa_dataset.extend(batch_entries)
                pbar.update(len(batch_paths))

                # Save checkpoint
                with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
                    json.dump(qa_dataset, f, ensure_ascii=False, indent=2)

            except Exception as e:
                print(f"\n[!] Error processing batch starting at {i}: {e}")
                import traceback
                traceback.print_exc()

    # Print summary statistics
    total_qa = len(qa_dataset)
    unique_images = len(set(entry["source_image_file"] for entry in qa_dataset))
    avg_qa_per_image = total_qa / unique_images if unique_images > 0 else 0
    
    # Count entries with missing Arabic
    missing_ar_q = sum(1 for e in qa_dataset if e["question_en"] and not e["question_ar"])
    missing_ar_a = sum(1 for e in qa_dataset if e["answer_en"] and not e["answer_ar"])
    
    print(f"\n{'='*60}")
    print(f"GENERATION COMPLETE")
    print(f"{'='*60}")
    print(f"Total QA pairs generated: {total_qa}")
    print(f"Unique images processed: {unique_images}")
    print(f"Average QA pairs per image: {avg_qa_per_image:.1f}")
    print(f"Entries missing Arabic question: {missing_ar_q}")
    print(f"Entries missing Arabic answer: {missing_ar_a}")
    print(f"Output saved to: {OUTPUT_FILE}")
    print(f"{'='*60}")