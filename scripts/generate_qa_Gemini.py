
"""Generate QA dataset from images using Gemini 3 Flash.

Optimized version: Generates all content (QA + Translations + Multiple Choice) 
in a SINGLE model pass per image to maximize throughput and efficiency.
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
GEN_MAX_TOKENS = 3072  # Increased to prevent truncation of Arabic translations
BATCH_SIZE = 100  # Sequential processing for Gemini API
MAX_RETRIES = 4  # Default number of retries for failed API calls
NUM_THREADS = 7  # Default number of parallel threads for API calls

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

IMPORTANT: Do not use double quotes (") inside the content values. Use single quotes (') instead if needed. Do not generate backslashes.

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

def extract_fields_with_regex(text: str) -> dict:
    """Fallback: extract fields using regex when JSON parsing fails."""
    entry = {}
    
    # More lenient patterns that handle quotes, newlines, and incomplete strings
    patterns = {
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
            # Clean up any remaining escape sequences or malformed content
            value = value.replace('\\n', ' ').replace('\\', '').strip()
            entry[field] = value
    
    return entry

def run_batch_inference(image_paths_batch, model, processor=None, max_retries=MAX_RETRIES):
    prompt_text = get_comprehensive_prompt()
    output_texts = []
    
    for image_path in image_paths_batch:
        success = False
        last_error = None
        
        for attempt in range(max_retries):
            try:
                # Load image using PIL
                img = Image.open(image_path)
                
                # Generate content
                response = model.generate_content([img, prompt_text])
                output_texts.append(response.text)
                success = True
                break
                
            except Exception as e:
                last_error = e
                wait_time = (2 ** attempt) + random.uniform(0, 1)  # Exponential backoff with jitter
                
                if attempt < max_retries - 1:
                    print(f"Retry {attempt + 1}/{max_retries} for {image_path.name} after {wait_time:.1f}s (Error: {str(e)[:100]})")
                    time.sleep(wait_time)
                else:
                    print(f"Failed after {max_retries} attempts for {image_path}: {e}")
        
        if not success:
            output_texts.append("{}")
            
    return output_texts

def process_single_image(image_path, model, max_retries=MAX_RETRIES):
    """Process a single image with retry logic - designed for parallel execution."""
    prompt_text = get_comprehensive_prompt()
    
    for attempt in range(max_retries):
        try:
            # Load image using PIL
            img = Image.open(image_path)
            
            # Generate content
            response = model.generate_content([img, prompt_text])
            raw_output = response.text
            
            # Quick validation: check if response looks complete
            # At minimum, we need question_en and answer_en (Arabic will be extracted if present)
            if 'question_en' in raw_output and 'answer_en' in raw_output:
                return image_path, raw_output, None
            else:
                # Incomplete response, retry
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
                return image_path, "{}", e
    
    return image_path, "{}", Exception("Max retries exceeded")

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
            print(f"\n[!] Warning: Failed to parse JSON for {image_path.name}. Attempting regex extraction. Error: {e}")
            # Fallback to regex extraction
            regex_data = extract_fields_with_regex(raw_output)
            for key, value in regex_data.items():
                if value:  # Only update if we extracted something
                    entry[key] = value

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

    # CLI args
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true", help="Limit run to 5 folders for testing")
    parser.add_argument("--max-retries", type=int, default=MAX_RETRIES, help=f"Maximum number of retries for failed API calls (default: {MAX_RETRIES})")
    parser.add_argument("--threads", type=int, default=NUM_THREADS, help=f"Number of parallel threads for API calls (default: {NUM_THREADS})")
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

    print(f"Done. Saved {len(qa_dataset)} entries to {OUTPUT_FILE}")