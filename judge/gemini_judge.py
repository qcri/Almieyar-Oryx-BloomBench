#!/usr/bin/env python3
"""
Gemini-3-Pro Analysis Script for Visual QA Dataset
Judges question and answer quality in both English and Arabic from final_oryx_v2.json
Samples 2 random items per hierarchy level and outputs JSON evaluation
"""

import pandas as pd
import json
import os
import time
import random
from pathlib import Path
from tqdm import tqdm
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
import threading
import traceback
from PIL import Image

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("✅ Loaded .env file")
except ImportError:
    print("⚠️ python-dotenv not installed, using system environment variables only")

# Google Gemini API setup
try:
    import google.generativeai as genai
    
    # Gemini API configuration
    gemini_api_key = os.environ.get("GEMINI_API_KEY", "")
    genai.configure(api_key=gemini_api_key)
    
    print("✅ Gemini API configured")
except ImportError:
    print("❌ Error: google-generativeai package not installed. Install with: pip install google-generativeai")
    exit(1)

# Configuration
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3-pro-preview")  # Gemini-3-Pro supports vision
NUM_THREADS = 20  
LOG_FILE = 'gemini3_analysis_log.txt'
INPUT_FILE = 'final_oryx_v2.json'
OUTPUT_FILE = 'gemini3_oryx_evaluation.json'
RETRY_DELAY = 10  # Delay between retries
MAX_RETRIES = 15  # More retries
REQUEST_DELAY = 3.0  # 3 seconds between requests
SAMPLES_PER_LEVEL = 10  # Number of random samples per hierarchy level

# Thread-safe lock for writing to shared resources
write_lock = threading.Lock()


def load_image_for_gemini(image_path):
    """Load image for Gemini API."""
    try:
        img = Image.open(image_path)
        return img
    except Exception as e:
        print(f"⚠️ Error loading image {image_path}: {e}")
        return None


def locate_image(path_candidate):
    """Try multiple locations to find the image file on disk."""
    if not path_candidate or str(path_candidate).strip() == '':
        return None
    p = Path(str(path_candidate))
    if p.exists():
        return p
    # try inside uploaded_files_dev
    p2 = Path('uploaded_files_dev') / p.name
    if p2.exists():
        return p2
    # try relative to script
    p3 = Path.cwd() / p
    if p3.exists():
        return p3
    return None


def analyze_with_gemini(item):
    """
    Use Gemini-3-Pro to analyze question and answer quality in both English and Arabic.
    
    Args:
        item: Dictionary with question, answer, image, hierarchy info
    
    Returns: dict with keys:
        - english_question_score: 1-5 rating
        - english_answer_score: 1-5 rating
        - arabic_question_score: 1-5 rating (if available)
        - arabic_answer_score: 1-5 rating (if available)
        - english_question_feedback: str
        - english_answer_feedback: str
        - arabic_question_feedback: str (if available)
        - arabic_answer_feedback: str (if available)
        - overall_quality_score: 1-5 rating
        - improvement_suggestions: str
    """
    
    # CRITICAL: Add delay BEFORE making request to spread out requests over time
    time.sleep(REQUEST_DELAY)
    
    # Extract data from item
    question_en = item.get('question_en', '')
    answer_en = item.get('answer_en', '')
    question_ar = item.get('question_ar', '')
    answer_ar = item.get('answer_ar', '')
    image_path = item.get('source_image_file', '')
    hierarchy = item.get('hierarchy', {})
    
    # Build multiple choice context if available
    mc_context = ""
    if 'multiple_choice_qa' in item:
        mc = item['multiple_choice_qa']
        mc_context = f"""

**Multiple Choice Format:**
Question (EN): {mc.get('question_en', '')}
Question (AR): {mc.get('question_ar', '')}
Choices:
  A (EN): {mc.get('choice_A_en', '')} | (AR): {mc.get('choice_A_ar', '')}
  B (EN): {mc.get('choice_B_en', '')} | (AR): {mc.get('choice_B_ar', '')}
  C (EN): {mc.get('choice_C_en', '')} | (AR): {mc.get('choice_C_ar', '')}
  D (EN): {mc.get('choice_D_en', '')} | (AR): {mc.get('choice_D_ar', '')}
Correct Answer: {mc.get('answer', '')}
"""
    
    analysis_prompt = f"""You are an expert LLM evaluating a Visual Question Answering (VQA) dataset. Your task is to determine if the question-answer pairs make sense and are correct from your perspective as a language model with vision capabilities.

**Image Context:** Look at the provided image.
**Hierarchy Level:** {hierarchy.get('lvl1', 'Unknown')}
**Leaf Category:** {hierarchy.get('leaf', 'Unknown')}

**English Question:** {question_en if question_en else 'N/A'}
**English Answer:** {answer_en if answer_en else 'N/A'}

**Arabic Question:** {question_ar if question_ar else 'N/A'}
**Arabic Answer:** {answer_ar if answer_ar else 'N/A'}
{mc_context}

**Task:**
For each question-answer pair, determine:
1. Does the question make sense for this image?
2. Is the provided answer correct and appropriate?
3. Do you AGREE or DISAGREE with this Q&A pair?

**Output Format (MUST follow exactly):**

ENGLISH_QA_JUDGMENT: [AGREE or DISAGREE]
ENGLISH_QA_REASONING: [Brief explanation of why you agree or disagree]

ARABIC_QA_JUDGMENT: [AGREE or DISAGREE or N/A if not available]
ARABIC_QA_REASONING: [Brief explanation or N/A]

OVERALL_JUDGMENT: [AGREE or DISAGREE]
DISAGREEMENT_REASON: [If DISAGREE, explain what's wrong. If AGREE, write 'N/A']
"""

    try:
        # Prepare content for Gemini
        system_instruction = "You are an expert LLM with vision capabilities. Evaluate VQA dataset items by determining if you AGREE or DISAGREE with the provided question-answer pairs. Be honest about what makes sense from your perspective."
        
        # Create model with system instruction
        model = genai.GenerativeModel(
            model_name=GEMINI_MODEL,
            system_instruction=system_instruction
        )
        
        # Prepare content (text + image if available)
        content_parts = [analysis_prompt]
        
        # Add image if available
        if image_path and str(image_path).strip():
            located = locate_image(image_path)
            if located is not None:
                img = load_image_for_gemini(located)
                if img:
                    content_parts.append(img)
                    print(f"    📸 Image loaded: {Path(located).name}")
                else:
                    print(f"    ⚠️ Failed to load image: {located}")
            else:
                print(f"    ⚠️ Image not found: {image_path}")
        else:
            print(f"    ℹ️ No image provided for this item")
        
        # Call Gemini API with retry logic for rate limits
        for attempt in range(MAX_RETRIES):
            try:
                response = model.generate_content(
                    content_parts,
                    generation_config=genai.types.GenerationConfig(
                        temperature=0.1,
                        max_output_tokens=2000,
                    ),
                    safety_settings=[
                        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                        {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
                    ]
                )
                break  # Success, exit retry loop
            except Exception as api_error:
                error_str = str(api_error)
                if "429" in error_str or "quota" in error_str.lower() or "rate" in error_str.lower():
                    if attempt < MAX_RETRIES - 1:
                        base_wait = RETRY_DELAY * (2 ** attempt)
                        jitter = random.uniform(0, 5)
                        wait_time = min(base_wait + jitter, 180)
                        print(f"⏳ [Rate Limit] Waiting {wait_time:.1f}s before retry {attempt + 2}/{MAX_RETRIES}...")
                        time.sleep(wait_time)
                    else:
                        print(f"❌ Max retries ({MAX_RETRIES}) exceeded for rate limit")
                        raise
                else:
                    raise
        
        # Check if response was blocked by safety filters
        if response.candidates and response.candidates[0].finish_reason != 1:  # 1 = STOP (normal completion)
            finish_reason = response.candidates[0].finish_reason
            finish_reasons = {
                2: "SAFETY (blocked by safety filters)",
                3: "RECITATION (blocked by recitation filters)", 
                4: "OTHER",
                5: "MAX_TOKENS"
            }
            reason_text = finish_reasons.get(finish_reason, f"Unknown ({finish_reason})")
            print(f"    ⚠️ Response blocked: {reason_text}")
            
            # Try to get partial text or return informative error
            try:
                result_text = response.text.strip()
            except:
                return {
                    'english_qa_judgment': 'BLOCKED',
                    'english_qa_reasoning': f"Content blocked by Gemini: {reason_text}",
                    'arabic_qa_judgment': 'BLOCKED',
                    'arabic_qa_reasoning': f"Content blocked by Gemini: {reason_text}",
                    'overall_judgment': 'BLOCKED',
                    'disagreement_reason': f"Content blocked by safety filters: {reason_text}"
                }
        
        result_text = response.text.strip()
        
        # Parse the structured response
        result = parse_gemini_response(result_text)
        
        # IMPORTANT: Delay between requests to respect rate limits
        time.sleep(REQUEST_DELAY)
        
        return result
        
    except Exception as e:
        print(f"⚠️ Error calling Gemini API: {e}")
        traceback.print_exc()
        return {
            'english_qa_judgment': 'ERROR',
            'english_qa_reasoning': f"Error: {str(e)}",
            'arabic_qa_judgment': 'ERROR',
            'arabic_qa_reasoning': f"Error: {str(e)}",
            'overall_judgment': 'ERROR',
            'disagreement_reason': f"Error: {str(e)}"
        }


def parse_gemini_response(text):
    """Parse Gemini's structured response for VQA evaluation."""
    result = {
        'english_qa_judgment': None,
        'english_qa_reasoning': '',
        'arabic_qa_judgment': None,
        'arabic_qa_reasoning': '',
        'overall_judgment': None,
        'disagreement_reason': ''
    }
    
    lines = text.split('\n')
    current_field = None
    feedback_buffer = []
    
    for line in lines:
        line_stripped = line.strip()
        if not line_stripped:
            continue
        
        # Check for judgment fields
        if 'ENGLISH_QA_JUDGMENT:' in line:
            judgment = line.split(':', 1)[1].strip().upper()
            if 'AGREE' in judgment and 'DISAGREE' not in judgment:
                result['english_qa_judgment'] = 'AGREE'
            elif 'DISAGREE' in judgment:
                result['english_qa_judgment'] = 'DISAGREE'
            else:
                result['english_qa_judgment'] = judgment
            current_field = None
        elif 'ENGLISH_QA_REASONING:' in line:
            reasoning = line.split(':', 1)[1].strip() if ':' in line else ''
            current_field = 'english_qa_reasoning'
            feedback_buffer = [reasoning] if reasoning else []
        elif 'ARABIC_QA_JUDGMENT:' in line:
            if current_field == 'english_qa_reasoning':
                result['english_qa_reasoning'] = ' '.join(feedback_buffer)
            judgment = line.split(':', 1)[1].strip().upper()
            if 'N/A' in judgment:
                result['arabic_qa_judgment'] = 'N/A'
            elif 'AGREE' in judgment and 'DISAGREE' not in judgment:
                result['arabic_qa_judgment'] = 'AGREE'
            elif 'DISAGREE' in judgment:
                result['arabic_qa_judgment'] = 'DISAGREE'
            else:
                result['arabic_qa_judgment'] = judgment
            current_field = None
        elif 'ARABIC_QA_REASONING:' in line:
            reasoning = line.split(':', 1)[1].strip() if ':' in line else ''
            current_field = 'arabic_qa_reasoning'
            feedback_buffer = [reasoning] if reasoning else []
        elif 'OVERALL_JUDGMENT:' in line:
            if current_field == 'arabic_qa_reasoning':
                result['arabic_qa_reasoning'] = ' '.join(feedback_buffer)
            judgment = line.split(':', 1)[1].strip().upper()
            if 'AGREE' in judgment and 'DISAGREE' not in judgment:
                result['overall_judgment'] = 'AGREE'
            elif 'DISAGREE' in judgment:
                result['overall_judgment'] = 'DISAGREE'
            else:
                result['overall_judgment'] = judgment
            current_field = None
        elif 'DISAGREEMENT_REASON:' in line:
            reason = line.split(':', 1)[1].strip() if ':' in line else ''
            current_field = 'disagreement_reason'
            feedback_buffer = [reason] if reason else []
        elif current_field:
            # Continuation of feedback
            feedback_buffer.append(line_stripped)
    
    # Capture any remaining feedback
    if current_field == 'disagreement_reason':
        result['disagreement_reason'] = ' '.join(feedback_buffer)
    
    return result


def main():
    """Main execution function."""
    print("="*80)
    print("GEMINI-3-PRO VQA DATASET EVALUATION TOOL")
    print("="*80)
    print(f"Model: {GEMINI_MODEL}")
    print(f"API Key: {'Set' if os.environ.get('GEMINI_API_KEY') else 'Using default'}")
    print(f"Input: {INPUT_FILE}")
    print(f"Output: {OUTPUT_FILE}")
    print(f"Samples per hierarchy level: {SAMPLES_PER_LEVEL}")
    print(f"Request delay: {REQUEST_DELAY}s between requests")
    print("="*80)
    
    # Check for Gemini API key
    if not os.environ.get("GEMINI_API_KEY") and not gemini_api_key:
        print("\n❌ ERROR: GEMINI_API_KEY environment variable not set!")
        print("Please set it with: export GEMINI_API_KEY='your-key-here'")
        return
    
    # Load JSON dataset
    print(f"\n📂 Loading {INPUT_FILE}...")
    try:
        with open(INPUT_FILE, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
        print(f"✅ Loaded {len(dataset):,} items from dataset")
    except Exception as e:
        print(f"❌ Error loading {INPUT_FILE}: {e}")
        return
    
    # Group by hierarchy level (lvl1 + leaf combination)
    print("\n📊 Grouping by hierarchy levels...")
    hierarchy_groups = defaultdict(list)
    for item in dataset:
        if 'hierarchy' in item and 'lvl1' in item['hierarchy']:
            lvl1 = item['hierarchy']['lvl1']
            leaf = item['hierarchy'].get('leaf', 'Unknown')
            # Create unique key from lvl1 + leaf combination
            hierarchy_key = f"{lvl1} -> {leaf}"
            hierarchy_groups[hierarchy_key].append(item)
    
    print(f"✅ Found {len(hierarchy_groups)} unique hierarchy level combinations (lvl1 + leaf)")
    
    # Sample items from each level
    print(f"\n🎲 Sampling {SAMPLES_PER_LEVEL} items per level combination...")
    sampled_items = []
    level_samples = {}
    
    for hierarchy_key, items in hierarchy_groups.items():
        # Sample up to SAMPLES_PER_LEVEL random items
        sample_size = min(SAMPLES_PER_LEVEL, len(items))
        samples = random.sample(items, sample_size)
        sampled_items.extend(samples)
        level_samples[hierarchy_key] = samples
        print(f"  • {hierarchy_key}: sampled {sample_size} items (from {len(items)} total)")
    
    print(f"\n✅ Total items to evaluate: {len(sampled_items)}")
    
    # Estimate time
    estimated_time_sec = len(sampled_items) * REQUEST_DELAY
    estimated_time_min = estimated_time_sec / 60
    print(f"⏱️ Estimated time: ~{estimated_time_min:.1f} minutes")
    
    # Process each sampled item
    print(f"\n🚀 Starting Gemini-3-Pro evaluation with {NUM_THREADS} threads...")
    results_by_level = {}
    
    def process_item(item):
        """Process a single item and return results."""
        lvl1 = item.get('hierarchy', {}).get('lvl1', 'Unknown')
        leaf = item.get('hierarchy', {}).get('leaf', 'Unknown')
        hierarchy_key = f"{lvl1} -> {leaf}"
        analysis = analyze_with_gemini(item)
        
        return {
            'hierarchy_key': hierarchy_key,
            'lvl1': lvl1,
            'item_id': item.get('question_id', 'unknown'),
            'image_id': item.get('image_id', 'unknown'),
            'leaf_category': leaf,
            'question_en': item.get('question_en', ''),
            'answer_en': item.get('answer_en', ''),
            'question_ar': item.get('question_ar', ''),
            'answer_ar': item.get('answer_ar', ''),
            'has_multiple_choice': 'multiple_choice_qa' in item,
            'judgments': {
                'english_qa': analysis['english_qa_judgment'],
                'arabic_qa': analysis['arabic_qa_judgment'],
                'overall': analysis['overall_judgment']
            },
            'reasoning': {
                'english_qa': analysis['english_qa_reasoning'],
                'arabic_qa': analysis['arabic_qa_reasoning'],
                'disagreement': analysis['disagreement_reason']
            }
        }
    
    # Use ThreadPoolExecutor for parallel processing
    with ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
        # Submit all tasks
        futures = {executor.submit(process_item, item): item for item in sampled_items}
        
        # Process completed tasks with progress bar
        with tqdm(total=len(sampled_items), desc="Evaluating items") as pbar:
            for future in as_completed(futures):
                try:
                    result = future.result()
                    hierarchy_key = result['hierarchy_key']
                    
                    # Thread-safe result storage
                    with write_lock:
                        if hierarchy_key not in results_by_level:
                            results_by_level[hierarchy_key] = {
                                'level': hierarchy_key,
                                'lvl1': result['lvl1'],
                                'leaf': result['leaf_category'],
                                'samples_evaluated': 0,
                                'evaluations': []
                            }
                        
                        results_by_level[hierarchy_key]['samples_evaluated'] += 1
                        results_by_level[hierarchy_key]['evaluations'].append({
                            'item_id': result['item_id'],
                            'image_id': result['image_id'],
                            'leaf_category': result['leaf_category'],
                            'question_en': result['question_en'],
                            'answer_en': result['answer_en'],
                            'question_ar': result['question_ar'],
                            'answer_ar': result['answer_ar'],
                            'has_multiple_choice': result['has_multiple_choice'],
                            'judgments': result['judgments'],
                            'reasoning': result['reasoning']
                        })
                        
                        # Log progress every 10 items
                        total_processed = sum(data['samples_evaluated'] for data in results_by_level.values())
                        if total_processed % 10 == 0:
                            timestamp = datetime.now(timezone.utc).isoformat()
                            with open(LOG_FILE, 'a', encoding='utf-8') as lf:
                                lf.write(f"{timestamp} - Processed {total_processed}/{len(sampled_items)} items\n")
                
                except Exception as e:
                    print(f"\n⚠️ ERROR processing item: {e}")
                    traceback.print_exc()
                finally:
                    pbar.update(1)
    
    # Calculate aggregate statistics per level
    print("\n📊 Calculating level statistics...")
    for hierarchy_key, data in results_by_level.items():
        evals = data['evaluations']
        
        # Calculate agreement/disagreement counts
        en_agrees = len([e for e in evals if e['judgments']['english_qa'] == 'AGREE'])
        ar_agrees = len([e for e in evals if e['judgments']['arabic_qa'] == 'AGREE'])
        overall_agrees = len([e for e in evals if e['judgments']['overall'] == 'AGREE'])
        overall_disagrees = len([e for e in evals if e['judgments']['overall'] == 'DISAGREE'])
        
        total = len(evals)
        data['statistics'] = {
            'total_samples': total,
            'overall_agreement': overall_agrees,
            'overall_disagreement': overall_disagrees,
            'agreement_percentage': (overall_agrees / total * 100) if total > 0 else 0,
            'english_qa_agreement': en_agrees,
            'english_qa_agreement_percentage': (en_agrees / total * 100) if total > 0 else 0,
            'arabic_qa_agreement': ar_agrees,
            'num_with_arabic': len([e for e in evals if e['question_ar'] or e['answer_ar']]),
            'num_with_multiple_choice': len([e for e in evals if e['has_multiple_choice']])
        }
    
    # Build evaluation summary for JSON
    summary_by_level = []
    total_agreement = 0
    total_evaluated = 0
    
    for hierarchy_key in sorted(results_by_level.keys()):
        level_data = results_by_level[hierarchy_key]
        stats = level_data['statistics']
        total_agreement += stats['overall_agreement']
        total_evaluated += stats['total_samples']
        
        summary_by_level.append({
            'hierarchy_level': hierarchy_key,
            'lvl1': level_data['lvl1'],
            'leaf': level_data['leaf'],
            'samples_evaluated': level_data['samples_evaluated'],
            'agreement': stats['overall_agreement'],
            'disagreement': stats['overall_disagreement'],
            'agreement_percentage': round(stats['agreement_percentage'], 1),
            'num_with_arabic': stats['num_with_arabic'],
            'num_with_multiple_choice': stats['num_with_multiple_choice']
        })
    
    # Calculate overall agreement percentage
    overall_agreement_pct = (total_agreement / total_evaluated * 100) if total_evaluated > 0 else 0
    
    # Save results to JSON
    print(f"\n💾 Saving results to {OUTPUT_FILE}...")
    output_data = {
        'evaluation_summary': {
            'total_hierarchy_levels_evaluated': len(results_by_level),
            'total_items_evaluated': total_evaluated,
            'total_agreement': total_agreement,
            'total_disagreement': total_evaluated - total_agreement,
            'overall_agreement_percentage': round(overall_agreement_pct, 2),
            'evaluation_date': datetime.now(timezone.utc).isoformat(),
            'model': GEMINI_MODEL,
            'agreement_by_level': summary_by_level
        },
        'metadata': {
            'evaluation_date': datetime.now(timezone.utc).isoformat(),
            'model': GEMINI_MODEL,
            'input_file': INPUT_FILE,
            'total_dataset_items': len(dataset),
            'total_hierarchy_levels': len(hierarchy_groups),
            'samples_per_level': SAMPLES_PER_LEVEL,
            'total_items_evaluated': len(sampled_items)
        },
        'results_by_level': results_by_level
    }
    
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    print(f"✅ Results saved to {OUTPUT_FILE}")
    
    # Print summary
    print(f"\n{'='*80}")
    print("EVALUATION SUMMARY")
    print(f"{'='*80}")
    print(f"Total hierarchy levels evaluated: {len(results_by_level)}")
    print(f"Total items evaluated: {total_evaluated}")
    print(f"Overall Agreement: {total_agreement}/{total_evaluated} ({overall_agreement_pct:.1f}%)")
    print(f"Overall Disagreement: {total_evaluated - total_agreement}/{total_evaluated} ({100 - overall_agreement_pct:.1f}%)")
    print(f"\nAgreement by Level (lvl1 -> leaf):")
    print(f"{'Hierarchy Level':<75} {'Samples':<10} {'Agree':<10} {'Disagree':<12} {'Agreement %':<15}")
    print("-" * 130)
    
    for hierarchy_key in sorted(results_by_level.keys()):
        stats = results_by_level[hierarchy_key]['statistics']
        samples = results_by_level[hierarchy_key]['samples_evaluated']
        agree = stats['overall_agreement']
        disagree = stats['overall_disagreement']
        agree_pct = f"{stats['agreement_percentage']:.1f}%"
        
        # Truncate level name if too long
        level_name = hierarchy_key[:72] + "..." if len(hierarchy_key) > 75 else hierarchy_key
        print(f"{level_name:<75} {samples:<10} {agree:<10} {disagree:<12} {agree_pct:<15}")
    
    print(f"\n{'='*80}")
    print("✅ EVALUATION COMPLETE!")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
