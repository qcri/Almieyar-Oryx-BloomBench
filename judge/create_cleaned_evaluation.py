#!/usr/bin/env python3
"""
Create cleaned evaluation JSON by:
1. Excluding BLOCKED responses
2. Recalculating percentages
3. Adding high-level category statistics
"""

import json
from collections import defaultdict
from datetime import datetime, timezone

# Load the original evaluation results
with open('gemini3_oryx_evaluation.json', 'r', encoding='utf-8') as f:
    original_data = json.load(f)

print("="*80)
print("CREATING CLEANED EVALUATION RESULTS")
print("="*80)

# Extract high-level category (first word before ->)
def get_high_level_category(hierarchy_key):
    return hierarchy_key.split(' ->')[0].strip()

# Initialize cleaned results
cleaned_results_by_level = {}
high_level_stats = defaultdict(lambda: {
    'total_samples': 0,
    'total_blocked': 0,
    'total_agree': 0,
    'total_disagree': 0,
    'agreement_percentage': 0
})

# Process each hierarchy level
for hierarchy_key, level_data in original_data['results_by_level'].items():
    high_level = get_high_level_category(hierarchy_key)
    
    # Filter out BLOCKED evaluations
    valid_evaluations = []
    blocked_count = 0
    agree_count = 0
    disagree_count = 0
    
    for eval_item in level_data['evaluations']:
        if eval_item['judgments']['overall'] == 'BLOCKED':
            blocked_count += 1
            high_level_stats[high_level]['total_blocked'] += 1
        else:
            valid_evaluations.append(eval_item)
            high_level_stats[high_level]['total_samples'] += 1
            
            if eval_item['judgments']['overall'] == 'AGREE':
                agree_count += 1
                high_level_stats[high_level]['total_agree'] += 1
            elif eval_item['judgments']['overall'] == 'DISAGREE':
                disagree_count += 1
                high_level_stats[high_level]['total_disagree'] += 1
    
    # Only include level if it has valid evaluations
    if valid_evaluations:
        total_valid = len(valid_evaluations)
        agreement_pct = (agree_count / total_valid * 100) if total_valid > 0 else 0
        
        cleaned_results_by_level[hierarchy_key] = {
            'level': hierarchy_key,
            'lvl1': level_data['lvl1'],
            'leaf': level_data['leaf'],
            'samples_evaluated': level_data['samples_evaluated'],
            'valid_samples': total_valid,
            'blocked_samples': blocked_count,
            'samples_evaluated': 0,
            'evaluations': valid_evaluations,
            'statistics': {
                'total_samples': total_valid,
                'overall_agreement': agree_count,
                'overall_disagreement': disagree_count,
                'agreement_percentage': agreement_pct,
                'english_qa_agreement': len([e for e in valid_evaluations if e['judgments']['english_qa'] == 'AGREE']),
                'english_qa_agreement_percentage': len([e for e in valid_evaluations if e['judgments']['english_qa'] == 'AGREE']) / total_valid * 100 if total_valid > 0 else 0,
                'arabic_qa_agreement': len([e for e in valid_evaluations if e['judgments']['arabic_qa'] == 'AGREE']),
                'num_with_arabic': len([e for e in valid_evaluations if e['question_ar'] or e['answer_ar']]),
                'num_with_multiple_choice': len([e for e in valid_evaluations if e['has_multiple_choice']])
            }
        }

# Calculate high-level category statistics
for category, stats in high_level_stats.items():
    total = stats['total_samples']
    if total > 0:
        stats['agreement_percentage'] = (stats['total_agree'] / total * 100)

# Build evaluation summary
total_valid = sum(s['total_samples'] for s in high_level_stats.values())
total_agree = sum(s['total_agree'] for s in high_level_stats.values())
total_disagree = sum(s['total_disagree'] for s in high_level_stats.values())
total_blocked = sum(s['total_blocked'] for s in high_level_stats.values())
overall_agreement_pct = (total_agree / total_valid * 100) if total_valid > 0 else 0

# Build summary by level
summary_by_level = []
for hierarchy_key in sorted(cleaned_results_by_level.keys()):
    level_data = cleaned_results_by_level[hierarchy_key]
    stats = level_data['statistics']
    
    summary_by_level.append({
        'hierarchy_level': hierarchy_key,
        'lvl1': level_data['lvl1'],
        'leaf': level_data['leaf'],
        'total_samples': level_data['samples_evaluated'],
        'valid_samples': stats['total_samples'],
        'blocked_samples': level_data['blocked_samples'],
        'agreement': stats['overall_agreement'],
        'disagreement': stats['overall_disagreement'],
        'agreement_percentage': round(stats['agreement_percentage'], 2),
        'num_with_arabic': stats['num_with_arabic'],
        'num_with_multiple_choice': stats['num_with_multiple_choice']
    })

# Build high-level category summary
high_level_summary = []
for category in sorted(high_level_stats.keys()):
    stats = high_level_stats[category]
    high_level_summary.append({
        'category': category,
        'valid_samples': stats['total_samples'],
        'blocked_samples': stats['total_blocked'],
        'agreement': stats['total_agree'],
        'disagreement': stats['total_disagree'],
        'agreement_percentage': round(stats['agreement_percentage'], 2)
    })

# Create cleaned output data
cleaned_data = {
    'evaluation_summary': {
        'total_hierarchy_levels_evaluated': len(cleaned_results_by_level),
        'total_items_evaluated': original_data['metadata']['total_items_evaluated'],
        'total_blocked_responses': total_blocked,
        'total_valid_responses': total_valid,
        'total_agreement': total_agree,
        'total_disagreement': total_disagree,
        'overall_agreement_percentage': round(overall_agreement_pct, 2),
        'evaluation_date': datetime.now(timezone.utc).isoformat(),
        'model': original_data['metadata']['model'],
        'high_level_categories': high_level_summary,
        'detailed_by_level': summary_by_level
    },
    'metadata': {
        'evaluation_date': datetime.now(timezone.utc).isoformat(),
        'model': original_data['metadata']['model'],
        'input_file': original_data['metadata']['input_file'],
        'total_dataset_items': original_data['metadata']['total_dataset_items'],
        'total_hierarchy_levels': original_data['metadata']['total_hierarchy_levels'],
        'samples_per_level': original_data['metadata']['samples_per_level'],
        'total_items_evaluated': original_data['metadata']['total_items_evaluated'],
        'blocked_responses_excluded': True,
        'original_file': 'gemini3_oryx_evaluation.json'
    },
    'results_by_level': cleaned_results_by_level
}

# Save cleaned results
output_file = 'gemini3_oryx_evaluation_CLEANED.json'
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(cleaned_data, f, ensure_ascii=False, indent=2)

print(f"\n✅ Cleaned evaluation saved to: {output_file}")

# Print summary
print(f"\n{'='*80}")
print("CLEANED EVALUATION SUMMARY")
print(f"{'='*80}")
print(f"Original evaluations: {original_data['metadata']['total_items_evaluated']}")
print(f"Blocked responses: {total_blocked} ({total_blocked/original_data['metadata']['total_items_evaluated']*100:.2f}%)")
print(f"Valid responses: {total_valid}")
print(f"  - Agreement: {total_agree} ({overall_agreement_pct:.2f}%)")
print(f"  - Disagreement: {total_disagree} ({total_disagree/total_valid*100:.2f}%)")

print(f"\n{'='*80}")
print("HIGH-LEVEL CATEGORY STATISTICS (6 Categories)")
print(f"{'='*80}")
print(f"{'Category':<20} {'Valid':<12} {'Blocked':<12} {'Agree':<12} {'Disagree':<12} {'Agreement %':<15}")
print("-"*90)

for category_data in high_level_summary:
    print(f"{category_data['category']:<20} "
          f"{category_data['valid_samples']:<12} "
          f"{category_data['blocked_samples']:<12} "
          f"{category_data['agreement']:<12} "
          f"{category_data['disagreement']:<12} "
          f"{category_data['agreement_percentage']:.2f}%")

print(f"\n{'='*80}")
print("✅ CLEANED EVALUATION COMPLETE!")
print(f"{'='*80}")
