#!/usr/bin/env python3
"""
Step 3: Analyze MMMU → Bloom-Bench taxonomy coverage.

Reads the classified MMMU samples and reports:
  1. How many of our 106 leaf categories are covered by MMMU
  2. Which leaves are covered and which are NOT
  3. Distribution of MMMU samples across our taxonomy
  4. Coverage by Bloom-Bench high-level category (Remember, Understand, Analyze, Apply, Create, Evaluate)
  5. "NONE" rate — questions that don't fit our taxonomy at all

Outputs:
  - mmmu_coverage_report.txt   (human-readable)
  - mmmu_coverage_report.csv   (machine-readable)
  - mmmu_coverage_summary.json (for downstream use)
"""

import json, os, csv
from collections import defaultdict, Counter
from pathlib import Path

BASE = os.path.dirname(os.path.abspath(__file__))
REABUTTLE = os.path.dirname(BASE)

CLASSIFIED_JSON = os.path.join(BASE, "mmmu_classified.json")
EVAL_JSON       = os.path.join(REABUTTLE, "gemini3_oryx_evaluation_CLEANED.json")

REPORT_TXT = os.path.join(BASE, "mmmu_coverage_report.txt")
REPORT_CSV = os.path.join(BASE, "mmmu_coverage_report.csv")
SUMMARY_JSON = os.path.join(BASE, "mmmu_coverage_summary.json")


def load_taxonomy():
    with open(EVAL_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)
    leaves = []
    for level_key, level_data in data["results_by_level"].items():
        leaves.append({
            "full_path": level_data["level"],
            "lvl1": level_data["lvl1"],
            "leaf": level_data["leaf"],
        })
    return leaves


def normalise(s):
    """Lower-case, strip, collapse whitespace."""
    return " ".join(s.strip().lower().split())


def main():
    taxonomy = load_taxonomy()
    leaf_names = [t["leaf"] for t in taxonomy]
    leaf_norm_map = {}   # normalised leaf → original leaf
    for t in taxonomy:
        leaf_norm_map[normalise(t["leaf"])] = t["leaf"]

    # Build leaf → high-level Bloom category
    leaf_to_bloom = {}
    leaf_to_full_path = {}
    for t in taxonomy:
        top = t["full_path"].split(" -> ")[0].strip()
        leaf_to_bloom[t["leaf"]] = top
        leaf_to_full_path[t["leaf"]] = t["full_path"]

    # Load classified MMMU results
    with open(CLASSIFIED_JSON, "r", encoding="utf-8") as f:
        results = json.load(f)

    total = len(results)
    print(f"Loaded {total} classified MMMU samples")

    # ── Normalize predictions ─────────────────────────────────────────
    none_count = 0
    error_count = 0
    matched_count = 0
    unmatched_preds = Counter()
    leaf_hit_count = Counter()      # leaf → number of MMMU items mapped
    bloom_hit_count = Counter()     # bloom level → number of items

    for r in results:
        pred_num = r.get("predicted_num", None)
        pred_raw = r.get("predicted_leaf", "")
        pred_norm = normalise(pred_raw)

        # Fast path: if predicted_num is a valid 1-106, use it directly
        if pred_num is not None and isinstance(pred_num, int) and 1 <= pred_num <= len(taxonomy):
            matched_leaf = taxonomy[pred_num - 1]["leaf"]
            leaf_hit_count[matched_leaf] += 1
            bloom_hit_count[leaf_to_bloom.get(matched_leaf, "?")] += 1
            matched_count += 1
            r["_matched_leaf"] = matched_leaf
            continue
        if pred_num == 0 or pred_norm in ("none", ""):
            none_count += 1
            r["_matched_leaf"] = "NONE"
            continue
        if pred_norm in ("error", "image_missing", "blocked") or (pred_num is not None and pred_num < 0):
            error_count += 1
            r["_matched_leaf"] = "ERROR"
            continue

        # Fallback: text-based matching for old-format data
        # Try exact normalised match
        if pred_norm in leaf_norm_map:
            matched_leaf = leaf_norm_map[pred_norm]
            leaf_hit_count[matched_leaf] += 1
            bloom_hit_count[leaf_to_bloom.get(matched_leaf, "?")] += 1
            matched_count += 1
            r["_matched_leaf"] = matched_leaf
        else:
            # Fuzzy: check if prediction is substring of any leaf or vice versa
            found = False
            for ln, orig in leaf_norm_map.items():
                if pred_norm in ln or ln in pred_norm:
                    leaf_hit_count[orig] += 1
                    bloom_hit_count[leaf_to_bloom.get(orig, "?")] += 1
                    matched_count += 1
                    r["_matched_leaf"] = orig
                    found = True
                    break
            if not found:
                unmatched_preds[pred_raw] += 1
                none_count += 1
                r["_matched_leaf"] = "NONE"

    # ── Coverage stats ────────────────────────────────────────────────
    covered_leaves  = set(leaf_hit_count.keys())
    all_leaves      = set(leaf_names)
    uncovered_leaves = all_leaves - covered_leaves

    coverage_pct = len(covered_leaves) / len(all_leaves) * 100 if all_leaves else 0

    # ── Per-Bloom-level breakdown ─────────────────────────────────────
    bloom_levels = ["Remember", "Understand", "Analyze", "Apply", "Create", "Evaluate"]
    bloom_leaf_total = Counter()
    bloom_leaf_covered = Counter()
    for t in taxonomy:
        top = t["full_path"].split(" -> ")[0].strip()
        bloom_leaf_total[top] += 1
        if t["leaf"] in covered_leaves:
            bloom_leaf_covered[top] += 1

    # ── Build report ──────────────────────────────────────────────────
    lines = []
    lines.append("=" * 70)
    lines.append("  MMMU → Bloom-Bench Taxonomy Coverage Report")
    lines.append("=" * 70)
    lines.append("")
    lines.append(f"Total MMMU samples classified:  {total}")
    lines.append(f"Matched to a leaf category:     {matched_count}  ({matched_count/total*100:.1f}%)")
    lines.append(f"Classified as NONE (no match):  {none_count}  ({none_count/total*100:.1f}%)")
    lines.append(f"Errors / missing images:        {error_count}")
    lines.append("")
    lines.append(f"Our taxonomy total leaves:      {len(all_leaves)}")
    lines.append(f"Leaves covered by MMMU:         {len(covered_leaves)}  ({coverage_pct:.1f}%)")
    lines.append(f"Leaves NOT covered by MMMU:     {len(uncovered_leaves)}  ({100-coverage_pct:.1f}%)")
    lines.append("")

    lines.append("-" * 70)
    lines.append("  Coverage by Bloom's Taxonomy Level")
    lines.append("-" * 70)
    for bl in bloom_levels:
        tot = bloom_leaf_total.get(bl, 0)
        cov = bloom_leaf_covered.get(bl, 0)
        hits = bloom_hit_count.get(bl, 0)
        pct = cov / tot * 100 if tot else 0
        lines.append(f"  {bl:<15}  leaves: {cov:>3}/{tot:>3} covered ({pct:5.1f}%)   |  MMMU samples mapped: {hits}")
    lines.append("")

    lines.append("-" * 70)
    lines.append("  COVERED leaves (sorted by # MMMU samples mapped)")
    lines.append("-" * 70)
    for leaf, cnt in sorted(leaf_hit_count.items(), key=lambda x: -x[1]):
        lines.append(f"  {cnt:>4}  {leaf:<45} [{leaf_to_full_path.get(leaf, '')}]")
    lines.append("")

    lines.append("-" * 70)
    lines.append("  UNCOVERED leaves (MMMU has 0 samples for these)")
    lines.append("-" * 70)
    for leaf in sorted(uncovered_leaves):
        lines.append(f"  ✗  {leaf:<45} [{leaf_to_full_path.get(leaf, '')}]")
    lines.append("")

    if unmatched_preds:
        lines.append("-" * 70)
        lines.append("  Gemini predictions that didn't match any leaf")
        lines.append("-" * 70)
        for pred, cnt in unmatched_preds.most_common(30):
            lines.append(f"  {cnt:>4}  \"{pred}\"")
        lines.append("")

    # ── MMMU subject → leaf distribution ──────────────────────────────
    lines.append("-" * 70)
    lines.append("  MMMU Subject → mapped Bloom-Bench leaves")
    lines.append("-" * 70)
    subj_leaf = defaultdict(Counter)
    for r in results:
        ml = r.get("_matched_leaf", "NONE")
        subj_leaf[r["subject"]][ml] += 1
    for subj in sorted(subj_leaf.keys()):
        top3 = subj_leaf[subj].most_common(5)
        top_str = ", ".join(f"{l}({c})" for l, c in top3)
        lines.append(f"  {subj:<35} → {top_str}")
    lines.append("")

    report = "\n".join(lines)
    print(report)

    with open(REPORT_TXT, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\n✓ Report saved → {REPORT_TXT}")

    # ── CSV output ────────────────────────────────────────────────────
    with open(REPORT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["bloom_level", "leaf", "full_path", "mmmu_samples_mapped", "covered"])
        for t in taxonomy:
            top = t["full_path"].split(" -> ")[0].strip()
            cnt = leaf_hit_count.get(t["leaf"], 0)
            cov = "yes" if t["leaf"] in covered_leaves else "no"
            w.writerow([top, t["leaf"], t["full_path"], cnt, cov])
    print(f"✓ CSV saved → {REPORT_CSV}")

    # ── JSON summary ──────────────────────────────────────────────────
    summary = {
        "total_mmmu_samples": total,
        "matched_to_leaf": matched_count,
        "none_no_match": none_count,
        "errors": error_count,
        "total_bloom_leaves": len(all_leaves),
        "covered_leaves": len(covered_leaves),
        "uncovered_leaves": len(uncovered_leaves),
        "coverage_pct": round(coverage_pct, 2),
        "bloom_level_coverage": {
            bl: {
                "total_leaves": bloom_leaf_total.get(bl, 0),
                "covered_leaves": bloom_leaf_covered.get(bl, 0),
                "coverage_pct": round(bloom_leaf_covered.get(bl, 0) / bloom_leaf_total.get(bl, 1) * 100, 2),
                "mmmu_samples": bloom_hit_count.get(bl, 0),
            }
            for bl in bloom_levels
        },
        "covered_leaf_list": sorted(covered_leaves),
        "uncovered_leaf_list": sorted(uncovered_leaves),
        "per_leaf_counts": dict(leaf_hit_count),
    }
    with open(SUMMARY_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"✓ Summary JSON saved → {SUMMARY_JSON}")


if __name__ == "__main__":
    main()
