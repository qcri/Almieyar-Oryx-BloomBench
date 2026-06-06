#!/usr/bin/env python3
"""
Step 4: Compute LBS (Likelihood-Based Score) and RAE (Regex-Accuracy Extraction)
        – micro and macro – for each language.

Micro = flat accuracy over all samples.
Macro = mean of per-leaf-category accuracies (unweighted by category size).
"""

import json, os
import pandas as pd
from collections import defaultdict

BASE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE, "results_rebuttal")
REPORT_PATH = os.path.join(RESULTS_DIR, "metrics_report.txt")
REPORT_CSV = os.path.join(RESULTS_DIR, "metrics_report.csv")

LANGS = ["en", "ar", "es"]
LANG_NAMES = {"en": "English", "ar": "Arabic", "es": "Spanish"}


def compute_metrics(df):
    """Return dict with micro/macro for RAE and LBS."""
    df = df[df["gold"] != "ERROR"].copy()
    # Handle potentially missing leaf names
    df["hierarchy_leaf"] = df["hierarchy_leaf"].fillna("Unknown")
    n = len(df)
    if n == 0:
        return {"n": 0}

    # ── Micro ─────────────────────────────────────────────────────────
    rae_micro = (df["regex_answer"] == df["gold"]).mean() * 100
    lbs_micro = (df["likelihood_answer"] == df["gold"]).mean() * 100

    # ── Macro (per leaf category) ─────────────────────────────────────
    rae_by_cat = defaultdict(list)
    lbs_by_cat = defaultdict(list)
    for _, row in df.iterrows():
        cat = row["hierarchy_leaf"]
        rae_by_cat[cat].append(int(row["regex_answer"] == row["gold"]))
        lbs_by_cat[cat].append(int(row["likelihood_answer"] == row["gold"]))

    rae_macro = 0.0
    lbs_macro = 0.0
    num_cats = len(rae_by_cat)
    for cat in rae_by_cat:
        rae_macro += sum(rae_by_cat[cat]) / len(rae_by_cat[cat])
        lbs_macro += sum(lbs_by_cat[cat]) / len(lbs_by_cat[cat])
    rae_macro = (rae_macro / num_cats) * 100 if num_cats else 0
    lbs_macro = (lbs_macro / num_cats) * 100 if num_cats else 0

    return {
        "n": n,
        "num_categories": num_cats,
        "RAE_micro": round(rae_micro, 2),
        "RAE_macro": round(rae_macro, 2),
        "LBS_micro": round(lbs_micro, 2),
        "LBS_macro": round(lbs_macro, 2),
    }


def main():
    lines = []
    all_rows = []

    header = (
        f"{'Language':<12} {'N':>5} {'#Cats':>6}  "
        f"{'RAE_micro':>10} {'RAE_macro':>10}  "
        f"{'LBS_micro':>10} {'LBS_macro':>10}"
    )
    sep = "─" * len(header)
    lines.append("=" * len(header))
    lines.append("  Qwen2.5-VL-7B-Instruct  –  Rebuttal Evaluation (Agreed Samples)")
    lines.append("=" * len(header))
    lines.append("")
    lines.append(header)
    lines.append(sep)

    for lang in LANGS:
        csv_path = os.path.join(RESULTS_DIR, f"{lang}_results.csv")
        if not os.path.exists(csv_path):
            print(f"⚠ Missing {csv_path}, skipping {lang}")
            continue

        df = pd.read_csv(csv_path, dtype=str)
        m = compute_metrics(df)

        row_str = (
            f"{LANG_NAMES[lang]:<12} {m['n']:>5} {m.get('num_categories',''):>6}  "
            f"{m.get('RAE_micro','—'):>10} {m.get('RAE_macro','—'):>10}  "
            f"{m.get('LBS_micro','—'):>10} {m.get('LBS_macro','—'):>10}"
        )
        lines.append(row_str)
        all_rows.append({"Language": LANG_NAMES[lang], **m})

    lines.append(sep)
    lines.append("")

    # ── Per-category breakdown (optional detail) ──────────────────────
    for lang in LANGS:
        csv_path = os.path.join(RESULTS_DIR, f"{lang}_results.csv")
        if not os.path.exists(csv_path):
            continue
        df = pd.read_csv(csv_path)
        df = df[df["gold"] != "ERROR"]
        df["hierarchy_leaf"] = df["hierarchy_leaf"].fillna("Unknown")
        if df.empty:
            continue

        lines.append(f"\n── {LANG_NAMES[lang]} per-category breakdown ──")
        lines.append(f"{'Category':<45} {'N':>4}  {'RAE%':>6}  {'LBS%':>6}")
        lines.append("─" * 70)

        cats = sorted(df["hierarchy_leaf"].unique())
        for cat in cats:
            sub = df[df["hierarchy_leaf"] == cat]
            n = len(sub)
            rae = (sub["regex_answer"] == sub["gold"]).mean() * 100
            lbs = (sub["likelihood_answer"] == sub["gold"]).mean() * 100
            lines.append(f"{cat:<45} {n:>4}  {rae:>6.1f}  {lbs:>6.1f}")
        lines.append("")

    report = "\n".join(lines)
    print(report)

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\nSaved report → {REPORT_PATH}")

    if all_rows:
        pd.DataFrame(all_rows).to_csv(REPORT_CSV, index=False)
        print(f"Saved CSV   → {REPORT_CSV}")


if __name__ == "__main__":
    main()
