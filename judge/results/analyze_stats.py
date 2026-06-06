#!/usr/bin/env python3
"""
Filtering & Quality Analysis Report
====================================
Generates:
  • csv/report_summary.csv          – grand totals per source
  • csv/report_bloom_l1.csv         – L1 Bloom category × source
  • csv/report_l2.csv               – L2 sub-category × source
  • csv/report_l3.csv               – L3 sub-sub-category × source
  • csv/report_l4.csv               – L4 leaf path × source
  • csv/report_cleaned_dist.csv     – cleaned dataset distribution (all levels)
  • Filtering_Quality_Analysis_Report.txt

Taxonomy normalisation:
  'Apply - Conceptual Understanding ...' (broken separator) → treated as Apply

Usage: python3 analyze_stats.py
"""

import json, os, re, csv, io
from collections import Counter, defaultdict
from datetime import datetime

# ──────────────────────────────────────────────────────────────────────────────
SCRIPTS = '/fanar-image-understanding-01/OryxTrain/OryxSegments/fanar3/Omid/Bloom-bench-acl/scripts'
RESULTS = '/fanar-image-understanding-01/OryxTrain/OryxSegments/fanar3/Omid/Bloom-bench-acl/judge/results'
CSV_DIR = os.path.join(RESULTS, 'csv')
os.makedirs(CSV_DIR, exist_ok=True)

SOURCES = {
    'Qwen3-VL-235B':     f'{SCRIPTS}/QA_Qwen3-VL-235B-A22B-Instruct.json',
    'Qwen2.5-VL-72B':    f'{SCRIPTS}/QA_Qwen2.5-VL-72B-Instruct.json',
    'Gemini-3-Flash':    f'{SCRIPTS}/QA_Gemini-3-Flash_Dataset.json',
    'Gemini-3-Flash-10': f'{SCRIPTS}/QA_Gemini-3-Flash_Dataset-10.json',
}

SEP   = '=' * 90
DSEP  = '-' * 90
DATE  = datetime.now().strftime('%Y-%m-%d %H:%M')

# ──────────────────────────────────────────────────────────────────────────────
# Taxonomy helpers
# ──────────────────────────────────────────────────────────────────────────────
def parse_bloom_path(raw: str) -> list:
    """
    Split a raw lvl1 string like 'Remember -_ Recognition -_ Object Recognition'
    into ['Remember', 'Recognition', 'Object Recognition'].
    Also normalises 'Apply - Conceptual Understanding ...' (broken separator
    using ' - ' instead of ' -_ ') so it is parsed correctly.
    """
    parts = re.split(r' -_? ', raw.strip())
    parts = [p.strip().strip('_').strip('-').strip() for p in parts if p.strip()]
    return parts if parts else ['Unknown']


def levels_from_item(item: dict) -> tuple:
    """Return (l1, l2, l3, l4, leaf) for one dataset item."""
    h    = item.get('hierarchy', {})
    raw  = h.get('lvl1', 'Unknown')
    leaf = h.get('leaf', '')
    parts = parse_bloom_path(raw)
    l1 = parts[0] if len(parts) > 0 else 'Unknown'
    l2 = parts[1] if len(parts) > 1 else ''
    l3 = parts[2] if len(parts) > 2 else ''
    l4 = parts[3] if len(parts) > 3 else ''
    return l1, l2, l3, l4, leaf


# ──────────────────────────────────────────────────────────────────────────────
# CSV helpers
# ──────────────────────────────────────────────────────────────────────────────
def write_csv(path: str, header: list, rows: list):
    with open(path, 'w', newline='', encoding='utf-8') as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)
    print(f'  ✓  Saved {os.path.relpath(path, RESULTS)}  ({len(rows)} data rows)')


def pct(num, den):
    return round(num / den * 100, 2) if den else 0.0


# ──────────────────────────────────────────────────────────────────────────────
# Text-report buffer
# ──────────────────────────────────────────────────────────────────────────────
buf = io.StringIO()

def P(*args, **kw):
    print(*args, **kw, file=buf)
    print(*args, **kw)


# ──────────────────────────────────────────────────────────────────────────────
# 1.  Load cleaned dataset
# ──────────────────────────────────────────────────────────────────────────────
print('Loading cleaned dataset …')
with open(f'{RESULTS}/cleaned_VQA_dataset.json') as f:
    cleaned = json.load(f)

cleaned_ids = {item['question_id'] for item in cleaned}
total_c     = len(cleaned)

# ──────────────────────────────────────────────────────────────────────────────
# 2.  Load all source files
# ──────────────────────────────────────────────────────────────────────────────
print('Loading source files …')
source_data = {}
for name, path in SOURCES.items():
    with open(path) as f:
        source_data[name] = json.load(f)
    print(f'  {name}: {len(source_data[name]):,} items')

all_source_ids = set()
id_sources     = defaultdict(list)
for name, data in source_data.items():
    for item in data:
        qid = item['question_id']
        all_source_ids.add(qid)
        id_sources[qid].append(name)

grand_total    = sum(len(v) for v in source_data.values())
grand_kept     = sum(1 for qid in all_source_ids if qid in cleaned_ids)
grand_filtered = grand_total - grand_kept


# ──────────────────────────────────────────────────────────────────────────────
# 3.  Per-source taxonomy counters  (kept vs filtered per level)
# ──────────────────────────────────────────────────────────────────────────────
print('Building taxonomy counters …')

src_stats = {n: defaultdict(lambda: {'kept': 0, 'filtered': 0})
             for n in SOURCES}
src_totals = {n: {'total': 0, 'kept': 0} for n in SOURCES}

for name, data in source_data.items():
    for item in data:
        qid    = item['question_id']
        l1,l2,l3,l4,lf = levels_from_item(item)
        status = 'kept' if qid in cleaned_ids else 'filtered'
        src_totals[name]['total'] += 1
        if status == 'kept':
            src_totals[name]['kept'] += 1

        for key in [
            ('l1', l1),
            ('l2', l1, l2) if l2 else None,
            ('l3', l1, l2, l3) if l3 else None,
            ('l4', l1, l2, l3, l4) if l4 else None,
        ]:
            if key:
                src_stats[name][key][status] += 1

# Cleaned dataset distribution
clean_dist = defaultdict(int)
for item in cleaned:
    l1,l2,l3,l4,lf = levels_from_item(item)
    clean_dist[(l1,l2,l3,l4,lf)] += 1


# ──────────────────────────────────────────────────────────────────────────────
# 4.  Aggregate across all sources
# ──────────────────────────────────────────────────────────────────────────────
ALL = 'ALL_SOURCES'
agg_stats = defaultdict(lambda: {'kept': 0, 'filtered': 0})
for name in SOURCES:
    for key, cnt in src_stats[name].items():
        agg_stats[key]['kept']     += cnt['kept']
        agg_stats[key]['filtered'] += cnt['filtered']

all_l1 = sorted({k[1]         for k in agg_stats if k[0]=='l1'})
all_l2 = sorted({k[1:3]       for k in agg_stats if k[0]=='l2'})
all_l3 = sorted({k[1:4]       for k in agg_stats if k[0]=='l3'})
all_l4 = sorted({k[1:5]       for k in agg_stats if k[0]=='l4'})


# ──────────────────────────────────────────────────────────────────────────────
# 5.  Build & write CSV files
# ──────────────────────────────────────────────────────────────────────────────

# ── 5a. Summary ───────────────────────────────────────────────────────────────
summary_rows = []
for name in SOURCES:
    t = src_totals[name]['total']
    k = src_totals[name]['kept']
    fi = t - k
    summary_rows.append([name, t, k, fi, pct(k,t), pct(fi,t)])
summary_rows.append([ALL, grand_total, grand_kept, grand_total - grand_kept,
                     pct(grand_kept, grand_total), pct(grand_total - grand_kept, grand_total)])
write_csv(f'{CSV_DIR}/report_summary.csv',
    ['source', 'total_generated', 'kept', 'filtered', 'kept_pct', 'filtered_pct'],
    summary_rows)

# ── 5b. L1 Bloom ──────────────────────────────────────────────────────────────
l1_rows = []
for name in list(SOURCES.keys()) + [ALL]:
    stats = src_stats[name] if name != ALL else agg_stats
    for l1 in all_l1:
        key = ('l1', l1)
        k  = stats[key]['kept']
        fi = stats[key]['filtered']
        t  = k + fi
        if t > 0:
            l1_rows.append([name, l1, t, k, fi, pct(k,t), pct(fi,t)])
write_csv(f'{CSV_DIR}/report_bloom_l1.csv',
    ['source', 'bloom_l1', 'generated', 'kept', 'filtered', 'kept_pct', 'filtered_pct'],
    l1_rows)

# ── 5c. L2 ────────────────────────────────────────────────────────────────────
l2_rows = []
for name in list(SOURCES.keys()) + [ALL]:
    stats = src_stats[name] if name != ALL else agg_stats
    for (l1, l2) in all_l2:
        key = ('l2', l1, l2)
        k  = stats[key]['kept']
        fi = stats[key]['filtered']
        t  = k + fi
        if t > 0:
            l2_rows.append([name, l1, l2, t, k, fi, pct(k,t), pct(fi,t)])
write_csv(f'{CSV_DIR}/report_l2.csv',
    ['source', 'bloom_l1', 'l2_subcategory', 'generated', 'kept', 'filtered', 'kept_pct', 'filtered_pct'],
    l2_rows)

# ── 5d. L3 ────────────────────────────────────────────────────────────────────
l3_rows = []
for name in list(SOURCES.keys()) + [ALL]:
    stats = src_stats[name] if name != ALL else agg_stats
    for (l1,l2,l3) in all_l3:
        key = ('l3', l1, l2, l3)
        k  = stats[key]['kept']
        fi = stats[key]['filtered']
        t  = k + fi
        if t > 0:
            l3_rows.append([name, l1, l2, l3, t, k, fi, pct(k,t), pct(fi,t)])
write_csv(f'{CSV_DIR}/report_l3.csv',
    ['source', 'bloom_l1', 'l2_subcategory', 'l3_subcategory', 'generated', 'kept', 'filtered', 'kept_pct', 'filtered_pct'],
    l3_rows)

# ── 5e. L4 ────────────────────────────────────────────────────────────────────
l4_rows = []
for name in list(SOURCES.keys()) + [ALL]:
    stats = src_stats[name] if name != ALL else agg_stats
    for (l1,l2,l3,l4) in all_l4:
        key = ('l4', l1, l2, l3, l4)
        k  = stats[key]['kept']
        fi = stats[key]['filtered']
        t  = k + fi
        if t > 0:
            l4_rows.append([name, l1, l2, l3, l4, t, k, fi, pct(k,t), pct(fi,t)])
write_csv(f'{CSV_DIR}/report_l4.csv',
    ['source', 'bloom_l1', 'l2_subcategory', 'l3_subcategory', 'l4_subcategory', 'generated', 'kept', 'filtered', 'kept_pct', 'filtered_pct'],
    l4_rows)

# ── 5f. Cleaned distribution (all levels) ────────────────────────────────────
dist_rows = []
for (l1,l2,l3,l4,lf), cnt in sorted(clean_dist.items(), key=lambda x: (-x[1], x[0])):
    dist_rows.append([l1, l2, l3, l4, lf, cnt, pct(cnt, total_c)])
write_csv(f'{CSV_DIR}/report_cleaned_dist.csv',
    ['bloom_l1', 'l2_subcategory', 'l3_subcategory', 'l4_subcategory', 'leaf', 'count', 'pct_of_cleaned'],
    dist_rows)


# ──────────────────────────────────────────────────────────────────────────────
# 6.  Text report
# ──────────────────────────────────────────────────────────────────────────────
def fmt_table(headers, rows, col_widths=None):
    """Returns aligned plain-text table."""
    if not rows:
        return '  (no data)\n'
    all_rows = [headers] + rows
    if col_widths is None:
        col_widths = [max(len(str(r[i])) for r in all_rows)
                      for i in range(len(headers))]
    fmt = '  ' + '  '.join(
        f'{{:<{w}}}' if i < len(headers) - 1 else f'{{:>{w}}}'
        for i, w in enumerate(col_widths))
    lines = [fmt.format(*[str(h) for h in headers]),
             '  ' + '-' * (sum(col_widths) + 2 * (len(col_widths) - 1))]
    for row in rows:
        lines.append(fmt.format(*[str(c) for c in row]))
    return '\n'.join(lines) + '\n'


P()
P(SEP)
P(f'  FILTERING & QUALITY ANALYSIS REPORT')
P(f'  Generated : {DATE}')
P(f'  Dataset   : {RESULTS}/cleaned_VQA_dataset.json')
P(f'  Total QA  : {total_c:,} items')
P(SEP)

# ── Section 1: Grand Summary ──────────────────────────────────────────────────
P()
P(SEP)
P('  SECTION 1 · GRAND SUMMARY — ALL SOURCES')
P(SEP)
hdr = ['Source', 'Generated', 'Kept', 'Filtered', 'Kept %', 'Filtered %']
rows = []
for name in SOURCES:
    t  = src_totals[name]['total']
    k  = src_totals[name]['kept']
    fi = t - k
    rows.append([name, f'{t:,}', f'{k:,}', f'{fi:,}', f'{pct(k,t):.1f}%', f'{pct(fi,t):.1f}%'])
rows.append([ALL, f'{grand_total:,}', f'{grand_kept:,}',
             f'{grand_total-grand_kept:,}',
             f'{pct(grand_kept,grand_total):.1f}%',
             f'{pct(grand_total-grand_kept,grand_total):.1f}%'])
P(fmt_table(hdr, rows, col_widths=[20, 11, 9, 10, 8, 11]))

cross_dups        = sum(1 for srcs in id_sources.values() if len(srcs) > 1)
unique_in_sources = len(all_source_ids)
P(f'  Unique question_ids across all 4 sources : {unique_in_sources:,}')
P(f'  Cross-source duplicate question_ids      : {cross_dups:,}')
P(f'  IDs in cleaned missing from sources      : {len(cleaned_ids - all_source_ids):,}')

# ── Section 2: Bloom L1 Distribution (Cleaned) ───────────────────────────────
P()
P(SEP)
P('  SECTION 2 · BLOOM TAXONOMY L1 — CLEANED DATASET DISTRIBUTION')
P(SEP)
l1_clean = Counter()
for item in cleaned:
    l1,*_ = levels_from_item(item)
    l1_clean[l1] += 1
hdr  = ['Bloom L1 Category', 'Count', '% of Cleaned']
rows = [[cat, f'{n:,}', f'{pct(n,total_c):.1f}%']
        for cat, n in sorted(l1_clean.items(), key=lambda x: -x[1])]
P(fmt_table(hdr, rows, col_widths=[40, 9, 14]))

# ── Section 3: L1 Filtering per Source ───────────────────────────────────────
P()
P(SEP)
P('  SECTION 3 · BLOOM L1 — FILTERING RATE PER SOURCE')
P(SEP)
for l1 in all_l1:
    P(f'\n  ── {l1}')
    hdr  = ['Source', 'Generated', 'Kept', 'Filtered', 'Kept %', 'Filtered %']
    rows = []
    for name in SOURCES:
        key = ('l1', l1)
        k  = src_stats[name][key]['kept']
        fi = src_stats[name][key]['filtered']
        t  = k + fi
        if t > 0:
            rows.append([name, f'{t:,}', f'{k:,}', f'{fi:,}',
                         f'{pct(k,t):.1f}%', f'{pct(fi,t):.1f}%'])
    k  = agg_stats[('l1',l1)]['kept']
    fi = agg_stats[('l1',l1)]['filtered']
    t  = k + fi
    rows.append([ALL, f'{t:,}', f'{k:,}', f'{fi:,}',
                 f'{pct(k,t):.1f}%', f'{pct(fi,t):.1f}%'])
    P(fmt_table(hdr, rows, col_widths=[20, 11, 9, 10, 8, 11]))

# ── Section 4: L2 Combined ───────────────────────────────────────────────────
P()
P(SEP)
P('  SECTION 4 · TAXONOMY L2 — SUB-CATEGORY FILTERING  (ALL SOURCES COMBINED)')
P(SEP)
l2_by_l1 = defaultdict(list)
for (l1,l2) in all_l2:
    l2_by_l1[l1].append(l2)

for l1 in sorted(l2_by_l1):
    P(f'\n  ── {l1}')
    hdr  = ['L2 Sub-Category', 'Generated', 'Kept', 'Filtered', 'Kept %', 'Filtered %']
    rows = []
    for l2 in sorted(l2_by_l1[l1],
                     key=lambda x: -(agg_stats[('l2',l1,x)]['kept']
                                    +agg_stats[('l2',l1,x)]['filtered'])):
        k  = agg_stats[('l2',l1,l2)]['kept']
        fi = agg_stats[('l2',l1,l2)]['filtered']
        t  = k + fi
        rows.append([l2, f'{t:,}', f'{k:,}', f'{fi:,}',
                     f'{pct(k,t):.1f}%', f'{pct(fi,t):.1f}%'])
    P(fmt_table(hdr, rows, col_widths=[55, 11, 9, 10, 8, 11]))

# ── Section 5: L3 Combined ───────────────────────────────────────────────────
P()
P(SEP)
P('  SECTION 5 · TAXONOMY L3 — SUB-SUB-CATEGORY FILTERING  (ALL SOURCES COMBINED)')
P(SEP)
if all_l3:
    l3_by_l1 = defaultdict(list)
    for (l1,l2,l3) in all_l3:
        l3_by_l1[l1].append((l2,l3))
    for l1 in sorted(l3_by_l1):
        P(f'\n  ── {l1}')
        hdr  = ['L2 Sub-Category', 'L3 Sub-Category', 'Generated', 'Kept', 'Filtered', 'Kept %', 'Filtered %']
        rows = []
        for (l2,l3) in sorted(l3_by_l1[l1],
                               key=lambda x: -(agg_stats[('l3',l1,x[0],x[1])]['kept']
                                              +agg_stats[('l3',l1,x[0],x[1])]['filtered'])):
            k  = agg_stats[('l3',l1,l2,l3)]['kept']
            fi = agg_stats[('l3',l1,l2,l3)]['filtered']
            t  = k + fi
            rows.append([l2, l3, f'{t:,}', f'{k:,}', f'{fi:,}',
                         f'{pct(k,t):.1f}%', f'{pct(fi,t):.1f}%'])
        P(fmt_table(hdr, rows, col_widths=[45, 35, 11, 9, 10, 8, 11]))
else:
    P('\n  (No L3 sub-sub-categories found)\n')

# ── Section 6: L4 ─────────────────────────────────────────────────────────────
P()
P(SEP)
P('  SECTION 6 · TAXONOMY L4 — DEEPEST PATH FILTERING  (ALL SOURCES COMBINED)')
P(SEP)
if all_l4:
    hdr  = ['Full Taxonomy Path (L1→L2→L3→L4)', 'Generated', 'Kept', 'Filtered', 'Kept %', 'Filtered %']
    rows = []
    for tup in sorted(all_l4,
                      key=lambda x: -(agg_stats[('l4',)+x]['kept']
                                     +agg_stats[('l4',)+x]['filtered'])):
        l1,l2,l3,l4 = tup
        k  = agg_stats[('l4',l1,l2,l3,l4)]['kept']
        fi = agg_stats[('l4',l1,l2,l3,l4)]['filtered']
        t  = k + fi
        path = ' → '.join(filter(None, [l1,l2,l3,l4]))
        rows.append([path, f'{t:,}', f'{k:,}', f'{fi:,}',
                     f'{pct(k,t):.1f}%', f'{pct(fi,t):.1f}%'])
    P(fmt_table(hdr, rows, col_widths=[65, 11, 9, 10, 8, 11]))
else:
    P('\n  (No L4 entries found)\n')

# ── Section 7: Cleaned dataset — full tree ───────────────────────────────────
P()
P(SEP)
P('  SECTION 7 · CLEANED DATASET — FULL TAXONOMY TREE')
P(SEP)

tree = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(int))))
for item in cleaned:
    l1,l2,l3,l4,lf = levels_from_item(item)
    tree[l1][l2 or '(top)'][l3 or '(top)'][l4 or lf or '(leaf)'] += 1

for l1 in sorted(tree, key=lambda x: -sum(
        sum(sum(v4.values()) for v4 in v3.values())
        for v3 in tree[x].values())):
    l1_tot = sum(sum(sum(v4.values()) for v4 in v3.values()) for v3 in tree[l1].values())
    P(f'\n  ┌─ {l1}  [{l1_tot:,} items  {pct(l1_tot,total_c):.1f}%]')
    for l2 in sorted(tree[l1],
                     key=lambda x: -sum(sum(v4.values()) for v4 in tree[l1][x].values())):
        l2_tot = sum(sum(v4.values()) for v4 in tree[l1][l2].values())
        if l2 == '(top)':
            P(f'  │  └─ (no sub-category)  [{l2_tot:,}]')
            continue
        P(f'  │  ├─ {l2}  [{l2_tot:,}]')
        for l3 in sorted(tree[l1][l2],
                         key=lambda x: -sum(tree[l1][l2][x].values())):
            l3_tot = sum(tree[l1][l2][l3].values())
            if l3 == '(top)':
                P(f'  │  │  └─ (no sub-sub-category)  [{l3_tot:,}]')
                continue
            P(f'  │  │  ├─ {l3}  [{l3_tot:,}]')
            for l4, cnt in sorted(tree[l1][l2][l3].items(), key=lambda x: -x[1]):
                if l4 in ('(leaf)',) or cnt == 0:
                    continue
                P(f'  │  │  │  └─ {l4}  [{cnt:,}]')

# ── Section 8: Per-source L2 breakdown ───────────────────────────────────────
P()
P(SEP)
P('  SECTION 8 · TAXONOMY L2 — PER-SOURCE BREAKDOWN')
P(SEP)
for name in SOURCES:
    P(f'\n  ═══ {name} ═══')
    hdr  = ['L1', 'L2 Sub-Category', 'Generated', 'Kept', 'Filtered', 'Kept %', 'Filtered %']
    rows = []
    for (l1,l2) in sorted(all_l2, key=lambda x: x[0]):
        k  = src_stats[name][('l2',l1,l2)]['kept']
        fi = src_stats[name][('l2',l1,l2)]['filtered']
        t  = k + fi
        if t > 0:
            rows.append([l1, l2, f'{t:,}', f'{k:,}', f'{fi:,}',
                         f'{pct(k,t):.1f}%', f'{pct(fi,t):.1f}%'])
    P(fmt_table(hdr, rows, col_widths=[14, 55, 11, 9, 10, 8, 11]))

P()
P(SEP)
P(f'  END OF REPORT  |  {DATE}')
P(SEP)
P()

# ── Write TXT report ─────────────────────────────────────────────────────────
report_path = os.path.join(RESULTS, 'Filtering_Quality_Analysis_Report.txt')
with open(report_path, 'w', encoding='utf-8') as f:
    f.write(buf.getvalue())
print(f'\n  ✓  Saved Filtering_Quality_Analysis_Report.txt')
print('\nAll outputs written successfully.')
