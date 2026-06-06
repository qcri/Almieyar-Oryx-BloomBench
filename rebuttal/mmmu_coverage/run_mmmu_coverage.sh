#!/bin/bash
# Run the full MMMU coverage analysis pipeline
# Usage: bash run_mmmu_coverage.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "=============================================="
echo "  MMMU → Bloom-Bench Coverage Analysis"
echo "=============================================="

echo ""
echo "Step 1: Download & sample ~1000 MMMU questions …"
python3 step1_sample_mmmu.py

echo ""
echo "Step 2: Classify MMMU samples using Gemini Flash …"
python3 step2_classify_with_gemini.py

echo ""
echo "Step 3: Analyze coverage …"
python3 step3_analyze_coverage.py

echo ""
echo "✓ Pipeline complete.  See results in:"
echo "    $SCRIPT_DIR/mmmu_coverage_report.txt"
echo "    $SCRIPT_DIR/mmmu_coverage_report.csv"
echo "    $SCRIPT_DIR/mmmu_coverage_summary.json"
