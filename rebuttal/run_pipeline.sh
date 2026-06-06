#!/bin/bash
# Master pipeline: Extract → Translate → Infer → Report
# Run from the Reabuttle directory.
set -e

DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR"

echo "══════════════════════════════════════════════"
echo "  Step 1 – Extract agreed MC samples"
echo "══════════════════════════════════════════════"
python3 step1_extract_agreed.py

echo ""
echo "══════════════════════════════════════════════"
echo "  Step 2 – Translate to Spanish (Gemini Flash)"
echo "══════════════════════════════════════════════"
python3 step2_translate_spanish.py

echo ""
echo "══════════════════════════════════════════════"
echo "  Step 3 – Run Qwen2.5-VL-7B-Instruct"
echo "══════════════════════════════════════════════"
python3 step3_run_qwen.py

echo ""
echo "══════════════════════════════════════════════"
echo "  Step 4 – Compute LBS & RAE metrics"
echo "══════════════════════════════════════════════"
python3 step4_compute_metrics.py

echo ""
echo "✓ Pipeline complete!"
