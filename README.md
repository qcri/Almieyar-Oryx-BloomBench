# Almieyar-Oryx-BloomBench

[![arXiv](https://img.shields.io/badge/arXiv-2606.05531-b31b1b.svg)](https://arxiv.org/abs/2606.05531) [![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-Dataset-yellow)](https://huggingface.co/datasets/QCRI/BloomBench)

**BloomBench** is a cognitively grounded, **bilingual (English–Arabic)** multimodal benchmark for evaluating **vision–language models (VLMs)**. Part of the **Almieyar** benchmarking series, it organizes tasks according to **Bloom’s revised taxonomy**—from *Remember* through *Create*—so that performance reflects **where** models succeed or fail in multimodal reasoning, not only aggregate accuracy.

This is the official repository for the paper *Almieyar-Oryx-BloomBench: A Bilingual Multimodal Benchmark for Cognitively Informed Evaluation of Vision-Language Models*, accepted to **[ACL 2026](https://2026.aclweb.org/) Findings**.

---

## Why BloomBench?

Most VLM benchmarks emphasize disconnected tasks or headline scores. BloomBench is designed to:

- **Diagnose cognitive profiles** across six Bloom levels: *Remember*, *Understand*, *Apply*, *Analyze*, *Evaluate*, and *Create*.
- **Stress-test cross-lingual multimodal reasoning** with parallel English and Arabic items, moving beyond English-centric evaluation.
- **Combine scalable construction with quality control**: a semi-automated pipeline plus **hybrid validation** (LLM-as-judge on a stratified subset, with human follow-up on flagged cases).

Empirically, the benchmark reveals a **sharp cognitive asymmetry** in current VLMs: strong ceilings on discriminative skills (e.g., facets of *Understand* / *Evaluate*) coexist with substantially weaker **factual recall** (*Remember*), **procedural application** (*Apply*), and **creative synthesis** (*Create*), especially under stricter evaluation protocols. It also exposes a **persistent Arabic–English gap**, underscoring limitations in current cross-lingual multimodal reasoning.

---

## Taxonomy at a glance

Each Bloom level is instantiated with VLM-centric task families (full leaf list in the paper appendix):

| Level | Role in BloomBench (high level) |
|--------|----------------------------------|
| **Remember** | Recognition and recall: objects, attributes, activities, symbols, text-in-image, etc. |
| **Understand** | Compositional and relational understanding; semantic, emotional, and paraphrase-style comprehension. |
| **Apply** | Using knowledge or rules in new visual contexts; basic multimodal logic (e.g., negation, structure). |
| **Analyze** | Decomposition and inference: logic/science, context, charts/tables, atypical attributes. |
| **Evaluate** | Judgment: coherence / hallucination-style checks, harm & safety, quality assessment. |
| **Create** | Discriminative creativity in MCQ form—choosing the best synthesis among options (e.g., narrative or structured constraints). |

![our-taxonomy](https://github.com/user-attachments/assets/2af3d079-86bb-452b-90b8-f34389d0b6df)

---

## Dataset (paper statistics)

As reported in the paper:

- **7,747** bilingual image–question–answer items across **106** distinct task types (taxonomy leaves), spanning all six Bloom levels.
- Per-level counts: *Remember* 2,948 · *Understand* 1,592 · *Apply* 499 · *Analyze* 1,431 · *Evaluate* 592 · *Create* 685.
- **Hybrid quality validation** on a stratified subset of **969** items (≈1/8 of the dataset, ≥4 samples per leaf): Gemini 3 Pro audited sample quality and flagged only **15** items, all confirmed as errors by human verification—an estimated **98.45%** quality rate.

Items are **multiple-choice (four options)** with professionally styled distractors (including a deliberate “trap” distractor), built from web-sourced images and scenario-guided generation, then translated into **Modern Standard Arabic** with cognitive and semantic alignment in mind.

🤗 **Dataset:** [QCRI/BloomBench](https://huggingface.co/datasets/QCRI/BloomBench)

---

## BloomBench Data Generation Pipeline

The pipeline pairs **Gemini 2.5 Pro** (scenario ideation and cognitively-grounded VQA generation) with an instruction-tuned MCQ converter and Arabic translator, validated through a **hybrid LLM-as-judge + human arbitration** stage (Gemini 3 Pro).

<img width="3556" height="900" alt="571051012-3d91f625-544d-4917-92c1-0c135ea8756d" src="https://github.com/user-attachments/assets/956c89ae-1b03-4961-8223-f1bd11536314" />

---

## Evaluation protocols

The reference evaluation uses **zero-shot** prompts and temperature **0**, and reports **accuracy** (micro and macro). Two complementary scoring modes are supported:

1. **Regex-based answer extraction (RAE)** — Parses free-form outputs for the chosen option (e.g., A–D), reflecting typical user-facing use. Invalid formats are assigned a wrong choice to account for catastrophic instruction-following failures.
2. **Likelihood-based scoring (LBS)** — Scores each choice by **length-normalized** conditional log-probability of the choice tokens given the image and question, reducing dependence on formatting and surfacing **calibration-style** behavior.

RAE and LBS can **diverge** across models (e.g., high RAE with weaker LBS), so reporting both is recommended.

### Headline results

- **Gemma 4 31B** achieves state-of-the-art RAE accuracy (**89.8%** English / **87.6%** Arabic), overtaking Qwen2.5-VL, but struggles notably under LBS.
- **Qwen2.5-VL-7B** shows the strongest internal consistency (0.869 RAE → 0.654 LBS English), while the **Gemma 3 family** exhibits an inverse-scaling trend under LBS—Gemma 3 27B posts the highest RAE (0.883) yet the steepest LBS drop (0.336).
- Arabic trails English across the board; the **Gemma 3** family shows the smallest cross-lingual drop. A controlled Spanish ablation confirms the Arabic LBS gap is a compound effect of tokenization fertility and weaker non-English probability priors.

Evaluated models: **Gemma 3** (4B / 12B / 27B), **Gemma 4** (26B-A4B / 31B), **Qwen2.5-VL-7B**, **Qwen2-VL-7B**, and **GPT-4o mini** (closed-source; RAE only, as LBS requires logit access).

---

## Repository layout

| Path | Purpose |
|------|---------|
| `scenarios/` | Scenario / taxonomy JSON files used in the construction pipeline. |
| `scripts/` | Data-generation and crawling utilities (e.g., scenario generation, QA generation, balancing). |
| `utils/` | Shared helpers (e.g., prompts, I/O). |
| `configs/` | Benchmark configurations. |
| `core/`, `adapters/`, `eval/`, `judge/` | Core runner logic, model adapters, and evaluation/judging scripts. |

Sample model configs live under `configs/` (e.g., `new_config.yaml` for BloomBench).

---

## Running the VLM benchmark runner

From the repository root directory:

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Edit a YAML config (set `hf_token`, `model_name`, `lang` to `en` or `ar`, and `eval_method` to `regex`, `likelihood`, or `hybrid`), then:

```bash
python cli.py --config configs/new_config.yaml
```

**Note:** The dataset loader expects Hugging Face assets (images archive and annotation JSON). Configure the dataset repository and filenames in `adapters/datasets/bloom.py` to match your released checkpoint, and ensure your token has access if the data are gated.

**Security:** If you extend the generation scripts, **do not commit API keys**. Use environment variables or a local secrets file that stays untracked.

---

## Citation

```bibtex
@misc{abootorabi2026almieyaroryxbloombenchbilingualmultimodalbenchmark,
      title={Almieyar-Oryx-BloomBench: A Bilingual Multimodal Benchmark for Cognitively Informed Evaluation of Vision-Language Models}, 
      author={Mohammad Mahdi Abootorabi and Omid Ghahroodi and Anas Madkoor and Marzia Nouri and Doratossadat Dastgheib and Mohamed Hefeeda and Ehsaneddin Asgari},
      year={2026},
      eprint={2606.05531},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2606.05531}, 
}
```
