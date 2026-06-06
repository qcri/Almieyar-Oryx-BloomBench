import os
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import torch

from adapters.datasets import load_dataset_by_name
from adapters.models import get_model_adapter
from utils.io import ensure_dir, encode_image
from utils.prompts import generate_prompts
from eval.regex import extract_regex


def _device(cfg):
    if torch.cuda.is_available():
        return f"cuda:{cfg.gpu_id}"
    return "cpu"


def run_experiment(cfg, limit=5000000000):
    device = cfg.device_map or _device(cfg)
    out_root = Path(cfg.cache_dir)
    ensure_dir(out_root)

    # Build model adapter (unified interface)
    model = get_model_adapter(cfg, device=device)

    # Load dataset
    data = load_dataset_by_name(cfg)

    # Output dir
    out_dir = out_root / f"{cfg.model_name.replace('/','_')}_{cfg.lang}_{cfg.dataset}_{cfg.temperature}_{cfg.exp}"
    ensure_dir(out_dir)
    out_csv = out_dir / "results.csv"

    # --- Load cache if exists ---
    if out_csv.exists():
        cache_df = pd.read_csv(out_csv)
        done_qids = set(cache_df["question_id"].tolist())
        rows = cache_df.to_dict("records")
        print(f"Loaded {len(done_qids)} cached results from {out_csv}")
    else:
        done_qids, rows = set(), []

    # Iterate samples
    for idx, sample in enumerate(tqdm(data, desc="Evaluating")):
        if idx >= limit:
            break

        # skip if already cached
        qid = sample.get("question_id")
        if qid in done_qids:
            continue
        try:
        # prompts
            pr, gold, qid, img_path = generate_prompts(sample, cfg.lang, likelihood=False, generate=False if cfg.dataset=='camel' else True)
    
            # image (as data URI if exists locally)
            # print(img_path)
            # if isinstance(img_path, str) and Path(img_path).exists():
            #     img_uri = "data:image;base64," + encode_image(img_path)
            # else:
            #     img_uri = "data:image;base64," + encode_image(sample.get("source_image_file", ""))
            img_uri = img_path
            # img_uri = img_uri.replace('/workspace/vlmbench/', '')
            # print(img_uri)
            # print(sample)
    
            # --- 1) Generation (regex) ---
            response_text, _token_probs = model.generate(pr, img_uri, lang=cfg.lang)
            if cfg.dataset!='camel':
                regex_answer = extract_regex(response_text, cfg.dataset)
                # --- 2) Likelihood over options (A/B/C/D) ---
                pl, _, _, _, choices = generate_prompts(sample, cfg.lang, likelihood=True)
                option_scores = model.score_options(pl, img_uri, choices)
                abcd = ["A", "B", "C", "D"]
                best_idx = max(range(len(choices)), key=lambda i: option_scores[i])
                likelihood_answer = abcd[best_idx]
            else:
                regex_answer = extract_regex(response_text, cfg.dataset)
                pl = pr
                likelihood_answer = 'NA'
                option_scores = [0]
                choices = ['NA']
    
            # --- 3) final aggregation ---
            if cfg.eval_method == "regex":
                final_answer = regex_answer
            elif cfg.eval_method == "likelihood":
                final_answer = likelihood_answer
            else:  # hybrid
                final_answer = regex_answer if regex_answer != "random" else likelihood_answer
    
            row = {
                "question_id": qid,
                "prompt_regex": pr,
                "prompt_likelihood": pl,
                "response_text": response_text,
                "regex_answer": regex_answer,
                "likelihood_answer": likelihood_answer,
                "option_scores": {c: float(s) for c, s in zip(choices, option_scores)},
                "final_answer": final_answer,
                "gold": gold,
                "image": img_path,
                "taxonomy": sample['hierarchy']
            }
        except:
            row = {
                "question_id": qid,
                "prompt_regex": 'ERROR',
                "prompt_likelihood": 'ERROR',
                "response_text": 'ERROR',
                "regex_answer": 'ERROR',
                "likelihood_answer": 'ERROR',
                "option_scores": 'ERROR',
                "final_answer": 'ERROR',
                "gold": 'ERROR',
                "image": 'ERROR',
                "taxonomy": 'ERROR'
            }

        rows.append(row)

        # --- Incremental caching (append to CSV) ---
        pd.DataFrame([row]).to_csv(out_csv, mode="a", header=not out_csv.exists(), index=False, encoding="utf-8")

    # Final full save (optional, to ensure consistent file)
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False, encoding="utf-8")
    print(f"Saved results to {out_csv}")
