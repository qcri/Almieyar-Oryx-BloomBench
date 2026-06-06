import json, os, tarfile
from huggingface_hub import hf_hub_download
from pathlib import Path

def load_bloom(cfg):
    file_path = hf_hub_download(repo_id="", filename="google_images_v2.tar.gz", repo_type="dataset", token=cfg.hf_token, local_dir='./')

    with tarfile.open(file_path, 'r:gz') as tar:
        if not os.path.exists("google_images"):
            # os.makedirs("google_images", exist_ok=True)
            tar.extractall()
    file_path = hf_hub_download(repo_id="", filename="final_oryx_v2.json", repo_type="dataset", token=cfg.hf_token, local_dir='./')

    import json
    with open('final_oryx_v2.json', 'r', encoding='utf-8') as f:
        dataset = json.load(f)

    for i in dataset:
        i['hierarchy']['lvl1'] = ' -> '.join([j.lower().strip() for j in i['hierarchy']['lvl1'].split('->')])
        i['hierarchy']['leaf'] = i['hierarchy']['leaf'].lower().strip()

    return dataset