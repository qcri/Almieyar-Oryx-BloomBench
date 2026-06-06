from dataclasses import dataclass, field
from typing import List, Optional
from pathlib import Path
import yaml

# @dataclass
# class MergeWeight:
#     path: str
#     alpha: Optional[float] = None
    

@dataclass
class VLMConfig:
    exp: str
    model_name: str
    dataset: str
    lang: str
    temperature: float = 0.0
    eval_method: str = "hybrid"  # regex | likelihood | hybrid
    cache_dir: str = "./cache"
    device_map: Optional[str] = None
    gpu_id: Optional[int] = None
    hf_token: Optional[str] = None
    
    lora: Optional[str] = None



def load_config(path: str) -> VLMConfig:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with p.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    cfg = data.get("VLMConfig", {})
    return VLMConfig(**cfg)