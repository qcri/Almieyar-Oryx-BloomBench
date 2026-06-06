from .bloom import load_bloom
from .camel import load_camel
from .task_galaxy import load_task_galaxy
from .arabic_cultures import load_arabic_cultures

def load_dataset_by_name(cfg):
    if cfg.dataset == "bloom":
        return load_bloom(cfg)
    if cfg.dataset == "camel":
        return load_camel(cfg)
    if cfg.dataset == "task_galaxy":
        return load_task_galaxy(cfg)
    if cfg.dataset == "arabic_cultures":
        return load_arabic_cultures(cfg)
    raise ValueError(f"Unknown dataset: {cfg.dataset}")