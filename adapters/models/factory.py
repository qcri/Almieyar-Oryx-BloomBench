from .qwen import QwenVLM
from .gemma import GemmaVLM
from .glm import GLMVLM

def get_model_adapter(cfg, device="cpu"):
    name = cfg.model_name.lower()
    if "glm-4.6v" in name or "glm" in name:
        return GLMVLM(cfg, device=device)
    if "gemma-3" in name:
        return GemmaVLM(cfg, device=device)
    if "qwen2.5-vl" in name or "qwen2.5" in name:
        return QwenVLM(cfg, device=device, version="2.5")
    if "qwen2-vl" in name or "qwen2" in name or "mbzuai/ain" in name:
        # Treat AIN as Qwen2-compatible
        override_id = "MBZUAI/AIN" if "mbzuai/ain" in name else None
        return QwenVLM(cfg, device=device, version="2", override_model_id=override_id)
    raise ValueError(f"Unknown model_name: {cfg.model_name}")
