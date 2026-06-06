class BaseVLM:
    def __init__(self, cfg, device="cpu"):
        self.cfg = cfg
        self.device = device
        self.model = None
        self.processor = None
        self.tokenizer = None
        self.load()

    def load(self):
        raise NotImplementedError

    def generate(self, prompt: str, image_uri: str, lang: str = "en"):
        """Return (response_text, token_probabilities_dict-for-choices-first-token)"""
        raise NotImplementedError

    def score_options(self, base_prompt: str, image_uri: str, options):
        """Return a list of avg log-likelihoods per option."""
        raise NotImplementedError