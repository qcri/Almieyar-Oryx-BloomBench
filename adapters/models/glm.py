import torch
from transformers import AutoProcessor, AutoTokenizer, AutoModelForCausalLM
from .base import BaseVLM

class GLMVLM(BaseVLM):
    def __init__(self, cfg, device="cpu"):
        super().__init__(cfg, device=device)

    def load(self):
        model_id = "zai-org/GLM-4.6V"
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, 
            torch_dtype=torch.bfloat16, 
            trust_remote_code=True,
            device_map=self.device,
            token=self.cfg.hf_token
        ).eval()
        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True, token=self.cfg.hf_token)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, token=self.cfg.hf_token)

    def generate(self, prompt: str, image_uri: str, lang: str):
        from PIL import Image
        import requests
        from io import BytesIO
        
        # Load image
        if image_uri.startswith('http'):
            response = requests.get(image_uri)
            image = Image.open(BytesIO(response.content))
        else:
            image = Image.open(image_uri)
        
        # Prepare messages
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt}
                ]
            }
        ]
        
        # Process inputs
        inputs = self.processor(messages, return_tensors="pt").to(self.device)
        
        # Generate
        with torch.inference_mode():
            gen = self.model.generate(
                **inputs,
                max_new_tokens=128,
                temperature=self.cfg.temperature,
                do_sample=self.cfg.temperature > 0,
                return_dict_in_generate=True,
                output_scores=True
            )
        
        # Get first token probabilities
        first_token_probs = torch.softmax(gen.scores[0], dim=-1)
        candidate_token = ['1', '2', '3', '4'] if self.cfg.dataset != "camel" else ['أ', 'ب', 'ج', 'د']
        cand_ids = [self.tokenizer.encode(c, add_special_tokens=False) for c in candidate_token]
        prob_dict = {}
        for j in set(sum(cand_ids, [])):
            prob_dict[self.tokenizer.decode(j)] = first_token_probs[0][j].item()
        
        # Decode output
        generated_ids = gen.sequences
        generated_ids_trimmed = [out_ids[len(inputs.input_ids[0]):] for out_ids in generated_ids]
        output_text = self.tokenizer.batch_decode(
            generated_ids_trimmed, 
            skip_special_tokens=True, 
            clean_up_tokenization_spaces=False
        )[0]
        
        return output_text, prob_dict

    def score_options(self, base_prompt: str, image_uri: str, options):
        from PIL import Image
        import requests
        from io import BytesIO
        from eval.likelihood import score_candidate
        
        # Load image
        if image_uri.startswith('http'):
            response = requests.get(image_uri)
            image = Image.open(BytesIO(response.content))
        else:
            image = Image.open(image_uri)
        
        scores = []
        
        # Context messages
        context_messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": base_prompt}
                ]
            }
        ]
        context_inputs = self.processor(context_messages, return_tensors="pt").to(self.device)
        
        for opt in options:
            full_messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": base_prompt + opt}
                    ]
                }
            ]
            full_inputs = self.processor(full_messages, return_tensors="pt").to(self.device)
            scores.append(score_candidate(context_inputs, full_inputs, self.model, self.tokenizer))
        
        return scores
