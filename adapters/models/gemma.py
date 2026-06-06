import torch
from transformers import AutoProcessor, AutoTokenizer, Gemma3ForConditionalGeneration
from .base import BaseVLM

class GemmaVLM(BaseVLM):
    def load(self):
        model_id = self.cfg.model_name
        self.model = Gemma3ForConditionalGeneration.from_pretrained(model_id, device_map=self.device, token=self.cfg.hf_token).eval()
        self.processor = AutoProcessor.from_pretrained(model_id, token=self.cfg.hf_token)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, token=self.cfg.hf_token)

    def generate(self, prompt: str, image_uri: str, lang: str):
        messages = [{"role":"user","content":[{"type":"image","image": image_uri},{"type":"text","text": prompt}]}]
        inputs = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt").to(self.device, dtype=torch.bfloat16)
        input_len = inputs["input_ids"].shape[-1]
        with torch.inference_mode():
            gen = self.model.generate(**inputs, max_new_tokens=128, temperature=self.cfg.temperature, do_sample=self.cfg.temperature>0, return_dict_in_generate=True, output_scores=True)
        first_token_probs = torch.softmax(gen.scores[0], dim=-1)
        candidate_token = ['1','2','3','4'] if self.cfg.dataset!= "camel" else ['أ','ب','ج','د']
        cand_ids = [self.tokenizer.encode(c) for c in candidate_token]
        prob_dict = {}
        for j in set(sum(cand_ids, [])):
            prob_dict[self.tokenizer.decode(j)] = first_token_probs[0][j].item()
        seq = gen.sequences[0][input_len:]
        output_text = self.processor.decode(seq, skip_special_tokens=True)
        return output_text, prob_dict

    def score_options(self, base_prompt: str, image_uri: str, options):
        from eval.likelihood import score_candidate
        scores = []
        messages = [{"role":"user","content":[{"type":"image","image": image_uri},{"type":"text","text": base_prompt}]}]
        context_inputs = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt").to(self.device, dtype=torch.bfloat16)
        for opt in options:
            messages = [{"role":"user","content":[{"type":"image","image": image_uri},{"type":"text","text": base_prompt + opt}]}]
            full_inputs = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt").to(self.device, dtype=torch.bfloat16)
            scores.append(score_candidate(context_inputs, full_inputs, self.model, self.tokenizer))
        return scores