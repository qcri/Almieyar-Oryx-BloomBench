import torch
from transformers import AutoProcessor, AutoTokenizer
from transformers import Qwen2VLForConditionalGeneration, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info
from .base import BaseVLM

class QwenVLM(BaseVLM):
    def __init__(self, cfg, device="cpu", version="2.5", override_model_id=None):
        self.version = version
        self.override_model_id = override_model_id
        super().__init__(cfg, device=device)

    def load(self):
        model_id = self.override_model_id or ("Qwen/Qwen2.5-VL-7B-Instruct" if self.version=="2.5" else "Qwen/Qwen2-VL-7B-Instruct")
        if self.version == "2.5":
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_id, torch_dtype="auto", device_map=self.device, token=self.cfg.hf_token)
        else:
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(model_id, torch_dtype="auto", device_map=self.device, token=self.cfg.hf_token)
        self.processor = AutoProcessor.from_pretrained(model_id, token=self.cfg.hf_token)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, token=self.cfg.hf_token)

    def generate(self, prompt: str, image_uri: str, lang: str):
        messages = [{"role":"user","content":[{"type":"image","image": image_uri},{"type":"text","text": prompt}]}]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt").to(self.device)
        with torch.inference_mode():
            gen = self.model.generate(**inputs, max_new_tokens=128, temperature=self.cfg.temperature, do_sample=self.cfg.temperature>0, return_dict_in_generate=True, output_scores=True)
        first_token_probs = torch.softmax(gen.scores[0], dim=-1)
        candidate_token = ['1','2','3','4'] if self.cfg.dataset!="camel" else ['أ','ب','ج','د']
        cand_ids = [self.tokenizer.encode(c) for c in candidate_token]
        prob_dict = {}
        for j in set(sum(cand_ids, [])):
            prob_dict[self.tokenizer.decode(j)] = first_token_probs[0][j].item()
        # decode
        generated_ids = gen.sequences
        generated_ids_trimmed = [out_ids[len(inputs.input_ids[0]):] for out_ids in generated_ids]
        output_text = self.processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        return output_text, prob_dict

    def score_options(self, base_prompt: str, image_uri: str, options):
        from eval.likelihood import score_candidate
        scores = []
        # context
        messages = [{"role":"user","content":[{"type":"image","image": image_uri},{"type":"text","text": base_prompt}]}]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        context_inputs = self.processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt").to(self.device)
        for opt in options:
            messages = [{"role":"user","content":[{"type":"image","image": image_uri},{"type":"text","text": base_prompt + opt}]}]
            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(messages)
            full_inputs = self.processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt").to(self.device)
            scores.append(score_candidate(context_inputs, full_inputs, self.model, self.tokenizer))
        return scores