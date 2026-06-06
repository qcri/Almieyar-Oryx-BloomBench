import torch
import torch.nn.functional as F

def score_candidate(context_inputs, full_inputs, model, tokenizer):
    context_len = context_inputs["input_ids"].shape[1]
    input_ids = full_inputs["input_ids"]
    with torch.no_grad():
        outputs = model(**full_inputs)
        logits = outputs.logits
    answer_logits = logits[:, context_len-1:-1, :]
    answer_ids = input_ids[:, context_len:]
    log_probs = F.log_softmax(answer_logits, dim=-1)
    token_log_probs = log_probs.gather(2, answer_ids.unsqueeze(-1)).squeeze(-1)
    return token_log_probs.mean().item()