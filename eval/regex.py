import re
def extract_regex(response: str, dataset: str) -> str:
    if dataset != "camel":
        match = re.findall(r"[1-4]", response.strip())
        if len(match) == 1:
            return {"1":"A","2":"B","3":"C","4":"D"}[match[0]]
        return "random"
    else:
        match = re.findall(r"[أب‌ج‌د]", response.strip())
        if len(match) == 1:
            return {"أ": "A", "ب": "B", "ج": "C", "د": "D"}[match[0]]
        return "random"