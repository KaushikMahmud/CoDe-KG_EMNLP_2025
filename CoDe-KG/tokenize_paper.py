# tokenize_paper.py
import re
from typing import List, Tuple

def tokenize(text: str, strip_punct: bool = True) -> List[str]:
    if strip_punct:
        out = []
        for token in text.split():
            cleaned = re.sub(r'^\W+|\W+$', '', token)
            if cleaned:
                out.append(cleaned)
        return out
    return text.split()

def tokens_with_indices(text: str, keep_punct: bool = True) -> List[Tuple[str, int]]:
    toks = tokenize(text, strip_punct=not keep_punct)
    return [(t, i) for i, t in enumerate(toks)]

def format_pairs(pairs: List[Tuple[str, int]]) -> str:
    return ", ".join(f'("{t}", {i})' for (t, i) in pairs)

def parse_tokenized_pairs(tokenized_text: str) -> List[Tuple[str, int]]:
    pattern = r'\("([^"]*)",\s*(\d+)\)'
    matches = re.findall(pattern, tokenized_text)
    return [(token, int(idx)) for token, idx in matches]
