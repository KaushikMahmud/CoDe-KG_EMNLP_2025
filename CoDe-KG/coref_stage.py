# coref_stage.py
import re
from typing import Dict, List, Tuple, Any
from tokenize_paper import parse_tokenized_pairs
from prompts import build_prompt, COREf_FICL_SYSTEM, COREf_FICL_USER

def _extract_coref_blocks(raw: str) -> List[Dict[str, Any]]:
    """
    Regex-first extraction for blocks that contain:
    Expression, StartToken, EndToken, RefersTo
    Works even if JSON is slightly malformed.
    """
    blocks = []

    # Find dict-like chunks
    # This is robust to many formats: { "Expression": "...", ... }
    candidates = re.findall(r'\{[^{}]*"Expression"[^{}]*\}', raw, flags=re.DOTALL)
    if not candidates:
        candidates = re.findall(r'\{[^{}]*Expression[^{}]*\}', raw, flags=re.DOTALL)

    for c in candidates:
        exp = re.search(r'Expression"\s*:\s*"([^"]+)"', c)
        if exp is None:
            exp = re.search(r'Expression\s*:\s*"([^"]+)"', c)

        st = re.search(r'StartToken"\s*:\s*(\d+)', c)
        if st is None:
            st = re.search(r'StartToken\s*:\s*(\d+)', c)

        en = re.search(r'EndToken"\s*:\s*(\d+)', c)
        if en is None:
            en = re.search(r'EndToken\s*:\s*(\d+)', c)

        rf = re.search(r'RefersTo"\s*:\s*"([^"]+)"', c)
        if rf is None:
            rf = re.search(r'RefersTo\s*:\s*"([^"]+)"', c)

        if exp and st and en and rf:
            blocks.append({
                "Expression": exp.group(1),
                "StartToken": int(st.group(1)),
                "EndToken": int(en.group(1)),
                "RefersTo": rf.group(1),
            })

    return blocks

def build_coref_prompt(tokenized_text: str) -> str:
    return build_prompt(COREf_FICL_SYSTEM, COREf_FICL_USER, tokenized_text=tokenized_text)

def apply_coref_switch(tokenized_text: str, corefs: List[Dict[str, Any]]) -> str:
    tokens = parse_tokenized_pairs(tokenized_text)
    token_map = {idx: tok for tok, idx in tokens}

    # Sort by StartToken increasing (important for consistent replacement)
    corefs_sorted = sorted(corefs, key=lambda x: (x["StartToken"], x["EndToken"]))

    # replacements: token_idx -> string or None (drop)
    replacements: Dict[int, str | None] = {}

    for c in corefs_sorted:
        start_token = int(c["StartToken"])
        end_token = int(c["EndToken"])
        refers_to = str(c["RefersTo"]).strip()

        for token_idx in range(start_token, end_token + 1):
            if token_idx == start_token:
                replacements[token_idx] = refers_to
            else:
                replacements[token_idx] = None

    # rebuild token list in original index order
    result_tokens: List[str] = []
    for tok, idx in tokens:
        if idx in replacements:
            if replacements[idx] is not None:
                result_tokens.append(replacements[idx])
        else:
            result_tokens.append(tok)

    text = " ".join(result_tokens)

    # punctuation cleanup similar to yours
    text = re.sub(r"\s+([.,;:!?])", r"\1", text)
    text = re.sub(r"\(\s+", "(", text)
    text = re.sub(r"\s+\)", ")", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def parse_coref_output(raw: str) -> List[Dict[str, Any]]:
    blocks = _extract_coref_blocks(raw)
    return blocks
