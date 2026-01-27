# simplify_stage.py
import re
from typing import List, Dict, Any
from prompts import (
    build_prompt,
    SIMPLIFY_COMPOUND_SYSTEM, SIMPLIFY_COMPOUND_USER,
    SIMPLIFY_COMPLEX_SYSTEM, SIMPLIFY_COMPLEX_USER,
    SIMPLIFY_COMPOUND_COMPLEX_SYSTEM, SIMPLIFY_COMPOUND_COMPLEX_USER,
)

def extract_simplified_sentences(raw: str) -> List[str]:
    sents = []
    for line in raw.splitlines():
        line = line.strip()
        if not line.startswith("S"):
            continue

        # Accept both "S1 -> ..." and "S1 → ..."
        if "->" in line:
            parts = line.split("->", 1)
        elif "→" in line:
            parts = line.split("→", 1)
        else:
            continue

        if len(parts) == 2:
            sent = parts[1].strip()
            if sent:
                sents.append(sent)
    return sents


def simplify_sentences_for_class(llm, sentence_type: str, sentences: List[str], batch_size: int) -> List[Dict[str, Any]]:
    rows = []
    if not sentences:
        return rows

    if sentence_type == "Compound Sentence":
        sys, usr = SIMPLIFY_COMPOUND_SYSTEM, SIMPLIFY_COMPOUND_USER
        max_new_tokens, temperature = 1024, 0.7
        top_k, top_p = 40, 0.9
    elif sentence_type == "Complex Sentence":
        sys, usr = SIMPLIFY_COMPLEX_SYSTEM, SIMPLIFY_COMPLEX_USER
        max_new_tokens, temperature = 512, 0.3
        top_k, top_p = 40, 0.9
    elif sentence_type == "Compound-Complex Sentence":
        sys, usr = SIMPLIFY_COMPOUND_COMPLEX_SYSTEM, SIMPLIFY_COMPOUND_COMPLEX_USER
        max_new_tokens, temperature = 1024, 0.7
        top_k, top_p = 40, 0.9
    else:
        raise ValueError(f"Unsupported sentence_type: {sentence_type}")

    prompts = [build_prompt(sys, usr, sentence=s) for s in sentences]
    outs = llm.generate_batch(
        prompts=prompts,
        max_new_tokens=max_new_tokens,
        batch_size=batch_size,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        do_sample=True,
        return_full_text=False,
    )

    for orig, out in zip(sentences, outs):
        raw = out[0]["generated_text"].strip()
        simplified = extract_simplified_sentences(raw)
        rows.append({
            "sentence_type": sentence_type,
            "original_sentence": orig,
            "simplified_sentences": simplified,
            "raw_output": raw,
        })

    return rows
