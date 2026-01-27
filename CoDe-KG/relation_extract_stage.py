# relation_extract_stage.py
import re
from typing import List, Dict, Any
from prompts import build_prompt, REL_EXTRACT_SYSTEM, REL_EXTRACT_USER

def extract_triples_regex(raw: str) -> List[Dict[str, str]]:
    """
    Tries to extract repeated blocks containing:
    "Entity 1": "...", "Relationship": "...", "Entity 2": "..."
    Works even with JSON-ish output.
    """
    triples = []
    blocks = re.findall(r'\{[^{}]*Entity\s*1[^{}]*\}', raw, flags=re.DOTALL)
    if not blocks:
        # fallback: look across entire text for triplets
        blocks = [raw]

    for b in blocks:
        e1 = re.search(r'Entity\s*1"\s*:\s*"([^"]+)"', b)
        if e1 is None:
            e1 = re.search(r'Entity\s*1\s*:\s*"([^"]+)"', b)

        rel = re.search(r'Relationship"\s*:\s*"([^"]+)"', b)
        if rel is None:
            rel = re.search(r'Relationship\s*:\s*"([^"]+)"', b)

        e2 = re.search(r'Entity\s*2"\s*:\s*"([^"]+)"', b)
        if e2 is None:
            e2 = re.search(r'Entity\s*2\s*:\s*"([^"]+)"', b)

        if e1 and rel and e2:
            triples.append({
                "Entity 1": e1.group(1).strip(),
                "Relationship": rel.group(1).strip(),
                "Entity 2": e2.group(1).strip(),
            })

    return triples

def build_relation_prompt(sentence: str) -> str:
    return build_prompt(REL_EXTRACT_SYSTEM, REL_EXTRACT_USER, sentence=sentence)

def extract_relationships(llm, sentences: List[str], batch_size: int) -> List[Dict[str, Any]]:
    rows = []
    if not sentences:
        return rows

    prompts = [build_relation_prompt(s) for s in sentences]
    outs = llm.generate_batch(
        prompts=prompts,
        max_new_tokens=512,
        batch_size=batch_size,
        temperature=0.3,
        top_k=40,
        top_p=0.9,
        do_sample=True,
        return_full_text=False,
    )

    for sent, out in zip(sentences, outs):
        raw = out[0]["generated_text"].strip()
        triples = extract_triples_regex(raw)
        rows.append({
            "sentence": sent,
            "triples": triples,
            "raw_output": raw,
        })
    return rows
