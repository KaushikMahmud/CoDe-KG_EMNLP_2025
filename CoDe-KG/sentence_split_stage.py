# sentence_split_stage.py
from typing import List
import nltk
from nltk.tokenize import sent_tokenize

def setup_nltk() -> None:
    for resource in ["punkt"]:
        try:
            sent_tokenize("Test sentence.")
            return
        except LookupError:
            nltk.download(resource, quiet=True)

def split_sentences(text: str) -> List[str]:
    if not text or not text.strip():
        return []
    setup_nltk()
    return [s.strip() for s in sent_tokenize(text.strip()) if s.strip()]
