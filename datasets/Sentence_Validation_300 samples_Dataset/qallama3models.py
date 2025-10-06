import os
import random
import numpy as np
import pandas as pd
import torch
from transformers import (
    LlamaTokenizerFast,
    LlamaForCausalLM,
    BitsAndBytesConfig
)

from huggingface_hub import login

login(token="Please Insert Token")
# 1) Reproducibility
SEED = 4000
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# 2) 4-bit quantization config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

# 3) Device
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 4) Models to run
MODELS = [
    ("meta-llama/Llama-3.1-8B-Instruct", "8"),
    ( "meta-llama/Llama-3.3-70B-Instruct", "70"),
]

import re

# 5) Ask the model and parse out <Answer> tag
def questionanswerer(question: str, tokenizer, model) -> str:
    prompt = f"""
You are a scientific reasoning assistant specialized in gene–disease causality.
For each yes/no question, think step by step, then output exactly three tagged sections:

<Question>{question}</Question>
<ChainOfThought>
1. …
2. …
3. …
</ChainOfThought>
<Answer>1 or 0</Answer>

• 1 means “Yes—the evidence supports a causal role.”
• 0 means “No, Unclear, or unsupported.”
• <ChainOfThought> must list numbered steps.
• <Answer> must be exactly one digit and nothing else.
"""
    # tokenize & generate
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    out = model.generate(
        **inputs,
        max_new_tokens=256,
        do_sample=False,
        eos_token_id=tokenizer.eos_token_id
    )
    text = tokenizer.decode(out[0], skip_special_tokens=True)

    # extract the first <Answer>…</Answer>
    m = re.search(r"<Answer>\s*([01])\s*</Answer>", text)
    return m.group(1) if m else "0"

# 6) Batch over your CSV
if __name__ == "__main__":
    df = pd.read_csv("data/QA.csv")

    for model_name, tag in MODELS:
        os.environ["HF_HOME"]        = "/data0/projects/Causal_and_Agentic_AI/huggingface_cache"
        os.environ["TRANSFORMERS_CACHE"] = os.path.join(os.environ["HF_HOME"], "transformers")
        # load tokenizer & model
        tokenizer = LlamaTokenizerFast.from_pretrained(model_name, trust_remote_code=True, use_fast=False)
        model = LlamaForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True
        )
        model.to(DEVICE)

        # apply to each question
        df_out = df.copy()
        df_out["answer"] = df_out["question"].apply(lambda q: questionanswerer(q, tokenizer, model))

        # save results
        out_path = f"results/llama/QAresults_llama3_{tag}B_GeneDisease.csv"
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        df_out.to_csv(out_path, index=False)
        print(f"→ Wrote {out_path}")
