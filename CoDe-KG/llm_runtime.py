# llm_runtime.py
import os
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TextGenerationPipeline

def set_seed(seed: int = 4000) -> None:
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class LLM:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.pipe = None

    def load(self) -> None:
        bnb_config = BitsAndBytesConfig(load_in_4bit=True)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, padding_side="left")
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=bnb_config,
            device_map="auto",
        )
        self.pipe = TextGenerationPipeline(model=self.model, tokenizer=self.tokenizer)

    def generate_batch(
        self,
        prompts,
        max_new_tokens: int,
        batch_size: int,
        temperature: float,
        top_k: int = 40,
        top_p: float = 0.9,
        do_sample: bool = True,
        repetition_penalty: float = 1.0,
        return_full_text: bool = False,
    ):
        if self.pipe is None:
            self.load()

        # Safe generation: try full batch; on OOM, reduce batch then reduce tokens.
        cur_batch = batch_size
        cur_tokens = max_new_tokens

        while True:
            try:
                outs = self.pipe(
                    prompts,
                    max_new_tokens=cur_tokens,
                    batch_size=cur_batch,
                    truncation=True,
                    return_full_text=return_full_text,
                    pad_token_id=self.tokenizer.eos_token_id,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    do_sample=do_sample,
                    repetition_penalty=repetition_penalty,
                    use_cache=True,
                )
                return outs
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                time.sleep(0.2)

                if cur_batch > 1:
                    cur_batch = max(1, cur_batch // 2)
                    continue

                if cur_tokens > 256:
                    cur_tokens = max(256, cur_tokens // 2)
                    continue

                # Last resort: fail loudly so you can see it.
                raise
