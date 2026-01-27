# bert_sentence_classifier.py
import os
from dataclasses import dataclass
from typing import List, Dict, Optional

import pandas as pd
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)

@dataclass
class BertClassifierConfig:
    model_name: str = "bert-large-uncased"
    num_labels: int = 5  # Simple, Compound, Complex, Compound-Complex
    out_dir: str = "models/bert_sentence_classifier"
    max_length: int = 512

LABEL_MAP_DEFAULT = {
    "Simple Sentence": 0,
    "Compound Sentence": 1,
    "Complex Sentence": 2,
    "Compound-Complex Sentence": 3,
    "Incomplete Sentence": 4,
}
INV_LABEL_MAP_DEFAULT = {v: k for k, v in LABEL_MAP_DEFAULT.items()}

class CSVDataset(torch.utils.data.Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer, text_col: str, label_col: str, label_map: Dict[str, int], max_length: int):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.text_col = text_col
        self.label_col = label_col
        self.label_map = label_map
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        text = str(self.df.loc[idx, self.text_col])
        label_str = str(self.df.loc[idx, self.label_col])
        label = self.label_map[label_str]

        enc = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
        )
        enc["labels"] = label
        return enc

class BertSentenceClassifier:
    def __init__(self, cfg: BertClassifierConfig):
        self.cfg = cfg
        self.tokenizer = None
        self.model = None

    def exists(self) -> bool:
        return os.path.isdir(self.cfg.out_dir) and os.path.isfile(os.path.join(self.cfg.out_dir, "config.json"))

    def load(self) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.out_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.cfg.out_dir)

    def train_if_missing(
        self,
        train_csv: Optional[str],
        text_col: str,
        label_col: str,
        label_map: Dict[str, int] = LABEL_MAP_DEFAULT,
        epochs: int = 3,
        batch_size: int = 8,
        lr: float = 2e-5,
    ) -> None:
        if self.exists():
            self.load()
            return

        if train_csv is None:
            raise ValueError("No finetuned BERT found and --bert_train_csv was not provided.")

        try:
            df = pd.read_csv(train_csv, encoding="utf-8")
        except UnicodeDecodeError:
            try:
                df = pd.read_csv(train_csv, encoding="utf-8-sig")
            except UnicodeDecodeError:
                df = pd.read_csv(train_csv, encoding="latin-1")

        self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.cfg.model_name,
            num_labels=self.cfg.num_labels
        )

        ds = CSVDataset(df, self.tokenizer, text_col, label_col, label_map, self.cfg.max_length)
        collator = DataCollatorWithPadding(tokenizer=self.tokenizer)

        args = TrainingArguments(
            output_dir=self.cfg.out_dir,
            overwrite_output_dir=True,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            learning_rate=lr,
            logging_steps=50,
            save_strategy="epoch",
            save_total_limit=1,
            report_to=[],
        )

        trainer = Trainer(
            model=self.model,
            args=args,
            train_dataset=ds,
            tokenizer=self.tokenizer,
            data_collator=collator,
        )
        trainer.train()
        trainer.save_model(self.cfg.out_dir)
        self.tokenizer.save_pretrained(self.cfg.out_dir)

        # Reload cleanly
        self.load()

    @torch.inference_mode()
    def predict_batch(self, sentences: List[str], batch_size: int = 32) -> List[Dict[str, object]]:
        if self.model is None or self.tokenizer is None:
            if self.exists():
                self.load()
            else:
                raise ValueError("BERT model missing. Train it or provide --train_bert_if_missing.")

        self.model.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)

        out = []
        for i in range(0, len(sentences), batch_size):
            batch = sentences[i:i+batch_size]
            enc = self.tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=self.cfg.max_length).to(device)
            logits = self.model(**enc).logits
            probs = torch.softmax(logits, dim=-1)
            pred = torch.argmax(probs, dim=-1).tolist()
            conf = torch.max(probs, dim=-1).values.tolist()

            for s, p, c in zip(batch, pred, conf):
                out.append({
                    "sentence": s,
                    "predicted_label": INV_LABEL_MAP_DEFAULT[int(p)],
                    "confidence": float(c),
                })
        return out
