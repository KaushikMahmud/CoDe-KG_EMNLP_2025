# run_pipeline.py
import os
import argparse
import pandas as pd

from tokenize_paper import tokens_with_indices, format_pairs
from llm_runtime import LLM, set_seed
from coref_stage import build_coref_prompt, parse_coref_output, apply_coref_switch
from sentence_split_stage import split_sentences
from bert_sentence_classifier import BertSentenceClassifier, BertClassifierConfig
from simplify_stage import simplify_sentences_for_class
from relation_extract_stage import extract_relationships

MIXTRAL = "mistralai/Mixtral-8x7B-Instruct-v0.1"
LLAMA_31_8B = "meta-llama/Llama-3.1-8B-Instruct"
LLAMA_33_70B = "meta-llama/Llama-3.3-70B-Instruct"

def read_abstracts(args) -> pd.DataFrame:
    if args.input_csv:
        df = pd.read_csv(args.input_csv)
        if args.text_col not in df.columns:
            raise ValueError(f"--text_col '{args.text_col}' not found in {args.input_csv}. Columns: {list(df.columns)}")
        df = df.reset_index(drop=True)
        if args.id_col and args.id_col in df.columns:
            df["doc_id"] = df[args.id_col].astype(str)
        else:
            df["doc_id"] = [f"doc_{i}" for i in range(len(df))]
        df["abstract"] = df[args.text_col].astype(str)
        return df[["doc_id", "abstract"]]

    if args.abstract:
        return pd.DataFrame([{"doc_id": "doc_0", "abstract": args.abstract}])

    # stdin
    text = input("Enter abstract: ").strip()
    return pd.DataFrame([{"doc_id": "doc_0", "abstract": text}])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", type=str, default=None, help="CSV containing abstracts")
    parser.add_argument("--text_col", type=str, default="abstract", help="Column containing abstract text")
    parser.add_argument("--id_col", type=str, default=None, help="Optional doc id column")

    parser.add_argument("--abstract", type=str, default=None, help="Single abstract (if not using --input_csv)")
    parser.add_argument("--out_csv", type=str, required=True, help="Output CSV path")

    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=4000)

    # BERT
    parser.add_argument("--bert_out_dir", type=str, default="models/bert_sentence_classifier")
    parser.add_argument("--train_bert_if_missing", action="store_true")
    parser.add_argument("--bert_train_csv", type=str, default="train.csv")
    parser.add_argument("--bert_text_col", type=str, default="sentence")
    parser.add_argument("--bert_label_col", type=str, default="label")
    parser.add_argument("--bert_epochs", type=int, default=3)
    parser.add_argument("--bert_train_batch_size", type=int, default=8)
    parser.add_argument("--bert_lr", type=float, default=2e-5)

    args = parser.parse_args()
    set_seed(args.seed)

    df_abs = read_abstracts(args)

    # Load LLMs (lazy: only load when needed)
    mixtral_llm = LLM(MIXTRAL)
    llama8_llm = LLM(LLAMA_31_8B)
    llama70_llm = LLM(LLAMA_33_70B)

    # BERT classifier (train once if needed)
    bert_cfg = BertClassifierConfig(out_dir=args.bert_out_dir)
    bert = BertSentenceClassifier(bert_cfg)
    if bert.exists():
        bert.load()
    else:
        if args.train_bert_if_missing:
            bert.train_if_missing(
                train_csv=args.bert_train_csv,
                text_col=args.bert_text_col,
                label_col=args.bert_label_col,
                epochs=args.bert_epochs,
                batch_size=args.bert_train_batch_size,
                lr=args.bert_lr,
            )
        else:
            raise ValueError(f"No finetuned BERT found at {args.bert_out_dir}. Use --train_bert_if_missing.")

    all_rows = []

    for _, row in df_abs.iterrows():
        doc_id = row["doc_id"]
        abstract = row["abstract"]

        # 1) Tokenize as in paper
        pairs = tokens_with_indices(abstract, keep_punct=True)
        tokenized_text = format_pairs(pairs)

        # 2) Coref with Mixtral
        coref_prompt = build_coref_prompt(tokenized_text)
        coref_out = mixtral_llm.generate_batch(
            prompts=[coref_prompt],
            max_new_tokens=1024,
            batch_size=1,
            temperature=0.7,
            top_k=40,
            top_p=0.9,
            do_sample=True,
            return_full_text=False,
        )[0][0]["generated_text"].strip()

        corefs = parse_coref_output(coref_out)

        # 3) Switch coref into text (PBC -> Primary biliary cirrhosis at indices)
        switched = apply_coref_switch(tokenized_text, corefs)

        print(f"\n[DOC {doc_id}] ==============================")
        print(f"[DOC {doc_id}] Abstract chars: {len(abstract)}")

        print(f"[DOC {doc_id}] Tokens: {len(pairs)}")
        print(f"[DOC {doc_id}] Coref spans: {len(corefs)}")

        print(f"[DOC {doc_id}] Switched snippet: {switched[:200]}...")



        # 4) Sentence split
        sentences = split_sentences(switched)

        print(f"[DOC {doc_id}] Sentences: {len(sentences)}")
        if len(sentences) > 0:
            print(f"[DOC {doc_id}] Sentence[0]: {sentences[0]}")


        # 5) BERT classify
        preds = bert.predict_batch(sentences, batch_size=32)
        cls_df = pd.DataFrame(preds)

        

        simple_sents = cls_df[cls_df["predicted_label"] == "Simple Sentence"]["sentence"].tolist()
        compound_sents = cls_df[cls_df["predicted_label"] == "Compound Sentence"]["sentence"].tolist()
        complex_sents = cls_df[cls_df["predicted_label"] == "Complex Sentence"]["sentence"].tolist()
        cc_sents = cls_df[cls_df["predicted_label"] == "Compound-Complex Sentence"]["sentence"].tolist()

        print(f"[DOC {doc_id}] Label counts → "
        f"Simple={len(simple_sents)} | "
        f"Compound={len(compound_sents)} | "
        f"Complex={len(complex_sents)} | "
        f"Compound-Complex={len(cc_sents)}")

        if len(sentences) > 0 and (len(simple_sents)+len(compound_sents)+len(complex_sents)+len(cc_sents)) != len(sentences):
          print(f"[DOC {doc_id}] ⚠️ WARNING: label split count != total sentences")

        # 6) Simplify (route to correct LLM)
        simplified_rows = []
        if compound_sents:
            simplified_rows += simplify_sentences_for_class(llama8_llm, "Compound Sentence", compound_sents, args.batch_size)
        if complex_sents:
            simplified_rows += simplify_sentences_for_class(llama70_llm, "Complex Sentence", complex_sents, args.batch_size)
        if cc_sents:
            simplified_rows += simplify_sentences_for_class(mixtral_llm, "Compound-Complex Sentence", cc_sents, args.batch_size)

        converted_simple = []
        for r in simplified_rows:
            converted_simple.extend(r.get("simplified_sentences") or [])

        final_simple_sentences = simple_sents + converted_simple

        print(f"[DOC {doc_id}] Simplify input counts → "
        f"Compound={len(compound_sents)} | "
        f"Complex={len(complex_sents)} | "
        f"Compound-Complex={len(cc_sents)}")

        print(f"[DOC {doc_id}] Relation extractor input simple sentences: {len(final_simple_sentences)}")



        # 7) Relationship extraction with Mixtral
        rel_rows = extract_relationships(mixtral_llm, final_simple_sentences, batch_size=args.batch_size)

        # 8) Flatten to CSV rows
        for sent_idx, rr in enumerate(rel_rows):
            sent = rr["sentence"]
            triples = rr.get("triples", []) or rr.get("triples", [])
            for t in triples:
                all_rows.append({
                    "doc_id": doc_id,
                    "simple_sentence_index": sent_idx,
                    "simple_sentence": sent,
                    "entity_1": t.get("Entity 1"),
                    "relationship": t.get("Relationship"),
                    "entity_2": t.get("Entity 2"),
                })

    out_df = pd.DataFrame(all_rows)
    out_df.to_csv(args.out_csv, index=False)
    print(f"Wrote CSV: {args.out_csv} | rows={len(out_df)}")

if __name__ == "__main__":
    main()
