# clean.py
# classify sentences into clean vs problem groups, keeping pairID/role

import pandas as pd
import re
import yaml
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("uer/gpt2-chinese-cluecorpussmall")

def load_dict(yaml_path="dict.yaml"):
    with open(yaml_path, "r", encoding="utf-8") as f:
        d = yaml.safe_load(f)
    terms = []
    for _, words in d.items():
        if words:
            terms.extend(words)
    return d, terms

def contains_terms(text: str, terms: list) -> bool:
    return any(term in str(text) for term in terms)

def contains_latin(text: str) -> bool:
    return bool(re.search(r"[A-Za-z]", str(text)))

def tokens_per_sentence(df, tokenizer):
    return [len(tokenizer.encode(sent, add_special_tokens=False)) for sent in df["sentence"]]

def main():
    df = pd.read_csv("data/xnli_zh_sentences.csv")

    # 标记
    df["has_latin"] = df["sentence"].apply(contains_latin)

    category_dict, all_terms = load_dict("dict.yaml")
    df["has_foreign_term"] = df["sentence"].apply(lambda s: contains_terms(s, all_terms))

    # 分组
    clean_df = df[(df["has_latin"] == False) & (df["has_foreign_term"] == False)]
    problem_df = df[(df["has_latin"]) | (df["has_foreign_term"])]

    clean_df.to_csv("data/xnli_zh_clean_dev.csv", index=False, encoding="utf-8")
    problem_df.to_csv("data/xnli_zh_problem_dev.csv", index=False, encoding="utf-8")

    # 统计
    clean_tokens = tokens_per_sentence(clean_df, tokenizer)
    problem_tokens = tokens_per_sentence(problem_df, tokenizer)

    print(f"总数据: {len(df)} 条")
    print(f"清洁组: {len(clean_df)} 条, 平均 token={sum(clean_tokens)/len(clean_tokens) if clean_tokens else 0:.2f}")
    print(f"问题组: {len(problem_df)} 条, 平均 token={sum(problem_tokens)/len(problem_tokens) if problem_tokens else 0:.2f}")

if __name__ == "__main__":
    main()
