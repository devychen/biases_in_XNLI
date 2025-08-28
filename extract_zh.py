# extract_zh.py
# extract Chinese sentences with pairID and role info

import pandas as pd

def main():
    df = pd.read_json("XNLI/xnli.dev.jsonl", lines=True)
    zh_df = df[df["language"] == "zh"]

    records = []
    for _, row in zh_df.iterrows():
        records.append({"pairID": row["pairID"], "role": "premise", "sentence": row["sentence1"]})
        records.append({"pairID": row["pairID"], "role": "hypothesis", "sentence": row["sentence2"]})

    sentences_df = pd.DataFrame(records)
    sentences_df.to_csv("data/xnli_zh_sentences.csv", index=False, encoding="utf-8")

    print(f"提取完成 ✅ 一共 {len(sentences_df)} 条中文句子，已保存到 xnli_zh_sentences.csv")

if __name__ == "__main__":
    main()
