import pandas as pd

def main():
    # 读取 jsonl 格式的 dev 数据
    df = pd.read_json("XNLI/xnli.dev.jsonl", lines=True)

    # 筛选中文
    zh_df = df[df["language"] == "zh"]

    # 只保留关键字段
    zh_df = zh_df[["gold_label", "sentence1", "sentence2", "promptID"]]

    # 保存为新的文件
    zh_df.to_csv("data/xnli_zh_dev.csv", index=False, encoding="utf-8")

    print(f"提取完成 ✅ 一共 {len(zh_df)} 条中文数据，已保存到 xnli_zh_dev.csv")

if __name__ == "__main__":
    main()
