import pandas as pd
import re

def contains_english(text):
    """检测句子中是否包含英文字符"""
    return bool(re.search(r'[A-Za-z]', text))

def main():
    # 读取上一步生成的中文数据
    df = pd.read_csv("data/xnli_zh_dev.csv")

    # 创建新列标记是否含有英文
    df["has_english"] = df.apply(
        lambda row: contains_english(row["sentence1"]) or contains_english(row["sentence2"]), 
        axis=1
    )

    # Problem group
    problem_group = df[df["has_english"] == True]
    problem_group.to_csv("data/xnli_zh_dev_problem.csv", index=False, encoding="utf-8")

    # Clean group
    clean_group = df[df["has_english"] == False]
    clean_group.to_csv("data/xnli_zh_dev_clean.csv", index=False, encoding="utf-8")

    print(f"Problem group: {len(problem_group)} 条")
    print(f"Clean group: {len(clean_group)} 条")

if __name__ == "__main__":
    main()
