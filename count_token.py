import pandas as pd
from transformers import AutoTokenizer

# 选择一个 tokenizer，例如 GPT2 Chinese
tokenizer = AutoTokenizer.from_pretrained("uer/gpt2-chinese-cluecorpussmall")

def add_token_counts(df):
    # 计算每条句子的 token 数
    df["sentence1_tokens"] = df["sentence1"].apply(lambda x: len(tokenizer.encode(str(x), add_special_tokens=False)))
    df["sentence2_tokens"] = df["sentence2"].apply(lambda x: len(tokenizer.encode(str(x), add_special_tokens=False)))
    # 可以算平均 token 数
    df["avg_tokens"] = (df["sentence1_tokens"] + df["sentence2_tokens"]) / 2
    return df

# 读取数据
clean_df = pd.read_csv("data/xnli_zh_clean_dev.csv")
problem_df = pd.read_csv("data/xnli_zh_problem_dev.csv")

# 计算 token 数
clean_df = add_token_counts(clean_df)
problem_df = add_token_counts(problem_df)

# # 保存结果
# clean_df.to_csv("output/xnli_zh_clean_dev_with_tokens.csv", index=False)
# problem_df.to_csv("output/xnli_zh_problem_dev_with_tokens.csv", index=False)

# 打印简单统计
print("Clean dataset token counts:")
print(clean_df[["sentence1_tokens", "sentence2_tokens", "avg_tokens"]].describe())

print("\nProblem dataset token counts:")
print(problem_df[["sentence1_tokens", "sentence2_tokens", "avg_tokens"]].describe())
