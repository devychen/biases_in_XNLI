# compute_length_controlled.py
# 控制长度的情况下，检验 Clean vs Problem 的差异
# 包含 sentence-level 和 pair-level 分析

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import math
import os
import statsmodels.api as sm
from scipy import stats

# -----------------------------
# 计算每个句子的平均交叉熵 (per-token loss) 和 PPL
# -----------------------------
def compute_sentence_loss(sentence, tokenizer, model, device="cpu"):
    model.eval()
    with torch.no_grad():
        encodings = tokenizer(sentence, return_tensors="pt")
        input_ids = encodings.input_ids.to(device)
        target_ids = input_ids.clone()

        outputs = model(input_ids, labels=target_ids)
        loss = outputs.loss.item()   # 平均 cross-entropy per token
        ppl = math.exp(loss)
    return loss, ppl, len(input_ids[0])

# -----------------------------
# 对 dataset 计算
# -----------------------------
def compute_dataset(df, tokenizer, model, device="cpu", name="dataset"):
    losses, ppls, tokens = [], [], []
    total = len(df)
    for idx, row in df.iterrows():
        loss, ppl, length = compute_sentence_loss(str(row["sentence"]), tokenizer, model, device)
        losses.append(loss)
        ppls.append(ppl)
        tokens.append(length)
        if (idx + 1) % 50 == 0 or (idx + 1) == total:
            print(f"{name}: processed {idx+1}/{total}")
    df["loss"] = losses
    df["ppl"] = ppls
    df["tokens"] = tokens
    return df

# -----------------------------
# 回归分析：loss ~ group + length
# -----------------------------
def regression_analysis(df, name="dataset"):
    df = df.copy()
    df["group_flag"] = df["group"].apply(lambda x: 1 if x == "Problem" else 0)

    X = pd.DataFrame({
        "group": df["group_flag"],
        "length": df["tokens"]
    })
    X = sm.add_constant(X)
    y = df["loss"]

    # 加入 HC3 稳健标准误
    model = sm.OLS(y, X).fit(cov_type="HC3")
    print(f"\n=== OLS Regression for {name} (HC3 robust SE) ===")
    print(model.summary())
    return model


# -----------------------------
# Pair-level 构建
# -----------------------------
def build_pairs(df):
    """
    将句子按原始顺序 (sentence1, sentence2) 成对
    假设数据是 extract_zh.py 得到的结构（两句一对）
    """
    pair_records = []
    for i in range(0, len(df), 2):
        if i+1 >= len(df):
            break
        s1 = df.iloc[i]
        s2 = df.iloc[i+1]
        avg_loss = (s1["loss"] + s2["loss"]) / 2
        avg_ppl = (s1["ppl"] + s2["ppl"]) / 2
        total_tokens = s1["tokens"] + s2["tokens"]
        pair_records.append({
            "loss": avg_loss,
            "ppl": avg_ppl,
            "tokens": total_tokens,
            "group": s1["group"]  # 两个句子属于同一 group
        })
    return pd.DataFrame(pair_records)

# -----------------------------
# Welch's t-test
# -----------------------------
def t_test_by_group(df, value_col="ppl", name="dataset"):
    clean_vals = df.loc[df["group"]=="Clean", value_col].dropna()
    problem_vals = df.loc[df["group"]=="Problem", value_col].dropna()
    t_stat, p_val = stats.ttest_ind(clean_vals, problem_vals, equal_var=False)
    print(f"\n=== Welch's t-test for {name} ({value_col}) ===")
    print(f"Clean mean={clean_vals.mean():.2f}, n={len(clean_vals)}")
    print(f"Problem mean={problem_vals.mean():.2f}, n={len(problem_vals)}")
    print(f"t={t_stat:.3f}, p={p_val:.5f}")


# -----------------------------
# 主函数
# -----------------------------
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained("uer/gpt2-chinese-cluecorpussmall")
    model = AutoModelForCausalLM.from_pretrained(
        "uer/gpt2-chinese-cluecorpussmall",
        use_safetensors=True
    ).to(device)

    os.makedirs("output", exist_ok=True)

    # 加载数据
    clean_df = pd.read_csv("data/xnli_zh_clean_dev.csv")
    problem_df = pd.read_csv("data/xnli_zh_problem_dev.csv")

    clean_df["group"] = "Clean"
    problem_df["group"] = "Problem"

    all_df = pd.concat([clean_df, problem_df], ignore_index=True)

    # -------- Sentence-level --------
    all_df = compute_dataset(all_df, tokenizer, model, device, "All sentences")
    all_df.to_csv("output/all_with_loss.csv", index=False)

    print("\n=== Sentence-level descriptive statistics ===")
    print(all_df.groupby("group")[["loss", "ppl", "tokens"]].mean())

    t_test_by_group(all_df, "ppl", "Sentence-level (ppl)")
    t_test_by_group(all_df, "loss", "Sentence-level (loss)")

    regression_analysis(all_df, "Sentence-level")

    # -------- Pair-level --------
    pair_df = build_pairs(all_df)
    pair_df.to_csv("output/all_pairs_with_loss.csv", index=False)

    print("\n=== Pair-level descriptive statistics ===")
    print(pair_df.groupby("group")[["loss", "ppl", "tokens"]].mean())

    t_test_by_group(pair_df, "ppl", "Pair-level (ppl)")
    t_test_by_group(pair_df, "loss", "Pair-level (loss)")

    regression_analysis(pair_df, "Pair-level")


if __name__ == "__main__":
    main()
