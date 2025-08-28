# compute.py
# compute sentence-level and pair-level PPL

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from scipy import stats
import math
import os

def standard_ppl(sentence, tokenizer, model, device="cpu"):
    model.eval()
    with torch.no_grad():
        encodings = tokenizer(sentence, return_tensors="pt")
        input_ids = encodings.input_ids.to(device)
        target_ids = input_ids.clone()
        outputs = model(input_ids, labels=target_ids)
        loss = outputs.loss.item()
        ppl = math.exp(loss)
    return ppl, len(input_ids[0])

def compute_dataset_ppl(df, tokenizer, model, device="cpu", name="dataset"):
    ppls, tokens = [], []
    for idx, row in df.iterrows():
        ppl, n_tok = standard_ppl(str(row["sentence"]), tokenizer, model, device)
        ppls.append(ppl); tokens.append(n_tok)
        if (idx+1) % 50 == 0 or (idx+1) == len(df):
            print(f"{name}: processed {idx+1}/{len(df)}")
    df["ppl"] = ppls
    df["tokens"] = tokens
    return df

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained("uer/gpt2-chinese-cluecorpussmall")
    model = AutoModelForCausalLM.from_pretrained(
        "uer/gpt2-chinese-cluecorpussmall",
        use_safetensors=True
    ).to(device)

    os.makedirs("output", exist_ok=True)

    clean_df = pd.read_csv("data/xnli_zh_clean_dev.csv")
    problem_df = pd.read_csv("data/xnli_zh_problem_dev.csv")

    print("Computing PPL for clean dataset...")
    clean_df = compute_dataset_ppl(clean_df, tokenizer, model, device, "Clean dataset")

    print("Computing PPL for problem dataset...")
    problem_df = compute_dataset_ppl(problem_df, tokenizer, model, device, "Problem dataset")

    # ===== sentence-level统计 =====
    clean_mean = clean_df["ppl"].mean()
    problem_mean = problem_df["ppl"].mean()
    t_stat, p_value = stats.ttest_ind(problem_df["ppl"], clean_df["ppl"], equal_var=False)

    print("=== Sentence-level Results ===")
    print(f"Clean mean PPL={clean_mean:.2f}, n={len(clean_df)}")
    print(f"Problem mean PPL={problem_mean:.2f}, n={len(problem_df)}")
    print(f"Welch's t-test: t={t_stat:.3f}, p={p_value:.5f}")

    # ===== pair-level聚合统计 =====
    def agg_pair(df):
        return df.groupby("pairID")["ppl"].mean().reset_index()

    clean_pair = agg_pair(clean_df)
    problem_pair = agg_pair(problem_df)

    clean_pair_mean = clean_pair["ppl"].mean()
    problem_pair_mean = problem_pair["ppl"].mean()
    t_stat2, p_value2 = stats.ttest_ind(problem_pair["ppl"], clean_pair["ppl"], equal_var=False)

    print("=== Pair-level Results ===")
    print(f"Clean mean PPL={clean_pair_mean:.2f}, pairs={len(clean_pair)}")
    print(f"Problem mean PPL={problem_pair_mean:.2f}, pairs={len(problem_pair)}")
    print(f"Welch's t-test: t={t_stat2:.3f}, p={p_value2:.5f}")

    # 保存
    clean_df.to_csv("output/clean_ppl.csv", index=False)
    problem_df.to_csv("output/problem_ppl.csv", index=False)
    clean_pair.to_csv("output/clean_pair_ppl.csv", index=False)
    problem_pair.to_csv("output/problem_pair_ppl.csv", index=False)

if __name__ == "__main__":
    main()
