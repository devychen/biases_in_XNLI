import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from scipy import stats
import matplotlib.pyplot as plt
import math
import os

# -----------------------------
# 计算标准 perplexity
# -----------------------------
def standard_ppl(sentence, tokenizer, model, device="cpu"):
    model.eval()
    with torch.no_grad():
        encodings = tokenizer(sentence, return_tensors="pt")
        input_ids = encodings.input_ids.to(device)
        target_ids = input_ids.clone()
        # 前向计算 log-likelihood
        outputs = model(input_ids, labels=target_ids)
        # outputs.loss 是平均负对数似然 (cross-entropy)
        loss = outputs.loss.item()
        ppl = math.exp(loss)
    return ppl

# -----------------------------
# 计算 dataset PPL 并打印进度
# -----------------------------
def compute_dataset_ppl(df, tokenizer, model, device="cpu", name="dataset"):
    ppls = []
    total = len(df)
    for idx, row in df.iterrows():
        ppl1 = standard_ppl(str(row["sentence1"]), tokenizer, model, device)
        ppl2 = standard_ppl(str(row["sentence2"]), tokenizer, model, device)
        avg_ppl = (ppl1 + ppl2) / 2
        ppls.append(avg_ppl)
        if (idx + 1) % 50 == 0 or (idx + 1) == total:
            print(f"{name}: processed {idx+1}/{total} sentences")
    df["ppl"] = ppls
    return df, ppls

# -----------------------------
# 主函数
# -----------------------------
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained("uer/gpt2-chinese-cluecorpussmall")
    model = AutoModelForCausalLM.from_pretrained("uer/gpt2-chinese-cluecorpussmall",
                                                 use_safetensors=True).to(device)

    # 确保 output 文件夹存在
    os.makedirs("output", exist_ok=True)

    clean_df = pd.read_csv("data/xnli_zh_clean_dev.csv")
    problem_df = pd.read_csv("data/xnli_zh_problem_dev.csv")

    print("计算 clean dataset PPL...")
    clean_df, clean_ppl = compute_dataset_ppl(clean_df, tokenizer, model, device, "Clean dataset")

    print("计算 problem dataset PPL...")
    problem_df, problem_ppl = compute_dataset_ppl(problem_df, tokenizer, model, device, "Problem dataset")

    # 文字结果
    print("=== 结果 ===")
    print(f"Clean dataset: mean={sum(clean_ppl)/len(clean_ppl):.2f}, std={pd.Series(clean_ppl).std():.2f}, n={len(clean_ppl)}")
    print(f"Problem dataset: mean={sum(problem_ppl)/len(problem_ppl):.2f}, std={pd.Series(problem_ppl).std():.2f}, n={len(problem_ppl)}")

    t_stat, p_value = stats.ttest_ind(problem_ppl, clean_ppl, equal_var=False)
    print(f"Welch's t-test: t={t_stat:.3f}, p={p_value:.5f}")

    # 保存 CSV
    clean_df.to_csv("output/clean_ppl.csv", index=False)
    problem_df.to_csv("output/problem_ppl.csv", index=False)

    # # 可视化
    # plt.figure(figsize=(8,5))
    # plt.hist(clean_ppl, bins=50, alpha=0.7, label="Clean dataset")
    # plt.hist(problem_ppl, bins=50, alpha=0.7, label="Problem dataset")
    # plt.xlabel("Perplexity")
    # plt.ylabel("Frequency")
    # plt.title("PPL Distribution")
    # plt.legend()
    # plt.tight_layout()
    # plt.savefig("output/ppl_distribution.png")
    # plt.show()

if __name__ == "__main__":
    main()



# # Pseudo PPL

# import pandas as pd
# import torch
# from transformers import AutoTokenizer, AutoModelForMaskedLM
# from scipy import stats
# import math

# # -----------------------------
# # 计算 pseudo-perplexity
# # -----------------------------
# def pseudo_ppl(sentence, tokenizer, model, device="cpu"):
#     model.eval()
#     tokens = tokenizer.tokenize(sentence)
#     if len(tokens) == 0:
#         return None

#     input_ids = tokenizer.encode(sentence, return_tensors="pt").to(device)
#     nll = 0.0
#     with torch.no_grad():
#         for i in range(1, input_ids.size(1)-1):
#             mask_input = input_ids.clone()
#             mask_input[0, i] = tokenizer.mask_token_id
#             outputs = model(mask_input)
#             logits = outputs.logits
#             softmax = torch.nn.functional.log_softmax(logits[0, i], dim=-1)
#             token_id = input_ids[0, i]
#             nll += -softmax[token_id].item()
#     ppl = math.exp(nll / (input_ids.size(1)-2))
#     return ppl

# # -----------------------------
# # 计算 dataset PPL 并打印进度
# # -----------------------------
# def compute_dataset_ppl(df, tokenizer, model, device="cpu", name="dataset"):
#     ppls = []
#     total = len(df)
#     for idx, row in df.iterrows():
#         ppl1 = pseudo_ppl(str(row["sentence1"]), tokenizer, model, device)
#         ppl2 = pseudo_ppl(str(row["sentence2"]), tokenizer, model, device)
#         if ppl1 is not None and ppl2 is not None:
#             avg_ppl = (ppl1 + ppl2) / 2
#         else:
#             avg_ppl = None
#         ppls.append(avg_ppl)
#         if (idx + 1) % 50 == 0 or (idx + 1) == total:
#             print(f"{name}: processed {idx+1}/{total} sentences")
#     df["ppl"] = ppls
#     return df, [x for x in ppls if x is not None]

# # -----------------------------
# # 主函数
# # -----------------------------
# def main():
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     tokenizer = AutoTokenizer.from_pretrained("hw2942/chinese-macbert-base-climate-related-prediction-vv5")
#     model = AutoModelForMaskedLM.from_pretrained(
#         "hw2942/chinese-macbert-base-climate-related-prediction-vv5",
#         use_safetensors=True
#     )
#     model.to(device)

#     clean_df = pd.read_csv("data/xnli_zh_clean_dev.csv")
#     problem_df = pd.read_csv("data/xnli_zh_problem_dev.csv")

#     print("计算 clean dataset PPL...")
#     clean_df, clean_ppl = compute_dataset_ppl(clean_df, tokenizer, model, device, "Clean dataset")

#     print("计算 problem dataset PPL...")
#     problem_df, problem_ppl = compute_dataset_ppl(problem_df, tokenizer, model, device, "Problem dataset")

#     print("=== 结果 ===")
#     print(f"Clean dataset: mean={sum(clean_ppl)/len(clean_ppl):.2f}, std={pd.Series(clean_ppl).std():.2f}, n={len(clean_ppl)}")
#     print(f"Problem dataset: mean={sum(problem_ppl)/len(problem_ppl):.2f}, std={pd.Series(problem_ppl).std():.2f}, n={len(problem_ppl)}")

#     t_stat, p_value = stats.ttest_ind(problem_ppl, clean_ppl, equal_var=False)
#     print(f"Welch's t-test: t={t_stat:.3f}, p={p_value:.5f}")

#     clean_df.to_csv("output/xnli_zh_clean_dev_with_ppl.csv", index=False)
#     problem_df.to_csv("output/xnli_zh_problem_dev_with_ppl.csv", index=False)

# if __name__ == "__main__":
#     main()
