import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 读入两个 dataset
clean_df = pd.read_csv("output/clean_ppl.csv")
problem_df = pd.read_csv("output/problem_ppl.csv")

# 提取 ppl 列
clean_ppl = clean_df["ppl"].values
problem_ppl = problem_df["ppl"].values

# 画图
plt.figure(figsize=(10,6))
sns.kdeplot(clean_ppl, label="Clean", fill=True, alpha=0.4)
sns.kdeplot(problem_ppl, label="Problem", fill=True, alpha=0.4)

plt.xlabel("Perplexity")
plt.ylabel("Density")
plt.title("PPL Distribution: Clean vs Problem")
plt.legend()
plt.tight_layout()
# plt.show()

# save

plt.savefig("output/ppl_distribution.png", dpi=300)
plt.close()