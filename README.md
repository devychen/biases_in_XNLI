# A Perplexity-Based Analysis of Translation Artefacts in XNLI: Measuring Bias from Untranslated Tokens

# Proposal

The study examines whether untranslated English terms (there’s a lot in the corpus) and name entities (e.g. “thanksgiving day”) in the XNLI’s non-English splits introduce biases when evaluating multi-lingual models, particularly for low-resource languages - and I choose Chinese for this study as a native speaker. And the hypothesis is that such terms increase sentence perplexity in targeted language (Chinese), and thus might artificially inflating model difficulty.

### The method:

#### Data: 
Extract Chinese XNLI dev sentences:
- **Problem group**: with untranslated English tokens and name entity recognition tokens (problem group) 
- **Clean group**: fully translated sentences (no Latin alphabet and no NER-matched English terms).

#### Metric: 
Mean PPL per group (Chinese LM), Welch’s t-test for significance.

#### Expected results: 
Higher PPL in experimental group would suggest untranslated terms degrade data quality. There will be a table clearly shows the mean PPL and p-value between two groups.

# Steps Overview

### Stage 1 - Prepare the Data

从XNLI中文开发集中提取数据，并创建两个组（“问题组”和“清洁组”）。

> `extract_zh.py`, then `ultra_clean.py` 

### Step 2 - Compute PPL

使用中文Causal LM为所有句子计算困惑度。

Use [google-bert/bert-base-chinese](https://huggingface.co/google-bert/bert-base-chinese?library=transformers), but too slow. Use a smaller one: [hfl/chinese-macbert-base](https://huggingface.co/hfl/chinese-macbert-base).

No BERT for this task. Should use causal LM, otherwise pseudo-PPL. Model used [gpt2-chinese-cluecorpussmall](https://huggingface.co/uer/gpt2-chinese-cluecorpussmall).

> `compute.py`

### Step 3 - Analysis

使用Welch's t检验比较两组的平均困惑度，判断差异是否显著。

### Step 4 - Results

制作表格展示平均PPL和p值，并解释结果。

```
All sentences: processed 4950/4980
All sentences: processed 4980/4980

=== Sentence-level descriptive statistics ===
             loss        ppl     tokens
group                                  
Clean    3.656271  49.516379  22.355804
Problem  3.735215  54.092297  30.044258

=== Welch's t-test for Sentence-level (ppl) (ppl) ===
Clean mean=49.52, n=3308
Problem mean=54.09, n=1672
t=-2.951, p=0.00319

=== Welch's t-test for Sentence-level (loss) (loss) ===
Clean mean=3.66, n=3308
Problem mean=3.74, n=1672
t=-4.024, p=0.00006

=== OLS Regression for Sentence-level (HC3 robust SE) ===
                            OLS Regression Results                            
==============================================================================
Dep. Variable:                   loss   R-squared:                       0.153
Model:                            OLS   Adj. R-squared:                  0.152
Method:                 Least Squares   F-statistic:                     481.3
Date:                Thu, 28 Aug 2025   Prob (F-statistic):          8.34e-192
Time:                        22:33:35   Log-Likelihood:                -4507.6
No. Observations:                4980   AIC:                             9021.
Df Residuals:                    4977   BIC:                             9041.
Df Model:                           2                                         
Covariance Type:                  HC3                                         
==============================================================================
                 coef    std err          z      P>|z|      [0.025      0.975]
------------------------------------------------------------------------------
const          4.0978      0.020    208.628      0.000       4.059       4.136
group          0.2308      0.019     12.334      0.000       0.194       0.267
length        -0.0198      0.001    -30.825      0.000      -0.021      -0.018
==============================================================================
Omnibus:                      229.504   Durbin-Watson:                   1.500
Prob(Omnibus):                  0.000   Jarque-Bera (JB):              282.732
Skew:                           0.489   Prob(JB):                     4.03e-62
Kurtosis:                       3.638   Cond. No.                         64.3
==============================================================================

Notes:
[1] Standard Errors are heteroscedasticity robust (HC3)

=== Pair-level descriptive statistics ===
             loss        ppl     tokens
group                                  
Clean    3.656271  49.516379  44.711608
Problem  3.735215  54.092297  60.088517

=== Welch's t-test for Pair-level (ppl) (ppl) ===
Clean mean=49.52, n=1654
Problem mean=54.09, n=836
t=-2.690, p=0.00722

=== Welch's t-test for Pair-level (loss) (loss) ===
Clean mean=3.66, n=1654
Problem mean=3.74, n=836
t=-3.566, p=0.00037

=== OLS Regression for Pair-level (HC3 robust SE) ===
                            OLS Regression Results                            
==============================================================================
Dep. Variable:                   loss   R-squared:                       0.157
Model:                            OLS   Adj. R-squared:                  0.156
Method:                 Least Squares   F-statistic:                     190.1
Date:                Thu, 28 Aug 2025   Prob (F-statistic):           1.44e-77
Time:                        22:33:36   Log-Likelihood:                -1669.8
No. Observations:                2490   AIC:                             3346.
Df Residuals:                    2487   BIC:                             3363.
Df Model:                           2                                         
Covariance Type:                  HC3                                         
==============================================================================
                 coef    std err          z      P>|z|      [0.025      0.975]
------------------------------------------------------------------------------
const          4.1598      0.031    135.775      0.000       4.100       4.220
group          0.2521      0.022     11.516      0.000       0.209       0.295
length        -0.0113      0.001    -18.834      0.000      -0.012      -0.010
==============================================================================
Omnibus:                       92.858   Durbin-Watson:                   1.111
Prob(Omnibus):                  0.000   Jarque-Bera (JB):              110.204
Skew:                           0.432   Prob(JB):                     1.17e-24
Kurtosis:                       3.562   Cond. No.                         151.
==============================================================================

Notes:
[1] Standard Errors are heteroscedasticity robust (HC3)
```

# Summary

### 1. 实验设计说明
#### 研究目标
- 研究 XNLI 中文数据集中的翻译 artefacts（未翻译的英文 token 和特定实体，如 “thanksgiving day”）是否会 人为提高句子的语言模型困惑度 (perplexity, PPL)。
- 假设：Problem 组（包含未翻译词或命名实体）比 Clean 组的句子更难预测，导致 PPL 增高。

#### 数据
- 数据来源：XNLI 中文开发集 (Chinese XNLI dev set)
- 分组：
    - Problem group：含未翻译英文词或命名实体的句子
    - Clean group：完全中文翻译句子，无英文 token
- 样本数：
    - Sentence-level：Clean = 3308，Problem = 1672
    - Pair-level（两句合并）：Clean = 1654，Problem = 836

### 2. 方法
- 语言模型
    - 使用 GPT2 中文模型：`uer/gpt2-chinese-cluecorpussmall`
    - 指标：
        - 每句交叉熵平均 loss：平均每个 token 的预测困惑
        - Perplexity (PPL)：`exp(loss)`，衡量模型对句子的预测难度
- 分析流程
1. Sentence-level
    - 计算每句 loss 和 PPL
    - 描述统计：各组均值、token 数
    - Welch’s t-test：检验 Clean vs Problem 的平均 PPL / loss 差异
    - 回归分析（控制句子长度）：`loss ~ group + length`，使用 HC3 稳健标准误
2. Pair-level
    - 将相邻两句合并为 pair，平均 loss/PPL，总 token 数求和
    - 描述统计 + Welch’s t-test + 回归分析（控制长度）
    - 回归模型形式：`avg_loss ~ group + total_length`

### 3. Results

#### Sentence level

| 指标     | Clean | Problem |
| ------ | ----- | ------- |
| loss   | 3.656 | 3.735   |
| PPL    | 49.52 | 54.09   |
| tokens | 22.36 | 30.04   |

Welch’s t-test:
- PPL: t=-2.951, p=0.00319 → Problem 明显高
- Loss: t=-4.024, p=0.00006 → Problem 明显高

回归（控制长度）：
- `group` 系数 = 0.231, p<0.001 → Problem 组 loss 平均高 0.231，长度控制后仍显著
- `length` 系数 = -0.020, p<0.001 → token 越多，单 token loss 略下降（模型自适应？）

#### Pair level

| 指标     | Clean | Problem |
| ------ | ----- | ------- |
| loss   | 3.656 | 3.735   |
| PPL    | 49.52 | 54.09   |
| tokens | 44.71 | 60.09   |

Welch’s t-test:
- PPL: t=-2.690, p=0.00722 → Problem 明显高
- Loss: t=-3.566, p=0.00037 → Problem 明显高

回归（控制长度）：
- `group` 系数 = 0.252, p<0.001 → Problem 组 pair-level loss 平均高 0.252
- `length` 系数 = -0.011, p<0.001 → token 越多，loss 略下降

#### 解读
- 无论是否控制句子长度，Problem 组的 PPL / loss 都明显高于 Clean 组 → 支持假设：未翻译词或命名实体 人为提高了中文句子的语言模型困惑度。
- 回归控制长度后，`group` 系数仍显著 → 差异不是由句子长度造成的，而是真正由翻译 artefacts 引起的。
- Pair-level 结果与 sentence-level 一致 → 说明这种现象在多句组合时仍成立。