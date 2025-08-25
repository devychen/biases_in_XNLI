# A Data Quality Analysis of XNLI: Untranslated Tokens as a Source of Bias in Chinese

# A Perplexity-Based Analysis of Translation Artefacts in XNLI

### Proposed Study: 

The study examines whether untranslated English terms (there’s a lot in the corpus) and name entities (e.g. “thanksgiving day”) in the XNLI’s non-English splits introduce biases when evaluating multi-lingual models, particularly for low-resource languages - and I choose Chinese for this study as a native speaker. And the hypothesis is that such terms increase sentence perplexity in targeted language (Chinese), and thus might artificially inflating model difficulty.

### The method:

#### Data: 
Extract Chinese XNLI dev sentences with untranslated English tokens and name entity recognition tokens (problem group) vs. fully translated sentences (clean group, no Latin alphabet and no NER-matched English terms).

#### Metric: 
Mean PPL per group (Chinese BERT), Welch’s t-test for significance.

#### Expected results: 
Higher PPL in experimental group would suggest untranslated terms degrade data quality. There will be a table clearly shows the mean PPL and p-value between two groups.

# Steps Overview

### Stage 1 - Prepare the Data

从XNLI中文开发集中提取数据，并创建两个组（“问题组”和“清洁组”）。

### Step 2 - Compute PPL

使用中文BERT模型为所有句子计算困惑度。

### Step 3 - Analysis

使用Welch's t检验比较两组的平均困惑度，判断差异是否显著。

### Step 4 - Results

制作表格展示平均PPL和p值，并解释结果。