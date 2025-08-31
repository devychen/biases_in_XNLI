# A Perplexity-Based Analysis of Translation Artefacts in XNLI: Measuring Bias from Untranslated Tokens

# Proposal

The study examines whether untranslated English terms (there’s a lot in the corpus) and name entities (e.g. “thanksgiving day”) in the XNLI’s non-English splits introduce biases when evaluating multi-lingual models, particularly for low-resource languages - and I choose Chinese for this study as a native speaker. And the hypothesis is that such terms increase sentence perplexity in targeted language (Chinese), and thus might artificially inflating model difficulty.

### The method:

#### Data: 
Extract Chinese XNLI dev sentences:
- **Problem group**: with untranslated English tokens and name entity recognition tokens (problem group) 
- **Clean group**: fully translated sentences (no Latin alphabet and no NER-matched English terms).

Available at folder `data > xnli_zh_clean_dev.csv & xnli_zh_problem_dev.csv`

#### Metric: 
Mean PPL per group (Chinese LM), Welch’s t-test for significance.  
OLS regession. 

#### Expected results: 
Higher PPL in experimental group would suggest untranslated terms degrade data quality. There will be a table clearly shows the mean PPL and p-value between two groups.

