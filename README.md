# A Data Quality Analysis of XNLI: Untranslated Tokens as a Source of Bias in Chinese

### Design

I want to examine whether untranslated English terms (there’s a lot in the corpus) and name entities (e.g. “thanksgiving day”) in the XNLI’s non-English splits introduce biases when evaluating multi-lingual models, particularly for low-resource languages - and I choose Chinese for this study as a native speaker. And the hypothesis is that such terms increase sentence perplexity in targeted language (Chinese), and thus might artificially inflating model difficulty.

The method:

Data: Extract Chinese XNLI dev sentences with untranslated English tokens and name entity recognition tokens problem group) vs. fully translated sentences (clean group, no Latin alphabet and no NER-matched English terms).
Metric: Mean PPL per group (Chinese BERT), Welch’s t-test for significance.
Expected results: Higher PPL in experimental group would suggest untranslated terms degrade data quality. There will be a table clearly shows the mean PPL and p-value between two groups.
