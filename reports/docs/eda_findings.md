### EDA Summary and Implications

The exploratory data analysis reveals a moderately sized dataset (37,249 comments) focused on Indian political discourse, evident from recurring themes like "Modi," "BJP," "Congress," and "India" in word clouds and samples. This domain specificity may introduce bias for YouTube generalization, where sentiments could skew toward entertainment or global topics. Key findings are structured below for clarity.

#### Dataset Quality and Structure
- **Shape and Columns**: 37,249 rows × 4 columns (`clean_comment`, `category`, `sentiment_label`, `word_count`). The addition of `sentiment_label` (categorical mapping: Negative/Neutral/Positive) and `word_count` enhances usability.
- **Missing Values**: Minimal (0.27% in `clean_comment`; 100 rows). Recommendation: Drop or impute these during preparation to avoid propagation errors.
- **Text Characteristics**:
  - Character lengths: Right-skewed (median 80, mean 181, max 8,665), indicating outliers from long rants. Neutral comments are shortest (median 39 chars), suggesting brevity in factual responses.
  - Word counts: Similar skew; medians by category (inferred from text_length trends): Negative ~20 words, Neutral ~7, Positive ~22. This implies negatives may convey frustration concisely, positives elaborately.

#### Sentiment Distribution
| Category | Count | Proportion (%) |
|----------|-------|----------------|
| Negative (-1) | 8,277 | 22.22 |
| Neutral (0) | 13,142 | 35.28 |
| Positive (1) | 15,830 | 42.50 |

- **Imbalance**: Positives dominate, risking model bias toward optimism. Address via undersampling positives or oversampling negatives in training.
- **Visual Insights**:
  - Histogram: Sharp peak near 0-200 chars, tailing off—truncate extremes (>1,000 chars) to reduce noise.
  - Boxplot: Negatives and positives have longer medians than neutrals, with similar variance; outliers in positives suggest enthusiastic endorsements.
  - KDE Plot: Overlap across categories, but negatives peak earliest—feature engineering (e.g., length ratios) could differentiate.

#### Linguistic Patterns
- **Top Words (Overall/Per Sentiment)**: Dominated by function words ("the," "and"), but sentiment signals emerge:
  - Positive: "good," "people," "way," "think" (affirmative, communal).
  - Neutral: "the," "this," "you" (factual, interrogative).
  - Negative: "the," "that," "this" (with "said," "think," "government"—critical tone).
- **Word Clouds**: Reinforce politics—positives highlight "vote," "leader"; negatives "shit," "fuck," "problem"; neutrals "question," "time." Innovation opportunity: Incorporate domain adaptation (e.g., fine-tune on YouTube politics subset) for transfer learning.

#### Practical Implications for Pipeline
- **Reliability**: High variance in lengths warrants robust preprocessing (e.g., padding for RNNs).
- **Scalability**: Skewed distributions suit batch processing; monitor for YouTube data drift (e.g., shorter, emoji-heavy comments).
- **Maintainability**: Version EDA outputs (e.g., save plots to `reports/figures/`) via DVC.
- **Adaptability**: Proxy works for baselines, but collect 1,000+ YouTube samples early to evaluate domain shift (F1 drop expected ~10-15%).

This analysis confirms viability for sentiment modeling while highlighting political bias—prioritize YouTube-specific augmentation for real-world deployment.

### Next Steps Recommendations
1. **Data Preparation**: Implement `src/data/make_dataset.py` for cleaning/splitting; add YouTube fetcher.
2. **Feature Engineering**: Prototype TF-IDF/BERT in `src/features/feature_engineering.py`.
3. **Baseline Modeling**: Train logistic regression in `src/models/train_model.py`; track with MLflow.
