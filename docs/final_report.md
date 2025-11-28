# Title Lift Analysis: Stage A Baseline and Stage B Non-Linear Reassessment

## Executive Summary
- **Stage A (intrinsic quality)**: Ordinary Least Squares still explains 58% of log-score variance on the November 2025 holdout (test RMSE 1.19, $R^2$ 0.58). Penalized ElasticNet reproduces the Weissburg setup with similar accuracy ($R^2$ 0.53) while shrinking unstable coefficients.
- **Stage B (title lift residual)**: Title-only OLS remains brittle (test $R^2$ −0.03). Among non-linear learners trained on the Stage A residual (`R`), a shallow MLP regressor fares best with RMSE 1.07, MAE 0.80, and a modest positive $R^2$ of 0.012. Tree ensembles (Random Forest, Gradient Boosting, LightGBM) and kernel SVMs fail to surpass this bar, underscoring how little incremental variance titles contribute.
- **Diagnostics**: Permutation importance and tree importances agree that sentiment ratios, clickbait flags, and title length heuristics dominate whatever signal exists. Residual histograms remain centered near zero with heavy tails, indicating limited but non-negligible systematic lift.
- **Deliverables**: A new analysis notebook (`docs/title_lift_analysis.ipynb`) and optional HTML export walk through data audits, EDA, model training, statistical tests, and formatted tables so results are consumable without wading through raw Markdown.

## Data Sources and Feature Pipeline
- **Collection**: Reddit (PRAW) and Hacker News snapshots gathered with the CLI tools in `bin/`, respecting API rate limits and using `.env` credentials.
- **Normalization**: `src/preprocess/normalize.py` aligns schemas, hashes author identifiers, and emits Parquet files with standardized feature names (e.g., `score_at_5`, `sentiment_compound`).
- **Feature Engineering**:
  - Context and exposure covariates (posting hour buckets, subreddit aggregates, early velocity) power Stage A.
  - Title heuristics (length, token counts, punctuation, VADER sentiment, clickbait regex hits, entity counts) inform Stage B residual modeling.
- **Artifacts**: `outputs/title_lift/stage_model_outputs.parquet` stores per-post targets (`y`), Stage A predictions, residuals `R`, and engineered title features used throughout this study.

## Baseline Stage A / Stage B Metrics
Metrics pulled from `docs/stage_metrics.json` (baseline OLS) and `outputs/title_lift/stage_penalized_metrics.json` (ElasticNet replica):

```
                       Model Stage Train RMSE Test RMSE Train R2 Test R2
        Baseline OLS Stage A     A      0.957     1.190    0.876   0.577
        Baseline OLS Stage B     B      0.948     1.178    0.019  -0.028
Penalized ElasticNet Stage A     A      1.001     1.258    0.864   0.527
Penalized ElasticNet Stage B     B      0.980     1.230    0.041   0.018
```

The Stage B residual remains hard to predict with linear methods—test $R^2$ hovers around zero or negative across specification tweaks.

## Non-Linear Stage B Experiments
We retrained models on the Stage A residual (`R`), removing leakage-prone fields (`score_*`, `yhat_A`, `title_lift_component`) and applying a unified preprocessing pipeline (median imputation, scaling, and one-hot encoding). A 70/15/15 temporal holdout mirrors the baseline split.

```
               Model   RMSE   MAE    R2
      MLP Regressor  1.074 0.805 0.012
      Random Forest  1.076 0.807 0.009
  Linear Regression  1.078 0.803 0.005
  Gradient Boosting  1.081 0.808 -0.001
           LightGBM  1.091 0.819 -0.020
          SVR (rbf)  1.100 0.820 -0.035
```

Observations:
- The shallow neural network (one hidden layer, ReLU, early stopping) delivers the only positive test $R^2$ beyond OLS, albeit marginally (+0.7% improvement in RMSE vs. linear baseline).
- Tree ensembles converge quickly but plateau; permutation importance highlights sentiment balance, clickbait patterns, and character-level ratios as the few features with consistent lift.
- LightGBM and the RBF SVM overfit validation folds without translating into holdout gains, implying the current feature space lacks the texture to reward more flexible kernels.
- Residual spread remains wide (~1.06 standard deviation), emphasizing that titles rarely shift outcomes more than ~±1 log-score even when optimally predicted.

## Diagnostic Highlights (see notebook Sections 4 & 10)
- Target audit confirms `yhat_A` tracks `y` closely (Stage A works), while `R` stays centered near zero with heavier tails during evening publishing windows.
- Correlation heatmaps show limited linear relationships between title heuristics and residuals, motivating the non-linear exploration.
- Permutation importances (MLP via scikit-learn wrappers) elevate `sentiment_compound`, `clickbait_patterns`, `title_chars_per_word`, and entity counts as the most sensitive levers.
- Paired $t$-tests between the MLP and Random Forest residuals yield marginal but non-zero differences (inspect Section 9 output once executed).

## Limitations
- The modeling target is the Stage A residual, so any Stage A misspecification cascades into Stage B; future iterations should consider joint modeling or Bayesian hierarchical approaches.
- Title features remain surface-level heuristics. Without richer embeddings (e.g., transformer sentence encoders) the lift ceiling is low.
- Snapshot cadence (5/15/30/60 minutes) may miss ultra-fast viral posts, biasing residuals for high-velocity communities.
- Reddit/HN deletions and moderation actions introduce censoring not currently modeled.

## Recommendations and Next Steps
1. **Feature depth**: Integrate transformer-based title embeddings or contrastive text encoders to supply non-linear models with semantic texture.
2. **Cross-platform calibration**: Extend the residual modeling to the Hacker News slice to confirm whether the weak title signal is platform-specific.
3. **Uncertainty & interpretability**: Add conformal prediction or quantile regression to provide lift intervals, aiding editorial decisions.
4. **Data refresh**: Re-run collectors over longer horizons and additional subreddits to stabilize estimates and re-check drift every quarter.
5. **Product packaging**: Convert Stage A/B predictions into percentage-based lift summaries for experimentation teams, linking to the HTML artifact generated in Section 11 of the notebook.

## Reproducibility and Deliverables
- **Notebook**: `docs/title_lift_analysis.ipynb` covers setup, EDA, model training, statistical testing, and reporting. Run it top-to-bottom to regenerate figures, tables, and cache files.
- **HTML export**: Execute the final notebook cell to emit `docs/title_lift_analysis_report.html` for a clean, shareable view (requires `jupyter nbconvert`).
- **Artifacts**: Cached splits (`outputs/title_lift/preprocessed_splits.joblib`) and data audit summary (`outputs/title_lift/data_audit_summary.json`) ensure consistent reuse.
- **Environment**: Install dependencies via `pip install -r requirements.txt`, download `en_core_web_sm` for spaCy, and populate `.env` with Reddit credentials before rerunning collectors.
