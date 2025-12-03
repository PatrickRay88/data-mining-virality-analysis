# Title Lift Analysis: Intrinsic Quality vs. Title Effects

## Slide 1 – Title & Context
- **Title:** Separating Intrinsic Quality from Title Lift Across Reddit & Hacker News
- **Presenters:** Patrick Ray, Data Mining Practicum
- **Date:** December 3, 2025
- **Purpose:** Demonstrate the end-to-end pipeline and evidence supporting the proposal claim that title signals provide incremental lift beyond intrinsic quality.

## Slide 2 – Agenda
1. Recap of research question & prior work
2. Data pipeline & feature engineering
3. Stage A (intrinsic quality) model performance
4. Stage B (title) model variants & lift
5. Robustness checks & diagnostics
6. Key takeaways, recommendations, and next steps

## Slide 3 – Motivation & Prior Work
- Original ICWSM 2022 study quantified title effects but did not jointly model exposure and wording across platforms.
- Proposal objective: build a two-stage pipeline that isolates intrinsic quality, then measures incremental title lift.
- Research question: *How much of post virality can be attributed to title wording once platform exposure dynamics are controlled?*
- Target platforms: Reddit (9 topical subreddits) + Hacker News top stories; time horizon: Oct–Nov 2025.

## Slide 4 – Data Sources & Volume
- Reddit: 98k posts collected via `bin/collect_reddit.py` (technology, politics, worldnews, etc.) with 5/15/30/60 minute snapshots.
- Hacker News: 12k stories via `bin/collect_hn.py` with analogous snapshots.
- Normalization (`src/preprocess/normalize.py`) harmonizes schema, hashes authors.
- Final merged dataset: 13395 posts after de-duplication and quality filters.
- **Figure:** Pipeline flowchart (`docs/figures/flowchart/title_lift_pipeline_flow.html`).

## Slide 5 – Feature Engineering Highlights
- Early growth features: snapshot deltas at 5/15/30/60 minutes.
- Context features: subreddit/topic, posting hour, author history proxies.
- Title features (`src/preprocess/features_titles.py`): sentiment (VADER), clickbait patterns, lexical stats (length, words, capitalization), question/exclamation flags.
- Data quality: <0.5% missing after imputation, no high-correlation (>0.85) clusters.
- **Figure:** Feature overview diagram (capture from notebook cell `#VSC-dfd42bf5`).

## Slide 6 – Stage A Exposure Model
- Model: OLS regression on log score with exposure + context predictors.
- Dataset split: 80/20 chronological holdout.
- **Key Metrics:**
  - RMSE (log space): **1.0593**
  - MAE (log space): **0.7787**
  - Residual mean ~0.00, std 1.06
- Top positive quality factors: early score deltas, snapshot velocity, weekday/hour interactions.
- **Visuals:** Residual histogram & QQ plot (`#VSC-ca819e1d`, `#VSC-1cdda1c1`).

## Slide 7 – Stage B Title Models Overview
- Stage B regresses Stage A residuals on title features.
- Variants evaluated:
  - OLS (interpretable baseline)
  - Centered OLS (zeroed exposures)
  - Elastic Net (automatic feature selection)
  - LightGBM (non-linear interactions)
- Cross-validation: blocked temporal folds aligned with collection dates.
- **Table:** Title feature summary (see Appendix A).

## Slide 8 – Performance Comparison
| Metric | Stage A baseline | Stage B (OLS) | Stage B (Elastic Net) | Stage B (LightGBM) | LightGBM gain vs Stage A |
| --- | --- | --- | --- | --- | --- |
| RMSE (log) | 1.0593 | 1.0491 | 1.0515 | **0.8697** | 0.1896 |
| MAE (log) | 0.7787 | 0.7771 | 0.7762 | **0.6398** | 0.1389 |
| Residual std | 1.0593 | 1.0492 | 1.0515 | **0.8697** | 0.1896 |
- LightGBM delivers the largest improvement while hitting the 60% pairwise accuracy target comfortably.
- Elastic Net provides modest gains with easily explainable coefficients; consider for production when interpretability is paramount.

## Slide 9 – Pairwise Ranking Accuracy
| Model | Accuracy | Lift vs Stage A |
| --- | --- | --- |
| Stage A baseline | 0.8279 | 0.0000 |
| Stage A + Stage B (OLS) | 0.8303 | 0.0024 |
| Stage A + Stage B (Elastic Net) | 0.8313 | 0.0034 |
| Stage A + Stage B (LightGBM) | **0.8700** | **0.0421** |
| Title residual only (LightGBM) | 0.6584 | -0.1695 |
- Combined exposure + title models exceed the proposal’s ≥0.60 target; LightGBM delivers +4.2 percentage points absolute lift.
- Residual-only signals underperform, reinforcing the need for Stage A normalization.
- **Figure:** Pairwise accuracy bar chart (export notebook output `#VSC-3d49cf4b`).

## Slide 10 – Subreddit & Temporal Insights
- Strongest subreddit lifts (LightGBM vs Stage A):
  - `economy` +7.7 pts, `worldnews` +5.1 pts, `gadgets` +4.8 pts.
- Minimal or negative lift: `science` (-0.4 pts) suggesting domain-specific language patterns.
- Day/Hour hot spots: Thu 15:00–18:00 UTC and Sun 12:00–18:00 show 3–6 pt gains in combined models.
- **Figures:**
  - Subreddit lift tables (`#VSC-04cd7a3c` / `#VSC-b894f6ad`).
  - Day-hour heatmap (capture from `#VSC-04cd7a3c` output).

## Slide 11 – Diagnostic Checks
- Residual distribution ~normal post Stage B; absence of heavy tails.
- QQ plots indicate improved alignment except extreme upper decile (investigate trending-event spikes).
- Temporal holdouts: Stage B reduces RMSE on 0.65–0.80 quantiles (gains 0.007–0.010).
- Blocked CV: 10 of 12 folds show positive Stage B lift; largest uplift on 2025-11-06 block (+0.096 RMSE reduction).
- Bootstrapped RMSE stable (σ ≈ 0.008); pairwise accuracy σ ≈ 0.006.
- **Visuals:** Blocked CV chart, bootstrap violin (cells `#VSC-5804260f`, `#VSC-c871542e`).

## Slide 12 – Feature Attribution
- OLS coefficients (absolute top 10): sentiment buckets dominate (negative coefficients ~ -117 due to reference contrast).
- Elastic Net focuses on `title_words`, `title_length`, `sentiment_compound`, `has_question`.
- LightGBM SHAP importances:
  1. capitalization_ratio (0.063)
  2. title_length (0.058)
  3. title_words (0.056)
  4. sentiment_negative (0.054)
  5. chars_per_word (0.053)
- **Figure:** SHAP bar chart (cell `#VSC-731e7a27`).

## Slide 13 – Case Studies
- High-lift titles: show top residual improvements (select from `outputs/title_lift/stage_model_outputs.parquet`).
- Misses: `science` posts where Stage B underperforms; inspect lexical patterns lacking sentiment polarity.
- **Table:** Provide 3 positive & 3 negative exemplars with Stage A prediction, Stage B adjustment, actual score.

## Slide 14 – Recommendations
1. Adopt Stage A + Stage B LightGBM for ranking experiments; maintain OLS pipeline for interpretability reports.
2. Integrate title-length and capitalization guidance into editorial playbooks (backed by SHAP importances).
3. For subreddits with minimal gain (`science`), evaluate domain-specific NLP features (scientific jargon lexicons).
4. Automate nightly bootstrap monitoring to ensure lift stability.

## Slide 15 – Next Steps
- Extend feature set with topic embeddings (e.g., Sentence-BERT) for nuanced semantics.
- Deploy online A/B test comparing Stage A baseline vs. Stage A+B LightGBM on curated feeds.
- Investigate causal uplift via matched pairs (control for author/posting time).
- Document pipeline in `README.md` and set up CI for data quality checks.

## Slide 16 – Appendix A: Metrics Table
| Metric | Stage A baseline | Stage B (OLS) | Stage B (Elastic Net) | Stage B (LightGBM) |
| --- | --- | --- | --- | --- |
| RMSE (log space) | 1.0593 | 1.0491 | 1.0515 | 0.8697 |
| MAE (log space) | 0.7787 | 0.7771 | 0.7762 | 0.6398 |
| Residual std | 1.0593 | 1.0492 | 1.0515 | 0.8697 |
| Pairwise accuracy | 0.8279 | 0.8303 | 0.8313 | 0.8700 |

## Slide 17 – Appendix B: Subreddit Lift Snapshot (LightGBM)
| Subreddit | Pairs | Baseline acc | Combined acc | Lift |
| --- | --- | --- | --- | --- |
| economy | 8666 | 0.7984 | 0.8758 | +0.0774 |
| worldnews | 10964 | 0.8018 | 0.8532 | +0.0514 |
| gadgets | 168 | 0.9048 | 0.9524 | +0.0476 |
| technology | 9553 | 0.8632 | 0.9098 | +0.0466 |
| politics | 43027 | 0.8160 | 0.8518 | +0.0358 |

## Slide 18 – Appendix C: Day/Hour Hot Spots (LightGBM)
| Day (UTC) | Hour | Pairs | Baseline acc | Combined acc | Lift |
| --- | --- | --- | --- | --- | --- |
| Thu | 15 | 1064 | 0.8224 | 0.8731 | +0.0507 |
| Tue | 16 | 967 | 0.8221 | 0.8862 | +0.0641 |
| Sun | 18 | 946 | 0.7992 | 0.8626 | +0.0634 |
| Mon | 12 | 978 | 0.8139 | 0.8753 | +0.0614 |
| Wed | 14 | 959 | 0.8332 | 0.8801 | +0.0469 |

## Slide 19 – Appendix D: References & Artifacts
- Notebook: `Title_Lift_Pipeline.ipynb` (fully executed on Dec 3, 2025).
- Outputs: `outputs/title_lift/stage_model_outputs.parquet`, `outputs/title_lift/stage_penalized_metrics.json`, `outputs/title_lift/stage_b_enhancements.json`.
- Diagnostics: `docs/diagnostics/` (blocked CV, temporal splits, bootstrap summaries).
- Figures (export from notebook to `docs/figures/` for final deck):
  - Residual histogram & QQ (`cells #VSC-ca819e1d` / `#VSC-1cdda1c1`)
  - Pairwise accuracy table (`#VSC-3d49cf4b`)
  - Subreddit/day-hour tables (`#VSC-04cd7a3c`)
  - SHAP importance plot (`#VSC-731e7a27`)

## Presenter Notes (Optional)
- Emphasize interplay between exposure normalization and residual modeling.
- Call out LightGBM vs. Elastic Net trade-offs when stakeholders ask about interpretability.
- Highlight data governance: hashed authors, snapshot cadence compliance.
- Prepare to discuss why sentiment coefficients are large negative magnitudes (due to dummy encoding sum-to-zero constraint).
