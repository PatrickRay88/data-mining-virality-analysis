# Title Lift Pipeline – Project Outline

## 1. Research Motivation & Proposal Alignment
- Investigate whether Reddit/Hacker News post titles drive incremental engagement beyond early exposure, extending Weissburg et al. (ICWSM 2022).
- Proposal success criteria: maintain reliable intrinsic scoring (Stage A) while reaching ≥60% pairwise accuracy for title-based ranking decisions.
- Deliverables: refreshed dataset, two-stage modeling pipeline, diagnostics/visuals, human-readable analysis assets for publication.

## 2. Data Collection & Sources
- Reddit snapshots via `bin/collect_reddit.py` (5-minute polling for target subreddits; credentials stored in `.env`).
- Hacker News stories via `bin/collect_hn.py` (REST polling with polite rate limiting).
- Outputs stored as timestamped parquet files under `data/`, preserving score snapshots (`score_5m`, `score_15m`, `score_30m`, `score_60m`).
- Logging and state tracking maintained in `data/state/` and `logs/` for reproducibility.

## 3. Normalization & Feature Engineering
- Harmonize Reddit + HN schemas with `src/preprocess/normalize.py` (hash authors, align timestamps, unify engagement metrics).
- Title features generated in `src/preprocess/features_titles.py`: sentiment (VADER), clickbait heuristics, readability, structural cues.
- Context features from `src/preprocess/features_context.py`: hour-of-day, day-of-week, exposure velocity, author/subreddit history.
- Combined feature table produced by `bin/make_features.py` → `data/features.parquet` (basis for notebook and modeling scripts).

## 4. Stage A – Intrinsic Quality Modeling
- Goal: predict log-scaled final score using early exposure (`score_5m`) plus contextual covariates.
- Implementation: OLS/Elastic Net within `Title_Lift_Pipeline.ipynb` Cell 9–13 and `src/models/stage_penalized.py` for CLI runs.
- Diagnostics: residual distributions, QQ plots, calibration tables by hour/subreddit.

## 5. Stage B – Title Lift Modeling
- Input: Stage A residuals (`R = y − ŷ_A`).
- Baseline: OLS regression on title features; Centered variant for interpretability.
- Enhancements: Elastic Net (sparse selection) and LightGBM with SHAP explanations; contextual embeddings excluded per latest scope.
- Outputs: coefficient tables, feature importances, residual scatter plots, stored under `outputs/title_lift/` and `docs/figures/`.

## 6. Evaluation & Diagnostics
- Stage metrics: RMSE/MAE comparisons (Stage A vs Stage B variants) from notebook Cells 24–27.
- Residual analytics: hourly/subreddit tables, residual vs fitted plots, temporal drift visualizations (Cells 28–35).
- Pairwise ranking evaluation: constructs within-hour subreddit pairs (Cells 30–33) to measure success criterion.
- External robustness: ingest CLI diagnostics (temporal splits, blocked CV, bootstrap, learning curves) in Cells 34–36.

## 7. Results Synthesis
- Stage B reduces RMSE modestly and improves pairwise accuracy toward the 60% target, with strongest gains in technology/business communities.
- Residual analyses confirm remaining exposure bias pockets (late-night cohorts) and highlight contexts for intervention.
- Human-readable dataset (`docs/title_lift_human_readable.csv`) and report (`docs/title_lift_final_report.docx`) summarize findings for stakeholders.

## 8. Discussion & Future Work
- Risks: residual calibration drift on low-volume subreddits, sensitivity to feature freshness (sentiment lexicon, clickbait patterns).
- Next steps: ongoing data refresh automation, consider contextual embeddings + metadata once scope reopens, deploy online A/B tests to validate offline ranking gains.
- Publication assets: final report, presentation deck, notebook outputs, and diagnostic figures ready for submission.
