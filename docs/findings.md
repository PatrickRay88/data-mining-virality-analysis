# Reddit Virality Modeling Findings (Stage A/B Update)

_Last updated: 2025-11-28_

## Dataset & Pipeline Recap
- Coverage: 13,395 Reddit submissions across 11 tech-adjacent communities with full 5/15/30/60 minute snapshot coverage (verified via `outputs/title_lift/stage_model_outputs.parquet`)
- Inputs: harmonized tables from `bin/make_features.py`, title/context features from `src/preprocess/features_titles.py` and `src/preprocess/features_context.py`, snapshots merged from `data/snapshots/`
- Targets: `y = log1p(score_60m)` for Stage A; Stage B regresses the saved residual `R = y - yhat_A`
- Temporal split: 70/30 chronological boundary at 2025-11-12 00:33:19 UTC to respect deployment-style evaluation
- Supporting artifacts: metrics in `docs/stage_metrics.json`, diagnostics ladder in `docs/stage_model_diagnostics.json`, residual exemplars in `docs/top_residual_posts.csv` and `docs/bottom_residual_posts.csv`

## Stage A Quality Model
|Model|Train RMSE|Train R²|Test RMSE|Test R²|Source|
|---|---|---|---|---|---|
|OLS (exposure controls + snapshots)|0.957|0.876|1.190|0.577|`docs/stage_metrics.json`|
|ElasticNet (Weissburg replica)|1.001|0.864|1.258|0.527|`outputs/title_lift/stage_penalized_metrics.json`|

- Feature signals: early velocity (`velocity_5_to_30m`), `log_score_5m`, subreddit dispersion (`subreddit_score_std`), and author engagement proxies dominate coefficient magnitudes
- Residual summary (`docs/stage_metrics.json.residual_summary`): mean 0.077, σ 1.03, skew 0.44; improvements vs. October runs stem from denser snapshots
- Diagnostics: temporal quantiles (0.6–0.8) in `docs/stage_model_diagnostics.json` keep Stage A test R² within 0.54–0.58 with shrinking skew as the split boundary moves forward

## Stage B Residual Models (Heuristics)
- Baseline ElasticNet (title heuristics): test RMSE 1.178, test R² −0.028, pairwise accuracy 0.550 (`docs/stage_metrics.json`)
- Gradient-boosted tree head: test RMSE 1.176, test R² −0.024; capitalization ratio, title length, sentiment components lead split importances
- Penalized replica (paired with ElasticNet Stage A): test RMSE 1.230, test R² 0.018 (`outputs/title_lift/stage_penalized_metrics.json`); numbers align with Weissburg et al.'s reported lift after heavy shrinkage
- Notebook non-linear baseline (MLP, `Title_Lift_Pipeline.ipynb` §8): test RMSE 1.074, test R² 0.012; forests/boosting trail with R² ≈ 0.0

## Embeddings, TF-IDF, and Search Experiments
|Experiment|Configuration|Test RMSE|Test R²|Pairwise|Artifact|
|---|---|---|---|---|---|
|Sentence transformer ridge|`all-mpnet-base-v2`, PCA64, ridge α∈{0.1,1,10}|1.1897|0.0085|0.600|`stage_b_embedding_search_v2/search_summary.json`|
|Sentence transformer ridge (MiniLM)|`all-MiniLM-L6-v2`, PCA64, ridge|1.1944|0.0007|0.574|`stage_b_embedding_search_v2/search_summary.json`|
|Sentence transformer ridge (MiniLM, no PCA)|`all-MiniLM-L6-v2`, no PCA, ridge|1.2009|−0.010|0.564|`outputs/title_lift/stage_b_embeddings.json`|
|TF-IDF + SVD ElasticNet|500 vocab, 100 SVD comps|1.2168|−0.037|—|`outputs/title_lift/stage_b_enhancements.json`|
|Heuristic MLP (notebook)|Single hidden layer 128|1.074|0.012|—|`Title_Lift_Pipeline.ipynb`|

- Embedding sweep takeaway: transformer features beat heuristics by pairwise ordering but yield marginal R² (≤0.009); PCA64 stabilizes ridge fits without inflating variance
- Over-parameterized MLP heads (search grid) collapse with extreme train/test gaps, indicating limited residual signal despite richer embeddings
- TF-IDF regression underperforms due to sparse titles and heavy regularization; no ranked pairs recorded for that run

## Per-Community Residual Behavior
|Subreddit|Count|Stage A RMSE|Stage A R²|Stage B RMSE|Stage B R²|Stage B Corr|
|---|---|---|---|---|---|---|
|Futurology|642|0.951|0.877|0.936|0.031|0.181|
|business|482|0.564|0.912|0.589|−0.091|−0.009|
|economy|1664|0.883|0.818|0.867|0.035|0.186|
|energy|611|0.636|0.916|0.610|0.080|0.284|
|gadgets|218|1.027|0.858|1.014|0.026|0.174|
|politics|4387|1.096|0.744|1.087|0.016|0.129|
|science|722|1.219|0.797|1.211|0.014|0.194|
|space|748|0.840|0.900|0.821|0.046|0.217|
|technews|439|0.753|0.885|0.757|−0.012|0.039|
|technology|1636|1.277|0.824|1.260|0.026|0.175|
|worldnews|1846|1.217|0.760|1.212|0.008|0.089|

- Communities with scientific/energy focus show the highest residual correlation (≤0.28) hinting at niche headline norms; business/technews trend negative
- Per-hour residual heatmap (`docs/figures/residual_heatmap.png`) highlights late-afternoon pockets where Stage B modestly closes the gap

## Residual Outliers & Distribution Checks
- Top positive residuals (`docs/top_residual_posts.csv`) illustrate sensational political/world news posts exceeding Stage A expectations by >3 log points (e.g., score 33k posts during Oct 31 news cycle)
- Bottom residuals (`docs/bottom_residual_posts.csv`) feature mid-score business pieces overestimated by Stage A, often with low engagement titles
- Residual histogram (`docs/figures/residual_hist.png`) is centered near zero with heavy right tail; hourly violin (`docs/figures/residual_by_hour.png`) confirms higher evening variance
- Pairwise evaluation: 15,224 ordered comparisons produce 0.55 accuracy for heuristics and 0.60 for best embedding run, demonstrating small but real ranking gains

## Alignment with Weissburg et al. (2022)
- Stage A deviance explained (≈0.58) matches the original 0.45–0.60 band once exposure controls and snapshots are dense
- Stage B lift remains marginal; penalized pipeline’s positive 0.018 test R² sits inside Weissburg’s reported incremental gains, underscoring consistent difficulty of headline-only prediction
- Snapshot instrumentation complete for all 13,395 posts, removing the prior blocker to causal counterfactual tests proposed in the paper’s follow-up work

## Figures & Artifacts Checklist
- Residual distribution suite: `docs/figures/residual_hist.png`, `residual_by_hour.png`, `residual_heatmap.png`
- Embedding leaderboard: `docs/figures/embedding_search_top5.png` with top-5 runs annotated
- Flow visuals: `docs/figures/title_lift_pipeline_flow.html`, `title_lift_pipeline_sankey*.html` for presentations
- Diagnostics tables: `docs/diagnostics/stage_model_bootstrap_summary.csv`, `stage_model_temporal_splits.csv`, `stage_model_blocked_cv.csv`
- Notebook export: `docs/title_lift_analysis_report.html` for end-to-end reproducibility without local data

## Discussion & Next Steps
- Exposure modeling is mature; the limiting factor is residual variance dominated by platform dynamics and author effects rather than headline tokens
- Embedding ridge confirms slight lift (pairwise 0.60) yet R² < 0.01, suggesting that richer semantic features alone cannot radically improve predictions without segmentation
- Segmenting by community or temporal regime, plus SHAP/permutation analysis on the embedding ridge, should clarify whether title tone (sentiment_neg, entity counts) drives the limited gains
- Next actions: maintain weekly snapshot schedule, extend embedding search to contrastive/multilingual encoders, trial per-subreddit Stage B fits, and prepare causal counterfactual experiments now that exposure coverage is complete
