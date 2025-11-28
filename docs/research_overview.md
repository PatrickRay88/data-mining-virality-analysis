# Virality Analysis Research Overview

_Last updated: 2025-11-28_

## Project At A Glance
- Focus: quantify Reddit title lift while separating intrinsic quality (Stage A) from headline effects (Stage B) across 11 technology-adjacent subreddits
- Sample: 13,395 posts collected between 2025-09-28 and 2025-11-18 with complete 5/15/30/60 minute exposure snapshots
- Outcomes: log-transformed 60-minute score (`y = log1p(score_60m)`) with Stage A residual (`R`) as the Stage B target
- Toolchain: Python 3.11 virtualenv, Click CLIs under `bin/`, modeling utilities under `src/models/`, artifacts persisted in `outputs/title_lift/`
- Key references: Weissburg et al. (ICWSM 2022) baseline, Stage A/B replication, and expanded embedding experiments documented in `Title_Lift_Pipeline.ipynb`

## Data Collection & Snapshot Instrumentation
- Reddit collectors (`bin/collect_reddit.py`, `bin/run_snapshot_collector.py`) authenticate via `.env` (`REDDIT_CLIENT_ID/SECRET/USER_AGENT`) and stream `/new` along with periodic `/top` backfills
- `run_snapshot_collector.py` pairs with `bin/run_snapshot_collector_week.ps1` to run a 7-day loop (300 second cadence) that captures 5/15/30/60 minute score, comment, and upvote ratio snapshots; state is persisted in `data/state/reddit_snapshot_state.parquet`
- Subreddits covered: `technology`, `science`, `worldnews`, `business`, `economy`, `space`, `futurology`, `gadgets`, `energy`, `technews`, `politics`; each loop writes normalized slices (`reddit_stream_*.parquet`) and aligned snapshot files (`data/snapshots/reddit_snapshots_*.parquet`)
- Reddit API helper (`src/ingest/reddit_client.py`) enforces polite rate limiting, retries, and field normalization; `snapshot_manager.py` tracks due intervals, tolerances (±2 minutes), and pruning grace windows (120 minutes)
- Hacker News scaffolding (`bin/collect_hn.py`, `src/ingest/hn_client.py`) mirrors the architecture but remains optional in this phase; cross-platform hooks are ready once more data is ingested

## Normalization, Storage, and Provenance
- `DataNormalizer` (`src/preprocess/normalize.py`) harmonizes raw API payloads into a unified schema, hashes authors via SHA256-8 to preserve privacy, stamps platform/subreddit metadata, and exposes Parquet IO helpers
- All raw Parquet assets live under `data/` with timestamped filenames (`reddit_<subreddit>_<ISO>.parquet`, `hackernews_<type>_<ISO>.parquet`) to support reproducible reruns; directory is `.gitignore`d and shared on request
- Snapshot alignment uses `SnapshotConfig` and `SnapshotStateManager` to merge newly seen IDs, mark collected intervals, and prune stale posts; ingest logs are streamed to `logs/snapshot_run_<timestamp>.log`
- Feature-ready table: `bin/make_features.py` consolidates normalized slices, deduplicates `post_id`, and emits `data/features.parquet` with 90+ engineered columns; final modeling table resides in `outputs/title_lift/stage_model_outputs.parquet`

## Feature Factory Summary
- Title features (`TitleFeatureExtractor`): length (`title_length`, `title_words`, `title_chars_per_word`), punctuation flags, capitalization ratios, numeric tokens, VADER sentiment scores, textstat readability, clickbait keyword/pattern counts, spaCy NER tallies (`person_entities`, `org_entities`, `date_entities`)
- Context features (`ContextFeatureExtractor`): hour and weekday indicators, cyclical encodings (`hour_sin`, `hour_cos`), recency buckets (`is_very_new`, `log_age_hours`), content-type detection (`is_text_post`, `is_image_post`, `is_news_post`, `is_external_link`), author activity proxies (`author_post_count_log`, `is_frequent_poster`), subreddit aggregates merged during feature build
- Early exposure metrics: dense snapshot alignment populates `score_5m`, `score_15m`, `score_30m`, `score_60m`, growth rates (`velocity_5_to_30m`, `early_momentum`, `sustained_growth`); fallback logic fills gaps with final score when snapshots missing (now zero cases after November reruns)
- Quality-of-life columns: `collection_type`, `is_new_collection`, platform flags, hashed authors, domain heuristics, and Stage A predictions/residuals persisted for downstream notebooks

## Modeling Methodology & Experiments
- **Stage A (Exposure & Intrinsic Quality)**
  - Baseline OLS (`docs/stage_metrics.json`): train RMSE 0.96 / R² 0.876, test RMSE 1.19 / R² 0.577 on `log1p(score_60m)` using temporal, author, subreddit, and early score features
  - ElasticNet replica (`outputs/title_lift/stage_penalized_metrics.json`): test RMSE 1.26 / R² 0.527 with top coefficients from early velocity and subreddit dispersion, matching Weissburg et al.'s penalized setup
  - Diagnostics across temporal quantiles (`docs/stage_model_diagnostics.json`) show test R² stability 0.54–0.58, residual skew dropping from 0.58 → 0.33 as snapshots densified
- **Stage B (Title Lift Residual)**
  - Heuristic ElasticNet (`docs/stage_metrics.json`): test RMSE 1.18 / R² −0.028, pairwise accuracy 0.55; dominant features include sentiment proportions, title length, and clickbait indicators
  - Tree baseline (`stage_metrics.json.stage_b_tree`): LightGBM-style regressor test RMSE 1.176 / R² −0.024 with capitalization ratio leading splits; corroborates weak linear signal
  - Penalized pipeline (`stage_penalized_metrics.json`): Stage B ElasticNet (same residual target) nudges test R² to 0.018 under stronger Stage A shrinkage
  - TF-IDF + SVD enhancements (`outputs/title_lift/stage_b_enhancements.json`): test RMSE 1.217 / R² −0.037, reinforcing that bag-of-words alone underperforms heuristic baseline
- **Embedding Sweep (`stage_b_embedding_search_v2/`)**
  - Grid over `all-MiniLM-L6-v2` and `all-mpnet-base-v2`, PCA {64,128}, ridge/elastic net/MLP heads; ranked results stored in `search_summary.json`
  - Best run: ridge + `all-mpnet-base-v2` + PCA64, batch 32 → test RMSE 1.1897, R² 0.0085, MAE 0.897, pairwise accuracy 0.600 (15,217 ordered pairs)
  - Secondary configs (PCA128, MiniLM) cluster around test R² 0.002–0.008 with lower pairwise accuracy; MLP heads overfit (train RMSE <0.3, test RMSE >1.25)
- **Non-Linear Heuristics (Notebook)**
  - `Title_Lift_Pipeline.ipynb` Section 8 compares MLP, Random Forest, Gradient Boosting, SVR on Stage B residuals: MLP achieves test RMSE 1.074 / R² 0.012; trees hover near zero lift
  - Notebook Sections 9–11 contain permutation importance, residual slicing, and HTML summary exported to `docs/title_lift_analysis_report.html`

## Evaluation & Diagnostics Assets
- `docs/stage_model_diagnostics.json`: temporal split ladder (quantiles 0.6–0.8), blocked day-wise CV, bootstrap summaries (`docs/diagnostics/stage_model_bootstrap_records.csv`, `stage_model_bootstrap_summary.csv`), learning curves (`stage_model_learning_curve.csv`)
- Residual inspection: `docs/top_residual_posts.csv`, `docs/bottom_residual_posts.csv`, residual distribution figures (`docs/figures/residual_hist.png`, `residual_by_hour.png`, `residual_heatmap.png`)
- Embedding leaderboard: `docs/figures/embedding_search_top5.png`; HTML Sankey/flow diagrams under `docs/figures/title_lift_pipeline_*.html`
- Data audit: `outputs/title_lift/data_audit_summary.json` confirms column set and target moments; `outputs/title_lift/rollup_summary.csv` aggregates per-stage metrics
- Stage B coefficients and embeddings: `outputs/title_lift/stageB_coefs.csv`, `stage_b_embeddings.json`, `stage_b_enhancements.json`, embedding search folders (`stage_b_embedding_search/`, `_v2`, `_smoke`)

## Deliverables & Reproducibility
- Primary notebook: `Title_Lift_Pipeline.ipynb` (exploration, modeling, diagnostics); mirrored report HTML in `docs/title_lift_analysis_report.html`
- Operational scripts: collectors + snapshot PS helper, feature builder, modeling runners (`bin/run_stage_b_embeddings.py`, `bin/run_stage_b_enhancements.py`, `bin/run_stage_penalized.py`, `bin/run_model_diagnostics.py`)
- Outputs: modeling metrics JSON/CSV under `docs/` and `outputs/title_lift/`, figures in `docs/figures/`, logs in `logs/`
- Setup checklist: `pip install -r requirements.txt`, `python -m spacy download en_core_web_sm`, populate `.env`, optionally schedule `run_snapshot_collector_week.ps1` inside Windows Task Scheduler or `Start-BitsTransfer`

## Current Status & Next Steps
- Stage A is production-ready for Reddit tech verticals with full snapshot coverage and stable diagnostics
- Stage B shows modest lift (≤0.018 R² except for embedding ridge 0.0085), with r/energy, r/space, r/futurology exhibiting the highest localized gains (see `docs/findings.md` for per-subreddit table)
- Pending work: cross-platform ingestion (Hacker News), segmentation experiments, explainability on embedding ridge model (SHAP/permutation), and causal counterfactual tests now that early exposure data is dense
- Continue weekly snapshot runs, refresh features via `bin/make_features.py`, rerun `bin/run_model_diagnostics.py` post-refresh, and keep `docs/findings.md` + `docs/final_report.md` synchronized after major modeling cycles
