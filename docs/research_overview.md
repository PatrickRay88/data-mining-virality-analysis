# Virality Analysis Research Overview

## 1. Baseline Paper: Weissburg et al. (ICWSM 2022)
- **Goal**: Decompose social news virality into intrinsic content quality vs. exposure dynamics.
- **Data**: Reddit + Hacker News submissions with title text, timestamps, article bodies, and engagement metrics (score, comments, upvotes).
- **Methodology**:
  - Stage A models exposure-adjusted outcomes with contextual controls (timing, subreddit/platform, author history, early score snapshots).
  - Stage B regresses Stage-A residuals on intrinsic/title features to quantify incremental lift.
  - Iterative residual analysis isolates how headlines and platform effects boost or suppress reach beyond exposure.
- **Key Findings**:
  - Title framing (e.g., questions, numerics) yields measurable lift beyond exposure when averaged across platforms.
  - Early exposure windows (first 60 minutes) strongly correlate with eventual virality.
  - Cross-platform comparisons reveal platform-specific lift patterns even for similar content.
- **Implications**:
  - Clean feature partitioning is required to separate exposure from headline effects.
  - High-quality temporal snapshots remain critical for early growth modeling.
  - Interpretable models (elastic nets, GAMs) surface actionable headline patterns.

## 2. Current Project Scope (Reddit Multi-Subreddit Phase)
- **Target Platform**: 11 technology-adjacent subreddits (e.g., r/technology, r/politics, r/worldnews, r/energy) collected via authenticated Reddit API (`.env`-backed PRAW).
- **Data Assets**:
  - `data/features.parquet`: harmonized Reddit feature table (13,395 posts, 90+ engineered columns) produced by `bin/make_features.py`.
  - `outputs/title_lift/stage_model_outputs.parquet`: Stage A/B predictions, residuals, and title features for reporting.
  - `docs/stage_metrics.json`: persisted metrics + calibration artifacts from the latest Stage A/B run.
- **Research Question** (phase-specific):
  - "How much incremental lift do Reddit headlines provide after accounting for exposure controls across technology-focused communities?"

## 3. Completed Work
- End-to-end ingest + normalization for 11 subreddits, including hashed authors and subreddit-level aggregates.
- Feature engineering covering temporal flags, early velocity proxies, author frequency, and standardized title attributes (sentiment, readability, clickbait heuristics, entity counts).
- Stage A OLS baseline (context-only) reaching **train RMSE 0.96 / R² 0.88** and **test RMSE 1.19 / R² 0.58** on log score targets (`docs/stage_metrics.json`).
- Stage B elastic net residual model (title-only) delivering **test RMSE 1.18 / R² -0.03**, indicating minimal out-of-sample lift; LightGBM and OLS corroborate the weak signal.
- Subreddit-level diagnostics show Stage B achieves only small positive R² (≤0.08) in best cases (e.g., r/energy) and remains negative for others (`docs/findings.md`).
- Added `bin/run_stage_b_enhancements.py` for sandboxing richer headline models (TF-IDF n-grams + additional heuristics) without touching the core pipeline; default run mirrors Stage A split for apples-to-apples tracking.
- Added `src/models/stage_penalized.py` + `bin/run_stage_penalized.py`, an ElasticNet-based replica of Weissburg et al.'s penalized workflow. First run (Ridge-like Stage A, residual Stage B) yields **Stage A test RMSE 1.28 / R² 0.51** and **Stage B test RMSE 1.24 / R² 0.03**, stored in `outputs/title_lift/stage_penalized_metrics.json`.

## 4. Next Action Plan
1. **Exposure Instrumentation**
  - Expand use of `bin/run_snapshot_collector.py` to capture dense 5/15/30/60 minute snapshots; backfill feature pipeline once coverage improves.
2. **Segmented Residual Modeling**
  - Fit Stage B per subreddit or cluster with sufficient support to test whether localized headline norms matter; document best/worst segments.
3. **Title Feature Enrichment**
  - Add semantic vectors (e.g., sentence-transformer embeddings) or topic proportions to complement heuristic features; evaluate on held-out weeks.
4. **Reporting & Docs**
  - Keep `docs/findings.md` synchronized with new diagnostics (per-subreddit table, calibration plots) and summarize takeaways in this overview.

## 5. Future Extensions (Deferred)
- **Cross-Platform Expansion**: Re-introduce Hacker News via `bin/collect_hn.py`, align normalization, and compare lift patterns across platforms.
- **Snapshot-Based Early Growth**: Automate long-running collectors with `--snapshots` to mirror Weissburg et al.'s exposure features.
- **Causal Storytelling**: Explore A/B-like comparisons (similar content, different headlines) or counterfactual simulations once Stage B shows reliable lift.

## 6. Resources & References
- Weissburg, G., Hohenstein, J., Zhang, A., & Shah, N. (2022). *Title Lift: Measuring Headline Impact on Social News*. ICWSM 2022.
- Project scripts: `bin/collect_reddit.py`, `bin/make_features.py`, `bin/run_model_diagnostics.py`.
- Feature modules: `src/preprocess/features_titles.py`, `src/preprocess/features_context.py`, `src/preprocess/normalize.py`.
- Modeling entry point: `src/models/stage_modeling.py` (CLI + helper functions).
- Environment setup: `.env` with Reddit credentials; see `.env.example`.

---
Keep this document updated as milestones are reached (Stage A modeling complete, residual insights captured, etc.). When resuming cross-platform analysis, revisit Sections 2 and 5 to expand scope and adjust timelines.
