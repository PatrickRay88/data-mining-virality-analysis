#!/usr/bin/env python3
"""CLI for Stage B enhancement experiments.

This script loads the existing stage model outputs, augments title features,
fits an ElasticNet residual model with TF-IDF n-grams, and reports metrics.
It writes results to JSON (and optional CSV of top coefficients) without
modifying the primary Stage A/B pipeline artifacts.
"""

from __future__ import annotations

import argparse
# Ensure project root on sys.path
import os
import sys
from pathlib import Path

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.models.stage_b_enhancements import (
    StageBExperimentConfig,
    run_stage_b_experiment,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run enhanced Stage B headline modeling experiments.",
    )
    parser.add_argument(
        "--stage-outputs",
        default="outputs/title_lift/stage_model_outputs.parquet",
        type=Path,
        help="Path to Stage A/B combined outputs parquet.",
    )
    parser.add_argument(
        "--features",
        default="data/features.parquet",
        type=Path,
        help="Feature table with created_dt column for temporal split.",
    )
    parser.add_argument(
        "--split-quantile",
        default=0.7,
        type=float,
        help="Temporal quantile for train/test split (matching Stage A baseline).",
    )
    parser.add_argument(
        "--tfidf-max-features",
        default=500,
        type=int,
        help="Maximum TF-IDF feature count.",
    )
    parser.add_argument(
        "--tfidf-min-df",
        default=5,
        type=int,
        help="Minimum document frequency for TF-IDF vocabulary.",
    )
    parser.add_argument(
        "--use-svd",
        action="store_true",
        help="Apply Truncated SVD to TF-IDF features to create dense embeddings.",
    )
    parser.add_argument(
        "--svd-components",
        default=100,
        type=int,
        help="Number of components for Truncated SVD when --use-svd is set.",
    )
    parser.add_argument(
        "--output-json",
        default="outputs/title_lift/stage_b_enhancements.json",
        type=Path,
        help="Where to write experiment metrics as JSON.",
    )
    parser.add_argument(
        "--output-coefs",
        default="outputs/title_lift/stage_b_enhancements_features.csv",
        type=Path,
        help="Optional CSV with top positive/negative coefficients.",
    )
    parser.add_argument(
        "--tfidf-ngram-max",
        default=2,
        type=int,
        help="Maximum n-gram length for TF-IDF (minimum is fixed at 1).",
    )
    parser.add_argument(
        "--max-iter",
        default=5000,
        type=int,
        help="Maximum iterations for ElasticNet solver.",
    )
    parser.add_argument(
        "--random-state",
        default=42,
        type=int,
        help="Random seed for reproducibility (impacts ElasticNet CV).",
    )
    parser.add_argument(
        "--l1-ratios",
        default="0.1,0.3,0.5,0.7,0.9",
        help="Comma-separated l1_ratio values for ElasticNetCV grid.",
    )
    parser.add_argument(
        "--alphas",
        default="",
        help="Optional comma-separated alpha grid. Leave blank to auto-tune.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    l1_ratios = tuple(
        float(item.strip())
        for item in args.l1_ratios.split(",")
        if item.strip()
    ) or (0.5,)

    alphas = tuple(
        float(item.strip())
        for item in args.alphas.split(",")
        if item.strip()
    )
    if not alphas:
        alphas = None

    config = StageBExperimentConfig(
        split_quantile=args.split_quantile,
        tfidf_max_features=args.tfidf_max_features,
        tfidf_min_df=args.tfidf_min_df,
        tfidf_ngram_range=(1, args.tfidf_ngram_max),
        l1_ratios=l1_ratios,
        alphas=alphas,
        max_iter=args.max_iter,
        random_state=args.random_state,
        use_svd_embeddings=args.use_svd,
        svd_components=args.svd_components,
    )

    result = run_stage_b_experiment(
        stage_outputs_path=args.stage_outputs,
        features_path=args.features,
        output_path=args.output_json,
        top_features_path=args.output_coefs,
        config=config,
    )

    print("Stage B enhancement metrics:")
    print(f"  Train RMSE: {result.train_rmse:.3f}")
    print(f"  Test RMSE:  {result.test_rmse:.3f}")
    print(f"  Train R^2:  {result.train_r2:.3f}")
    print(f"  Test R^2:   {result.test_r2:.3f}")
    print(f"  Train MAE:  {result.train_mae:.3f}")
    print(f"  Test MAE:   {result.test_mae:.3f}")
    print(f"  Train count: {result.n_train}")
    print(f"  Test count:  {result.n_test}")
    print("Top positive features (see CSV for full list):")
    for item in result.top_positive_features[:5]:
        print(f"    + {item['feature']}: {item['weight']:.4f}")
    print("Top negative features (see CSV for full list):")
    for item in result.top_negative_features[:5]:
        print(f"    - {item['feature']}: {item['weight']:.4f}")


if __name__ == "__main__":
    main()
