#!/usr/bin/env python3
"""Run penalized Stage A/B models inspired by Weissburg et al. (2022).

Run with::

    python -m src.cli.run_stage_penalized --data data/features.parquet

This CLI leaves the core pipeline untouched while fitting ElasticNet-based
Stage A (exposure controls) and Stage B (headline lift) models in parallel for
side-by-side comparison.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.models.stage_penalized import (
    PenalizedConfig,
    run_penalized_stage_models,
)


def _parse_float_sequence(raw: str) -> tuple[float, ...]:
    parts = [item.strip() for item in raw.split(",") if item.strip()]
    return tuple(float(item) for item in parts)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Penalized Stage A/B modeling (Weissburg-style replica)",
    )
    parser.add_argument(
        "--data",
        default="data/features.parquet",
        type=Path,
        help="Input feature parquet (same as Stage A/B baseline).",
    )
    parser.add_argument(
        "--split-quantile",
        default=0.7,
        type=float,
        help="Temporal split quantile for train/test separation.",
    )
    parser.add_argument(
        "--stage-a-l1-ratios",
        default="0.1,0.3,0.5,0.7,0.9",
        help="Comma-separated l1_ratio grid for Stage A ElasticNetCV.",
    )
    parser.add_argument(
        "--stage-b-l1-ratios",
        default="0.1,0.3,0.5,0.7,0.9",
        help="Comma-separated l1_ratio grid for Stage B ElasticNetCV.",
    )
    parser.add_argument(
        "--stage-a-alphas",
        default="",
        help="Optional comma-separated alpha grid for Stage A (blank = auto).",
    )
    parser.add_argument(
        "--stage-b-alphas",
        default="",
        help="Optional comma-separated alpha grid for Stage B (blank = auto).",
    )
    parser.add_argument(
        "--cv-folds",
        default=5,
        type=int,
        help="Number of CV folds used by ElasticNetCV.",
    )
    parser.add_argument(
        "--max-iter",
        default=8000,
        type=int,
        help="Max iterations for both Stage A and Stage B solvers.",
    )
    parser.add_argument(
        "--output-parquet",
        default="outputs/title_lift/stage_penalized_outputs.parquet",
        type=Path,
        help="Location to write predictions/residuals (optional).",
    )
    parser.add_argument(
        "--output-json",
        default="outputs/title_lift/stage_penalized_metrics.json",
        type=Path,
        help="Location to write metrics report JSON (optional).",
    )
    parser.add_argument(
        "--top-k",
        default=15,
        type=int,
        help="Top feature count reported per direction (Stage A/B).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    stage_a_l1 = _parse_float_sequence(args.stage_a_l1_ratios) or (0.5,)
    stage_b_l1 = _parse_float_sequence(args.stage_b_l1_ratios) or (0.5,)
    stage_a_alphas = _parse_float_sequence(args.stage_a_alphas) if args.stage_a_alphas else None
    stage_b_alphas = _parse_float_sequence(args.stage_b_alphas) if args.stage_b_alphas else None

    config = PenalizedConfig(
        split_quantile=args.split_quantile,
        stage_a_l1_ratios=stage_a_l1,
        stage_a_alphas=stage_a_alphas,
        stage_a_max_iter=args.max_iter,
        stage_b_l1_ratios=stage_b_l1,
        stage_b_alphas=stage_b_alphas,
        stage_b_max_iter=args.max_iter,
        cv_folds=args.cv_folds,
        top_k=args.top_k,
    )

    report = run_penalized_stage_models(
        data_path=args.data,
        output_parquet=args.output_parquet,
        metrics_output=args.output_json,
        config=config,
    )

    print("Penalized Stage A metrics:")
    print(f"  Train RMSE: {report.stage_a.train_rmse:.3f}")
    print(f"  Test RMSE:  {report.stage_a.test_rmse:.3f}")
    print(f"  Train MAE:  {report.stage_a.train_mae:.3f}")
    print(f"  Test MAE:   {report.stage_a.test_mae:.3f}")
    print(f"  Train R^2:  {report.stage_a.train_r2:.3f}")
    print(f"  Test R^2:   {report.stage_a.test_r2:.3f}")
    print()
    print("Penalized Stage B metrics:")
    print(f"  Train RMSE: {report.stage_b.train_rmse:.3f}")
    print(f"  Test RMSE:  {report.stage_b.test_rmse:.3f}")
    print(f"  Train MAE:  {report.stage_b.train_mae:.3f}")
    print(f"  Test MAE:   {report.stage_b.test_mae:.3f}")
    print(f"  Train R^2:  {report.stage_b.train_r2:.3f}")
    print(f"  Test R^2:   {report.stage_b.test_r2:.3f}")

    if report.stage_a_top_positive:
        print("Top Stage A positive coefficients:")
        for item in report.stage_a_top_positive[:5]:
            print(f"    + {item['feature']}: {item['weight']:.4f}")
    if report.stage_a_top_negative:
        print("Top Stage A negative coefficients:")
        for item in report.stage_a_top_negative[:5]:
            print(f"    - {item['feature']}: {item['weight']:.4f}")
    if report.stage_b_top_positive:
        print("Top Stage B positive coefficients:")
        for item in report.stage_b_top_positive[:5]:
            print(f"    + {item['feature']}: {item['weight']:.4f}")
    if report.stage_b_top_negative:
        print("Top Stage B negative coefficients:")
        for item in report.stage_b_top_negative[:5]:
            print(f"    - {item['feature']}: {item['weight']:.4f}")


if __name__ == "__main__":
    main()
