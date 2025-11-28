"""Penalized Stage A/B modeling inspired by Weissburg et al. (2022).

This module provides a parallel implementation that mirrors the paper's
regularized workflow while leaving the primary OLS-based pipeline untouched.
It applies ElasticNet regression to (1) exposure/context features (Stage A)

and (2) headline-only features on the residuals (Stage B). The goal is to
compare how penalization behaves relative to the baseline without editing
existing artifacts.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNetCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .stage_modeling import (
    STAGE_A_CONTEXT_COLS,
    STAGE_B_INTERACTION_COLS,
    STAGE_B_TITLE_COLS,
    prepare_stage_dataframe,
)


@dataclass
class StageStats:
    train_rmse: float
    test_rmse: float
    train_mae: float
    test_mae: float
    train_r2: float
    test_r2: float


@dataclass
class PenalizedConfig:
    split_quantile: float = 0.7
    stage_a_l1_ratios: Sequence[float] = (0.1, 0.3, 0.5, 0.7, 0.9)
    stage_a_alphas: Optional[Sequence[float]] = None
    stage_a_max_iter: int = 8000
    stage_b_l1_ratios: Sequence[float] = (0.1, 0.3, 0.5, 0.7, 0.9)
    stage_b_alphas: Optional[Sequence[float]] = None
    stage_b_max_iter: int = 8000
    cv_folds: int = 5
    n_jobs: int = -1
    top_k: int = 15


@dataclass
class PenalizedReport:
    stage_a: StageStats
    stage_b: StageStats
    split_time: str
    config: Dict[str, object]
    stage_a_top_positive: List[Dict[str, float]]
    stage_a_top_negative: List[Dict[str, float]]
    stage_b_top_positive: List[Dict[str, float]]
    stage_b_top_negative: List[Dict[str, float]]

    def to_dict(self) -> Dict[str, object]:
        payload = asdict(self)
        payload["config"] = dict(payload["config"])
        return payload


def _build_stage_a_matrix(df: pd.DataFrame) -> pd.DataFrame:
    base = df.reindex(columns=STAGE_A_CONTEXT_COLS, fill_value=0.0).astype(float)
    hour_dummies = pd.get_dummies(df.get("hour_of_day"), prefix="hour", drop_first=True)
    dow_dummies = pd.get_dummies(df.get("day_of_week"), prefix="dow", drop_first=True)
    subreddit_dummies = pd.get_dummies(df.get("subreddit"), prefix="sub", drop_first=True)
    pieces = [base]
    if not hour_dummies.empty:
        pieces.append(hour_dummies.astype(float))
    if not dow_dummies.empty:
        pieces.append(dow_dummies.astype(float))
    if not subreddit_dummies.empty:
        pieces.append(subreddit_dummies.astype(float))
    matrix = pd.concat(pieces, axis=1).fillna(0.0)
    return matrix


def _build_stage_b_matrix(df: pd.DataFrame) -> pd.DataFrame:
    cols: List[str] = list(STAGE_B_TITLE_COLS) + list(STAGE_B_INTERACTION_COLS)
    return df.reindex(columns=cols, fill_value=0.0).astype(float)


def _make_pipeline(l1_ratios: Sequence[float], alphas: Optional[Sequence[float]], max_iter: int, cv_folds: int, n_jobs: int) -> Pipeline:
    model = ElasticNetCV(
        l1_ratio=list(l1_ratios),
        alphas=None if alphas is None else list(alphas),
        max_iter=max_iter,
        cv=cv_folds,
        n_jobs=n_jobs,
    )
    return Pipeline([
        ("scaler", StandardScaler(with_mean=False)),
        ("model", model),
    ])


def _summarize_coefficients(
    model: Pipeline,
    feature_names: Iterable[str],
    top_k: int,
) -> Tuple[List[Dict[str, float]], List[Dict[str, float]]]:
    elastic: ElasticNetCV = model.named_steps["model"]
    coefs = elastic.coef_
    names = list(feature_names)
    pairs = [
        (name, coef) for name, coef in zip(names, coefs) if coef != 0 and not np.isnan(coef)
    ]
    if not pairs:
        return ([], [])
    pos_pairs = sorted(
        [(name, coef) for name, coef in pairs if coef > 0], key=lambda x: x[1], reverse=True
    )[:top_k]
    neg_pairs = sorted(
        [(name, coef) for name, coef in pairs if coef < 0], key=lambda x: x[1]
    )[:top_k]
    positive = [
        {"feature": name, "weight": float(weight)}
        for name, weight in pos_pairs
    ]
    negative = [
        {"feature": name, "weight": float(weight)}
        for name, weight in neg_pairs
    ]
    return (positive, negative)


def _metrics(y_true_train: np.ndarray, y_pred_train: np.ndarray, y_true_test: np.ndarray, y_pred_test: np.ndarray) -> StageStats:
    train_rmse = float(np.sqrt(mean_squared_error(y_true_train, y_pred_train)))
    test_rmse = float(np.sqrt(mean_squared_error(y_true_test, y_pred_test)))
    train_mae = float(mean_absolute_error(y_true_train, y_pred_train))
    test_mae = float(mean_absolute_error(y_true_test, y_pred_test))
    train_r2 = float(r2_score(y_true_train, y_pred_train))
    test_r2 = float(r2_score(y_true_test, y_pred_test))
    return StageStats(
        train_rmse=train_rmse,
        test_rmse=test_rmse,
        train_mae=train_mae,
        test_mae=test_mae,
        train_r2=train_r2,
        test_r2=test_r2,
    )


def run_penalized_stage_models(
    data_path: Path,
    output_parquet: Optional[Path] = None,
    metrics_output: Optional[Path] = None,
    config: Optional[PenalizedConfig] = None,
) -> PenalizedReport:
    cfg = config or PenalizedConfig()
    df_raw = pd.read_parquet(data_path)
    df = prepare_stage_dataframe(df_raw)
    if df.empty:
        raise ValueError("No Reddit records available for penalized modeling")

    split_time = df["created_dt"].quantile(cfg.split_quantile)
    train_mask = df["created_dt"] <= split_time

    X_a = _build_stage_a_matrix(df)
    y_a = df["stage_a_target"].to_numpy(dtype=float)

    stage_a_model = _make_pipeline(
        l1_ratios=cfg.stage_a_l1_ratios,
        alphas=cfg.stage_a_alphas,
        max_iter=cfg.stage_a_max_iter,
        cv_folds=cfg.cv_folds,
        n_jobs=cfg.n_jobs,
    )
    stage_a_model.fit(X_a.loc[train_mask], y_a[train_mask])
    y_a_train_pred = stage_a_model.predict(X_a.loc[train_mask])
    y_a_test_pred = stage_a_model.predict(X_a.loc[~train_mask])
    stage_a_metrics = _metrics(
        y_true_train=y_a[train_mask],
        y_pred_train=y_a_train_pred,
        y_true_test=y_a[~train_mask],
        y_pred_test=y_a_test_pred,
    )
    stage_a_positive, stage_a_negative = _summarize_coefficients(
        stage_a_model,
        feature_names=X_a.columns,
        top_k=cfg.top_k,
    )
    stage_a_full_pred = stage_a_model.predict(X_a)
    df["stage_a_penalized_pred"] = stage_a_full_pred
    df["stage_a_penalized_residual"] = df["stage_a_target"] - df["stage_a_penalized_pred"]

    X_b = _build_stage_b_matrix(df)
    y_b = df["stage_a_penalized_residual"].to_numpy(dtype=float)
    stage_b_model = _make_pipeline(
        l1_ratios=cfg.stage_b_l1_ratios,
        alphas=cfg.stage_b_alphas,
        max_iter=cfg.stage_b_max_iter,
        cv_folds=cfg.cv_folds,
        n_jobs=cfg.n_jobs,
    )
    stage_b_model.fit(X_b.loc[train_mask], y_b[train_mask])
    y_b_train_pred = stage_b_model.predict(X_b.loc[train_mask])
    y_b_test_pred = stage_b_model.predict(X_b.loc[~train_mask])
    stage_b_metrics = _metrics(
        y_true_train=y_b[train_mask],
        y_pred_train=y_b_train_pred,
        y_true_test=y_b[~train_mask],
        y_pred_test=y_b_test_pred,
    )
    stage_b_positive, stage_b_negative = _summarize_coefficients(
        stage_b_model,
        feature_names=X_b.columns,
        top_k=cfg.top_k,
    )
    df["stage_b_penalized_pred"] = stage_b_model.predict(X_b)

    if output_parquet is not None:
        output_parquet.parent.mkdir(parents=True, exist_ok=True)
        df[[
            "post_id",
            "created_dt",
            "stage_a_target",
            "stage_a_penalized_pred",
            "stage_a_penalized_residual",
            "stage_b_penalized_pred",
        ]].to_parquet(output_parquet, index=False)

    report = PenalizedReport(
        stage_a=stage_a_metrics,
        stage_b=stage_b_metrics,
        split_time=str(split_time),
        config=asdict(cfg),
        stage_a_top_positive=stage_a_positive,
        stage_a_top_negative=stage_a_negative,
        stage_b_top_positive=stage_b_positive,
        stage_b_top_negative=stage_b_negative,
    )

    if metrics_output is not None:
        metrics_output.parent.mkdir(parents=True, exist_ok=True)
        metrics_output.write_text(json.dumps(report.to_dict(), indent=2))

    return report


__all__ = [
    "PenalizedConfig",
    "PenalizedReport",
    "StageStats",
    "run_penalized_stage_models",
]
