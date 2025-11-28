"""Stage A and Stage B modeling pipeline for Reddit virality analysis.

This module reproduces the two-stage approach from Weissburg et al. (2022)
for the Reddit-only slice of the project:

- Stage A estimates expected popularity given exposure/time/author features.
- Stage B explains the residual lift using title features only.

It outputs metrics, residual summaries, and optionally writes artifacts for
downstream analysis (predictions/residuals and JSON metrics).
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from itertools import combinations
from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNetCV
import lightgbm as lgb


@dataclass
class StageMetrics:
    train_rmse: float
    train_r2: float
    test_rmse: float
    test_r2: float


@dataclass
class ModelingReport:
    stage_a: StageMetrics
    stage_b: StageMetrics
    residual_summary: Dict[str, float]
    top_title_features: Dict[str, float]
    calibration: Optional[Dict[str, object]] = None
    pairwise_accuracy: Optional[float] = None
    pairwise_pairs: Optional[int] = None
    stage_b_ols: Optional[StageMetrics] = None
    stage_b_ols_features: Optional[Dict[str, float]] = None
    stage_b_tree: Optional[StageMetrics] = None
    stage_b_tree_features: Optional[Dict[str, float]] = None
    stage_b_elasticnet: Optional[StageMetrics] = None
    stage_b_elasticnet_features: Optional[Dict[str, float]] = None

    def to_dict(self) -> Dict[str, object]:
        payload = {
            "stage_a": asdict(self.stage_a),
            "stage_b": asdict(self.stage_b),
            "residual_summary": self.residual_summary,
            "top_title_features": self.top_title_features,
        }
        if self.calibration is not None:
            payload["calibration"] = self.calibration
        if self.pairwise_accuracy is not None:
            payload["pairwise_accuracy"] = self.pairwise_accuracy
        if self.pairwise_pairs is not None:
            payload["pairwise_pairs"] = self.pairwise_pairs
        if self.stage_b_ols is not None:
            payload["stage_b_ols"] = asdict(self.stage_b_ols)
        if self.stage_b_ols_features is not None:
            payload["stage_b_ols_features"] = self.stage_b_ols_features
        if self.stage_b_tree is not None:
            payload["stage_b_tree"] = asdict(self.stage_b_tree)
        if self.stage_b_tree_features is not None:
            payload["stage_b_tree_features"] = self.stage_b_tree_features
        if self.stage_b_elasticnet is not None:
            payload["stage_b_elasticnet"] = asdict(self.stage_b_elasticnet)
        if self.stage_b_elasticnet_features is not None:
            payload["stage_b_elasticnet_features"] = self.stage_b_elasticnet_features
        return payload


STAGE_A_CONTEXT_COLS = [
    "log_score_5m",
    "velocity_5_to_30m",
    "is_text_post",
    "is_image_post",
    "is_news_post",
    "is_external_link",
    "has_author",
    "author_post_count",
    "author_post_count_log",
    "author_post_count_global",
    "author_avg_score",
    "is_frequent_poster",
    "hour_of_day",
    "is_weekend",
    "is_morning",
    "is_afternoon",
    "is_evening",
    "is_night",
    "is_recent",
    "is_old",
    "is_new_collection",
    "subreddit_post_count",
    "subreddit_post_count_global",
    "subreddit_avg_score",
    "subreddit_median_score",
    "subreddit_score_std",
    "hour_sin",
    "hour_cos",
]

STAGE_B_TITLE_COLS = [
    "title_length",
    "title_words",
    "title_chars_per_word",
    "has_question",
    "has_exclamation",
    "punctuation_count",
    "all_caps_words",
    "capitalization_ratio",
    "number_count",
    "has_numbers",
    "sentiment_compound",
    "sentiment_positive",
    "sentiment_negative",
    "sentiment_neutral",
    "flesch_kincaid_grade",
    "avg_word_length",
    "type_token_ratio",
    "clickbait_keywords",
    "clickbait_patterns",
    "has_clickbait",
    "person_entities",
    "org_entities",
    "date_entities",
    "total_entities",
]

STAGE_B_INTERACTION_COLS = [
    "sentiment_compound_offpeak",
    "sentiment_negative_offpeak",
    "sentiment_positive_offpeak",
    "has_numbers_offpeak",
    "clickbait_patterns_offpeak",
    "sentiment_compound_weekend",
    "has_question_weekend",
]


def prepare_stage_dataframe(frame: pd.DataFrame) -> pd.DataFrame:
    df = frame.copy()
    df = df[df["platform"] == "reddit"].copy()
    if df.empty:
        return df

    df["created_dt"] = pd.to_datetime(df["created_timestamp"], unit="s", utc=True)
    df = df.sort_values("created_dt").reset_index(drop=True)

    def _get_numeric_series(names: Sequence[str]) -> pd.Series:
        for name in names:
            if name in df.columns:
                return pd.to_numeric(df[name], errors="coerce")
        return pd.Series(np.nan, index=df.index, dtype=float)

    score_5m = _get_numeric_series(["score_5m", "score_at_5min"]).fillna(0.0)
    score_30m = _get_numeric_series(["score_30m", "score_at_30min"]).fillna(score_5m)
    score_60m = _get_numeric_series(["score_60m", "score_at_60min", "score_at_30min"]).fillna(df["score"])

    df["score_5m"] = score_5m.clip(lower=0)
    df["score_30m"] = score_30m.clip(lower=0)
    df["score_60m"] = score_60m.clip(lower=0)
    df["log_score_5m"] = np.log1p(df["score_5m"])
    df["velocity_5_to_30m"] = (
        (df["score_30m"] - df["score_5m"]) / 25.0
    ).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    df["stage_a_target"] = np.log1p(df["score_60m"])

    if "author_post_count_log" not in df.columns and "author_post_count" in df.columns:
        df["author_post_count_log"] = np.log1p(df["author_post_count"].clip(lower=0))
    df["is_new_collection"] = df.get("is_new_collection", 0).fillna(0).astype(float)
    df["author_post_count_global"] = df.get("author_post_count_global", 0).fillna(0)
    df["subreddit_post_count"] = df.get("subreddit_post_count", 0).fillna(0)
    df["subreddit_post_count_global"] = df.get("subreddit_post_count_global", 0).fillna(0)
    df["subreddit_score_std"] = df.get("subreddit_score_std", 0).fillna(0)
    df["hour_sin"] = df.get("hour_sin", 0.0)
    df["hour_cos"] = df.get("hour_cos", 0.0)

    score_mean = float(df["score"].mean()) if not df["score"].empty else 0.0
    for col, fallback in [
        ("subreddit_avg_score", score_mean),
        ("subreddit_median_score", score_mean),
        ("author_avg_score", score_mean),
    ]:
        if col in df.columns:
            df[col] = df[col].fillna(fallback)
        else:
            df[col] = fallback

    return df


def _compute_stage_b_lift(group: pd.DataFrame) -> pd.Series:
    return pd.Series(
        {
            "actual_mean": group["stage_a_residual"].mean(),
            "pred_mean": group["stage_b_pred"].mean(),
            "count": float(len(group)),
        }
    )


def _fit_stage_models(
    df: pd.DataFrame,
    train_indices: Sequence[int],
    test_indices: Sequence[int],
    residual_output_path: Optional[Path] = None,
    metrics_output_path: Optional[Path] = None,
) -> ModelingReport:
    if len(train_indices) == 0 or len(test_indices) == 0:
        raise ValueError("train_indices and test_indices must be non-empty")

    frame = df.copy()
    train_idx = np.asarray(train_indices, dtype=int)
    test_idx = np.asarray(test_indices, dtype=int)

    train_df = frame.iloc[train_idx].copy()
    test_df = frame.iloc[test_idx].copy()

    Xa_train = _build_stage_a_matrix(train_df)
    Xa_test = _build_stage_a_matrix(test_df)
    Xa_test = Xa_test.reindex(columns=Xa_train.columns, fill_value=0.0)

    Xa_train_const = sm.add_constant(Xa_train, has_constant="add")
    Xa_test_const = sm.add_constant(Xa_test, has_constant="add")
    Xa_test_const = Xa_test_const.reindex(columns=Xa_train_const.columns, fill_value=1.0)

    stage_a_model = sm.OLS(train_df["stage_a_target"], Xa_train_const).fit()

    y_train_pred = stage_a_model.predict(Xa_train_const)
    y_test_pred = stage_a_model.predict(Xa_test_const)

    stage_a_metrics = StageMetrics(
        train_rmse=float(
            np.sqrt(mean_squared_error(train_df["stage_a_target"], y_train_pred))
        ),
        train_r2=float(r2_score(train_df["stage_a_target"], y_train_pred)),
        test_rmse=float(
            np.sqrt(mean_squared_error(test_df["stage_a_target"], y_test_pred))
        ),
        test_r2=float(r2_score(test_df["stage_a_target"], y_test_pred)),
    )

    Xa_full = _build_stage_a_matrix(frame)
    Xa_full = Xa_full.reindex(columns=Xa_train.columns, fill_value=0.0)
    Xa_full_const = sm.add_constant(Xa_full, has_constant="add")
    Xa_full_const = Xa_full_const.reindex(columns=Xa_train_const.columns, fill_value=1.0)

    frame["stage_a_pred"] = stage_a_model.predict(Xa_full_const)
    frame["stage_a_residual"] = frame["stage_a_target"] - frame["stage_a_pred"]

    train_df = frame.iloc[train_idx].copy()
    test_df = frame.iloc[test_idx].copy()

    for col in STAGE_B_TITLE_COLS:
        if col not in frame.columns:
            frame[col] = 0.0

    frame["is_off_peak"] = frame["hour_of_day"].isin(list(range(0, 6)) + [22, 23]).astype(float)
    frame["is_weekend_flag"] = (frame["day_of_week"] >= 5).astype(float)
    frame["sentiment_compound_offpeak"] = frame.get("sentiment_compound", 0.0) * frame["is_off_peak"]
    frame["sentiment_negative_offpeak"] = frame.get("sentiment_negative", 0.0) * frame["is_off_peak"]
    frame["sentiment_positive_offpeak"] = frame.get("sentiment_positive", 0.0) * frame["is_off_peak"]
    frame["has_numbers_offpeak"] = frame.get("has_numbers", 0.0) * frame["is_off_peak"]
    frame["clickbait_patterns_offpeak"] = frame.get("clickbait_patterns", 0.0) * frame["is_off_peak"]
    frame["sentiment_compound_weekend"] = frame.get("sentiment_compound", 0.0) * frame["is_weekend_flag"]
    frame["has_question_weekend"] = frame.get("has_question", 0.0) * frame["is_weekend_flag"]

    residual_summary = {
        "residual_mean": float(frame["stage_a_residual"].mean()),
        "residual_std": float(frame["stage_a_residual"].std()),
        "residual_skew": float(frame["stage_a_residual"].skew()),
    }

    stage_b_feature_cols = STAGE_B_TITLE_COLS + STAGE_B_INTERACTION_COLS
    Xb_train_raw = train_df.reindex(columns=stage_b_feature_cols, fill_value=0.0).astype(float)
    Xb_test_raw = test_df.reindex(columns=stage_b_feature_cols, fill_value=0.0).astype(float)
    y_stage_b_train = train_df["stage_a_residual"]
    y_stage_b_test = test_df["stage_a_residual"]

    scaler = StandardScaler()
    Xb_train = pd.DataFrame(
        scaler.fit_transform(Xb_train_raw),
        columns=stage_b_feature_cols,
        index=Xb_train_raw.index,
    )
    Xb_test = pd.DataFrame(
        scaler.transform(Xb_test_raw),
        columns=stage_b_feature_cols,
        index=Xb_test_raw.index,
    )

    variance_mask = Xb_train.var(axis=0) > 1e-8
    active_cols = list(Xb_train.columns[variance_mask])
    if not active_cols:
        active_cols = list(Xb_train.columns)
    Xb_train = Xb_train[active_cols]
    Xb_test = Xb_test[active_cols]

    elastic_net = ElasticNetCV(
        l1_ratio=[0.05, 0.25, 0.5, 0.75, 0.9],
        alphas=np.logspace(-3, 2, 30),
        cv=5,
        random_state=42,
        n_jobs=-1,
        max_iter=20000,
        tol=1e-4,
    )
    elastic_net.fit(Xb_train, y_stage_b_train)

    stage_b_train_pred = elastic_net.predict(Xb_train)
    stage_b_test_pred = elastic_net.predict(Xb_test)

    stage_b_metrics = StageMetrics(
        train_rmse=float(
            np.sqrt(mean_squared_error(y_stage_b_train, stage_b_train_pred))
        ),
        train_r2=float(r2_score(y_stage_b_train, stage_b_train_pred)),
        test_rmse=float(
            np.sqrt(mean_squared_error(y_stage_b_test, stage_b_test_pred))
        ),
        test_r2=float(r2_score(y_stage_b_test, stage_b_test_pred)),
    )

    Xb_full_raw = frame.reindex(columns=stage_b_feature_cols, fill_value=0.0).astype(float)
    Xb_full = pd.DataFrame(
        scaler.transform(Xb_full_raw),
        columns=stage_b_feature_cols,
        index=frame.index,
    )
    Xb_full = Xb_full[active_cols]
    frame["stage_b_pred"] = elastic_net.predict(Xb_full)
    frame.loc[test_df.index, "stage_b_pred"] = stage_b_test_pred

    df_stage_b_eval = frame.loc[test_df.index, [
        "subreddit",
        "hour_of_day",
        "stage_a_residual",
        "stage_b_pred",
    ]].copy()
    stage_b_lift = (
        df_stage_b_eval.groupby(["subreddit", "hour_of_day"], dropna=True)
        .apply(_compute_stage_b_lift)
        .reset_index()
    )

    en_coef_series = pd.Series(elastic_net.coef_, index=active_cols)
    en_coef_series = en_coef_series[en_coef_series != 0]
    en_coef_series = en_coef_series.sort_values(key=np.abs, ascending=False)
    top_title_features = {
        feature: float(weight) for feature, weight in en_coef_series.head(10).items()
    }

    Xb_train_const = sm.add_constant(Xb_train, has_constant="add")
    Xb_test_const = sm.add_constant(Xb_test, has_constant="add")
    Xb_test_const = Xb_test_const.reindex(columns=Xb_train_const.columns, fill_value=1.0)

    stage_b_ols_model = sm.OLS(y_stage_b_train, Xb_train_const).fit()
    ols_train_pred = stage_b_ols_model.predict(Xb_train_const)
    ols_test_pred = stage_b_ols_model.predict(Xb_test_const)
    stage_b_ols_metrics = StageMetrics(
        train_rmse=float(np.sqrt(mean_squared_error(y_stage_b_train, ols_train_pred))),
        train_r2=float(r2_score(y_stage_b_train, ols_train_pred)),
        test_rmse=float(np.sqrt(mean_squared_error(y_stage_b_test, ols_test_pred))),
        test_r2=float(r2_score(y_stage_b_test, ols_test_pred)),
    )
    ols_coef_series = stage_b_ols_model.params.drop("const")
    ols_coef_series = ols_coef_series.sort_values(key=np.abs, ascending=False)
    top_ols_features = {
        feature: float(weight) for feature, weight in ols_coef_series.head(10).items()
    }

    tree_model = lgb.LGBMRegressor(
        random_state=42,
        n_estimators=400,
        learning_rate=0.05,
        num_leaves=15,
        min_child_samples=60,
        subsample=0.7,
        colsample_bytree=0.7,
        reg_alpha=0.1,
        reg_lambda=0.5,
        n_jobs=-1,
    )
    tree_model.fit(
        Xb_train,
        y_stage_b_train,
        eval_set=[(Xb_test, y_stage_b_test)],
        eval_metric="rmse",
        callbacks=[lgb.early_stopping(30), lgb.log_evaluation(0)],
    )
    best_iteration = tree_model.best_iteration_
    if best_iteration is None:
        best_iteration = tree_model.n_estimators_
    tree_train_pred = tree_model.predict(Xb_train, num_iteration=best_iteration)
    tree_test_pred = tree_model.predict(Xb_test, num_iteration=best_iteration)

    stage_b_tree_metrics = StageMetrics(
        train_rmse=float(np.sqrt(mean_squared_error(y_stage_b_train, tree_train_pred))),
        train_r2=float(r2_score(y_stage_b_train, tree_train_pred)),
        test_rmse=float(np.sqrt(mean_squared_error(y_stage_b_test, tree_test_pred))),
        test_r2=float(r2_score(y_stage_b_test, tree_test_pred)),
    )

    feature_importance = tree_model.feature_importances_
    importance_pairs = sorted(
        zip(stage_b_feature_cols, feature_importance),
        key=lambda x: x[1],
        reverse=True,
    )
    top_tree_features = {
        feature: float(importance) for feature, importance in importance_pairs[:10]
    }

    en_train_pred = stage_b_train_pred
    en_test_pred = stage_b_test_pred
    stage_b_elasticnet_metrics = stage_b_metrics
    top_elasticnet_features = top_title_features

    stage_a_hour_calibration = (
        frame.groupby("hour_of_day")["stage_a_residual"].mean().dropna().to_dict()
        if "hour_of_day" in frame.columns
        else {}
    )
    stage_a_hour_calibration = {
        int(hour): float(value)
        for hour, value in stage_a_hour_calibration.items()
        if not pd.isna(hour) and not pd.isna(value)
    }
    stage_a_subreddit_calibration = (
        frame.groupby("subreddit")["stage_a_residual"].mean().dropna().to_dict()
        if "subreddit" in frame.columns
        else {}
    )
    stage_a_subreddit_calibration = {
        str(sub): float(value)
        for sub, value in stage_a_subreddit_calibration.items()
        if sub is not None and not pd.isna(value)
    }
    calibration: Dict[str, object] = {
        "stage_a": {
            "hour_of_day": stage_a_hour_calibration,
            "subreddit": stage_a_subreddit_calibration,
        }
    }

    stage_b_calibration_records = (
        stage_b_lift.to_dict(orient="records") if not stage_b_lift.empty else []
    )
    if stage_b_calibration_records:
        calibration["stage_b"] = stage_b_calibration_records

    df_test_pairs = frame.loc[test_df.index, [
        "subreddit",
        "created_dt",
        "stage_a_residual",
    ]].copy()
    df_test_pairs["stage_b_pred"] = frame.loc[test_df.index, "stage_b_pred"]
    df_test_pairs["hour_bucket"] = df_test_pairs["created_dt"].dt.floor("h")
    pairwise_correct = 0
    pairwise_total = 0
    for (_, group) in df_test_pairs.groupby(["subreddit", "hour_bucket"], dropna=True):
        values = group[["stage_a_residual", "stage_b_pred"]].to_numpy()
        n = len(values)
        if n < 2:
            continue
        for i, j in combinations(range(n), 2):
            actual_diff = values[i, 0] - values[j, 0]
            pred_diff = values[i, 1] - values[j, 1]
            if actual_diff == 0 or pred_diff == 0:
                continue
            pairwise_total += 1
            if actual_diff * pred_diff > 0:
                pairwise_correct += 1
    pairwise_accuracy = pairwise_correct / pairwise_total if pairwise_total else None

    if residual_output_path is not None:
        residual_output_path.parent.mkdir(parents=True, exist_ok=True)
        frame[[
            "post_id",
            "created_dt",
            "score",
            "hour_of_day",
            "day_of_week",
            "is_off_peak",
            "is_weekend_flag",
            "stage_a_target",
            "stage_a_pred",
            "stage_a_residual",
            "score_5m",
            "score_30m",
            "score_60m",
            "velocity_5_to_30m",
            "stage_b_pred",
        ]].to_parquet(residual_output_path, index=False)

    report = ModelingReport(
        stage_a=stage_a_metrics,
        stage_b=stage_b_metrics,
        residual_summary=residual_summary,
        top_title_features=top_title_features,
        calibration=calibration,
        pairwise_accuracy=pairwise_accuracy,
        pairwise_pairs=pairwise_total if pairwise_total else None,
        stage_b_ols=stage_b_ols_metrics,
        stage_b_ols_features=top_ols_features,
        stage_b_tree=stage_b_tree_metrics,
        stage_b_tree_features=top_tree_features,
        stage_b_elasticnet=stage_b_elasticnet_metrics,
        stage_b_elasticnet_features=top_elasticnet_features,
    )

    if metrics_output_path is not None:
        metrics_output_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_output_path.write_text(json.dumps(report.to_dict(), indent=2))

    return report


def _build_stage_a_matrix(frame: pd.DataFrame) -> pd.DataFrame:
    """Construct Stage-A design matrix with exposure/time controls."""
    base = frame.reindex(columns=STAGE_A_CONTEXT_COLS, fill_value=0.0).astype(float)
    hour_dummies = (
        pd.get_dummies(frame["hour_of_day"], prefix="hour", drop_first=True)
        .astype(float)
    )
    dow_dummies = (
        pd.get_dummies(frame["day_of_week"], prefix="dow", drop_first=True)
        .astype(float)
    )
    pieces = [base, hour_dummies, dow_dummies]
    if "subreddit" in frame.columns:
        subreddit_dummies = (
            pd.get_dummies(frame["subreddit"], prefix="subreddit", drop_first=True)
            .astype(float)
        )
        pieces.append(subreddit_dummies)
    return pd.concat(pieces, axis=1)


def run_stage_modeling(
    data_path: Path,
    residual_output_path: Optional[Path] = None,
    metrics_output_path: Optional[Path] = None,
    split_quantile: float = 0.7,
) -> ModelingReport:
    if not 0 < split_quantile < 1:
        raise ValueError("split_quantile must be between 0 and 1")

    df_raw = pd.read_parquet(data_path)
    df = prepare_stage_dataframe(df_raw)
    if df.empty:
        raise ValueError("No Reddit records available for modeling")

    split_time = df["created_dt"].quantile(split_quantile)
    train_mask = df["created_dt"] <= split_time
    test_mask = ~train_mask
    if not test_mask.any():
        raise ValueError("Test split is empty; adjust split_quantile")

    train_idx = np.where(train_mask)[0]
    test_idx = np.where(test_mask)[0]

    return _fit_stage_models(
        df,
        train_idx,
        test_idx,
        residual_output_path,
        metrics_output_path,
    )


def run_stage_modeling_with_indices(
    prepared_df: pd.DataFrame,
    train_indices: Sequence[int],
    test_indices: Sequence[int],
    residual_output_path: Optional[Path] = None,
    metrics_output_path: Optional[Path] = None,
    write_artifacts: bool = False,
) -> ModelingReport:
    return _fit_stage_models(
        prepared_df,
        train_indices,
        test_indices,
        residual_output_path if write_artifacts else None,
        metrics_output_path if write_artifacts else None,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Stage A/B modeling on Reddit feature table."
    )
    parser.add_argument(
        "--data",
        default="data/reddit_features.parquet",
        type=Path,
        help="Path to feature parquet file",
    )
    parser.add_argument(
        "--residual-output",
        default="data/reddit_stage_predictions.parquet",
        type=Path,
        help="Where to save Stage-A predictions/residuals",
    )
    parser.add_argument(
        "--metrics-output",
        default="docs/stage_metrics.json",
        type=Path,
        help="Where to save modeling metrics JSON",
    )
    parser.add_argument(
        "--split-quantile",
        type=float,
        default=0.7,
        help="Time-based quantile for train/test split",
    )
    args = parser.parse_args()

    report = run_stage_modeling(
        data_path=args.data,
        residual_output_path=args.residual_output,
        metrics_output_path=args.metrics_output,
        split_quantile=args.split_quantile,
    )

    print(json.dumps(report.to_dict(), indent=2))


if __name__ == "__main__":
    main()
