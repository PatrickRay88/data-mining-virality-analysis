"""Stage B enhancement experiments.

This module provides utilities for experimenting with richer title-only
models without modifying the main Stage A/B pipeline. The helper focuses on
feature augmentation (hand-crafted heuristics plus TF-IDF n-grams) and uses an
ElasticNet with internal cross-validation to stay close to the interpretable
setup in Weissburg et al. (2022).
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import ElasticNetCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


TITLE_REGEXES: Dict[str, re.Pattern[str]] = {
    "wh_start": re.compile(r"^(?:who|what|when|where|why|how|is|are|can|should)\\b", re.I),
    "contains_colon": re.compile(r":"),
    "contains_dash": re.compile(r"-"),
    "contains_digit": re.compile(r"\\d"),
    "contains_year": re.compile(r"20[0-9]{2}|19[0-9]{2}"),
    "all_caps_word": re.compile(r"\\b[A-Z]{4,}\\b"),
}


@dataclass
class StageBExperimentConfig:
    split_quantile: float = 0.7
    tfidf_max_features: int = 500
    tfidf_min_df: int = 5
    tfidf_ngram_range: Sequence[int] = (1, 2)
    l1_ratios: Sequence[float] = (0.1, 0.3, 0.5, 0.7, 0.9)
    alphas: Sequence[float] | None = None
    max_iter: int = 5000
    random_state: int = 42
    use_svd_embeddings: bool = False
    svd_components: int = 100


@dataclass
class StageBExperimentResult:
    train_rmse: float
    test_rmse: float
    train_mae: float
    test_mae: float
    train_r2: float
    test_r2: float
    n_train: int
    n_test: int
    split_time: str
    top_positive_features: List[Dict[str, float]]
    top_negative_features: List[Dict[str, float]]
    config: Dict[str, object]

    def to_dict(self) -> Dict[str, object]:
        payload = asdict(self)
        payload["config"] = dict(payload["config"])
        return payload


def _augment_handcrafted_features(frame: pd.DataFrame) -> pd.DataFrame:
    df = frame.copy()
    lowered = df["title"].fillna("")
    df["title_lower"] = lowered.str.lower()
    df["title_token_len"] = df["title_lower"].str.split().apply(len)
    df["title_char_len"] = df["title_lower"].str.len()
    df["title_avg_token_len"] = (
        df["title_char_len"] / df["title_token_len"].replace(0, np.nan)
    ).fillna(0.0)

    for name, pattern in TITLE_REGEXES.items():
        df[f"feat_{name}"] = lowered.str.contains(pattern).astype(float)

    df["feat_title_exc"] = lowered.str.contains(r"!").astype(float)
    df["feat_title_question"] = lowered.str.contains(r"\\?").astype(float)
    df["feat_title_quote"] = lowered.str.contains(r"\"").astype(float)
    df["feat_token_shout_ratio"] = (
        df["all_caps_words"].fillna(0.0)
        / df["title_token_len"].replace(0, np.nan)
    ).fillna(0.0)

    return df


def _build_pipeline(
    numeric_cols: Sequence[str],
    config: StageBExperimentConfig,
) -> Pipeline:
    tfidf = TfidfVectorizer(
        ngram_range=tuple(config.tfidf_ngram_range),
        max_features=config.tfidf_max_features,
        min_df=config.tfidf_min_df,
    )
    numeric_scaler = StandardScaler(with_mean=False)
    if config.use_svd_embeddings:
        text_transformer: Pipeline = Pipeline(
            steps=[
                ("tfidf", tfidf),
                (
                    "svd",
                    TruncatedSVD(
                        n_components=config.svd_components,
                        random_state=config.random_state,
                    ),
                ),
            ]
        )
    else:
        text_transformer = tfidf
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_scaler, list(numeric_cols)),
            ("text", text_transformer, "title"),
        ],
        remainder="drop",
    )

    regressor = ElasticNetCV(
        l1_ratio=list(config.l1_ratios),
        alphas=None if config.alphas is None else list(config.alphas),
        max_iter=config.max_iter,
        cv=5,
        n_jobs=None,
        random_state=config.random_state,
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regressor", regressor),
        ]
    )
    return model


def _compute_metrics(
    y_true_train: np.ndarray,
    y_pred_train: np.ndarray,
    y_true_test: np.ndarray,
    y_pred_test: np.ndarray,
) -> Dict[str, float]:
    metrics = {
        "train_rmse": math.sqrt(mean_squared_error(y_true_train, y_pred_train)),
        "test_rmse": math.sqrt(mean_squared_error(y_true_test, y_pred_test)),
        "train_mae": mean_absolute_error(y_true_train, y_pred_train),
        "test_mae": mean_absolute_error(y_true_test, y_pred_test),
        "train_r2": r2_score(y_true_train, y_pred_train),
        "test_r2": r2_score(y_true_test, y_pred_test),
    }
    return metrics


def _extract_feature_importance(model: Pipeline, top_k: int = 15) -> Dict[str, List[Dict[str, float]]]:
    preprocessor: ColumnTransformer = model.named_steps["preprocessor"]
    regressor: ElasticNetCV = model.named_steps["regressor"]
    feature_names = preprocessor.get_feature_names_out()
    coefficients = regressor.coef_

    pairs = list(zip(feature_names, coefficients))
    pairs = [pair for pair in pairs if not np.isnan(pair[1]) and pair[1] != 0]
    if not pairs:
        return {"positive": [], "negative": []}

    sorted_pairs = sorted(pairs, key=lambda x: x[1])
    negative = [
        {"feature": name, "weight": float(weight)}
        for name, weight in sorted_pairs[:top_k]
    ]
    positive = [
        {"feature": name, "weight": float(weight)}
        for name, weight in reversed(sorted_pairs[-top_k:])
    ]
    return {"positive": positive, "negative": negative}


def _prepare_dataset(
    stage_outputs_path: Path,
    features_path: Path,
    config: StageBExperimentConfig,
) -> tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    stage_df = pd.read_parquet(stage_outputs_path)
    raw_features = pd.read_parquet(features_path)
    columns_needed = {"post_id"}
    if "created_dt" in raw_features.columns:
        columns_needed.add("created_dt")
    elif "created_timestamp" in raw_features.columns:
        columns_needed.add("created_timestamp")
    else:
        raise KeyError("features parquet must contain created_dt or created_timestamp column")

    features_df = raw_features[list(columns_needed)].copy()
    features_df["post_id"] = features_df["post_id"].astype(str)
    if "created_dt" in features_df.columns:
        features_df["created_dt"] = pd.to_datetime(features_df["created_dt"])
    else:
        features_df["created_dt"] = pd.to_datetime(features_df["created_timestamp"], unit="s")
        features_df = features_df.drop(columns=["created_timestamp"])

    stage_df["post_id"] = stage_df["post_id"].astype(str)
    merged = stage_df.merge(features_df, on="post_id", how="left")

    merged = merged.dropna(subset=["title", "R", "created_dt"])
    merged = _augment_handcrafted_features(merged)

    split_time = merged["created_dt"].quantile(config.split_quantile)
    train_mask = merged["created_dt"] <= split_time

    numeric_cols = [
        "title_length",
        "title_words",
        "title_chars_per_word",
        "has_question",
        "has_numbers",
        "has_exclamation",
        "capitalization_ratio",
        "all_caps_words",
        "number_count",
        "sentiment_compound",
        "sentiment_positive",
        "sentiment_negative",
        "sentiment_neutral",
        "clickbait_patterns",
        "clickbait_keywords",
        "has_clickbait",
        "title_token_len",
        "title_char_len",
        "title_avg_token_len",
        "feat_wh_start",
        "feat_contains_colon",
        "feat_contains_dash",
        "feat_contains_digit",
        "feat_contains_year",
        "feat_all_caps_word",
        "feat_title_exc",
        "feat_title_question",
        "feat_title_quote",
        "feat_token_shout_ratio",
    ]

    for col in numeric_cols:
        if col not in merged.columns:
            merged[col] = 0.0

    train_df = merged.loc[train_mask].copy()
    test_df = merged.loc[~train_mask].copy()

    target_train = train_df["R"].astype(float)
    target_test = test_df["R"].astype(float)

    return train_df, test_df, target_train, target_test


def run_stage_b_experiment(
    stage_outputs_path: Path,
    features_path: Path,
    output_path: Path | None = None,
    top_features_path: Path | None = None,
    config: StageBExperimentConfig | None = None,
) -> StageBExperimentResult:
    cfg = config or StageBExperimentConfig()
    train_df, test_df, y_train, y_test = _prepare_dataset(
        stage_outputs_path=stage_outputs_path,
        features_path=features_path,
        config=cfg,
    )

    numeric_cols = [
        "title_length",
        "title_words",
        "title_chars_per_word",
        "has_question",
        "has_numbers",
        "has_exclamation",
        "capitalization_ratio",
        "all_caps_words",
        "number_count",
        "sentiment_compound",
        "sentiment_positive",
        "sentiment_negative",
        "sentiment_neutral",
        "clickbait_patterns",
        "clickbait_keywords",
        "has_clickbait",
        "title_token_len",
        "title_char_len",
        "title_avg_token_len",
        "feat_wh_start",
        "feat_contains_colon",
        "feat_contains_dash",
        "feat_contains_digit",
        "feat_contains_year",
        "feat_all_caps_word",
        "feat_title_exc",
        "feat_title_question",
        "feat_title_quote",
        "feat_token_shout_ratio",
    ]

    model = _build_pipeline(numeric_cols=numeric_cols, config=cfg)
    model.fit(train_df, y_train)

    train_pred = model.predict(train_df)
    test_pred = model.predict(test_df)

    metrics = _compute_metrics(y_train, train_pred, y_test, test_pred)
    importances = _extract_feature_importance(model, top_k=15)

    result = StageBExperimentResult(
        train_rmse=float(metrics["train_rmse"]),
        test_rmse=float(metrics["test_rmse"]),
        train_mae=float(metrics["train_mae"]),
        test_mae=float(metrics["test_mae"]),
        train_r2=float(metrics["train_r2"]),
        test_r2=float(metrics["test_r2"]),
        n_train=int(len(train_df)),
        n_test=int(len(test_df)),
        split_time=train_df["created_dt"].max().isoformat(),
        top_positive_features=importances["positive"],
        top_negative_features=importances["negative"],
        config=asdict(cfg),
    )

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result.to_dict(), indent=2))

    if top_features_path is not None:
        rows: List[Dict[str, object]] = []
        for direction, items in importances.items():
            for item in items:
                rows.append(
                    {
                        "direction": direction,
                        "feature": item["feature"],
                        "weight": item["weight"],
                    }
                )
        if rows:
            top_features_path.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(rows).to_csv(top_features_path, index=False)

    return result


__all__ = [
    "StageBExperimentConfig",
    "StageBExperimentResult",
    "run_stage_b_experiment",
]
