"""Stage B title embedding experiments.

This module encodes post titles with a sentence-transformer model and fits
regressors on Stage A residuals to test whether dense representations unlock
additional lift. It mirrors the TF-IDF enhancement helper but focuses on
semantic embeddings and reports RMSE, R^2, and pairwise ordering accuracy.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import ElasticNetCV, RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

try:
    from sentence_transformers import SentenceTransformer
except ImportError as exc:  # pragma: no cover - handled at runtime
    raise ImportError(
        "sentence-transformers is required for title embedding experiments. "
        "Install it via `pip install sentence-transformers`."
    ) from exc


@dataclass
class StageBEmbeddingConfig:
    split_quantile: float = 0.7
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    batch_size: int = 32
    use_pca: bool = False
    pca_components: int = 128
    regressor: str = "ridge"  # choices: ridge, elasticnet, mlp
    ridge_alphas: Sequence[float] = (0.1, 1.0, 10.0)
    elasticnet_l1_ratios: Sequence[float] = (0.1, 0.5, 0.9)
    elasticnet_alphas: Sequence[float] = (0.01, 0.1, 1.0)
    mlp_hidden_layers: Sequence[int] = (256, 128)
    mlp_max_iter: int = 400
    random_state: int = 42


@dataclass
class StageBEmbeddingResult:
    train_rmse: float
    test_rmse: float
    train_mae: float
    test_mae: float
    train_r2: float
    test_r2: float
    pairwise_accuracy: float | None
    pairwise_pairs: int | None
    n_train: int
    n_test: int
    split_time: str
    config: Dict[str, object]

    def to_dict(self) -> Dict[str, object]:
        payload = asdict(self)
        payload["config"] = dict(payload["config"])
        return payload


@dataclass
class StageBEmbeddingSearchResult:
    best_result: StageBEmbeddingResult
    ranked_results: List[StageBEmbeddingResult]
    scoring: str
    best_score: float
    grid_spec: Dict[str, Any] | None = None


def _load_frame(stage_outputs_path: Path, features_path: Path) -> pd.DataFrame:
    stage_df = pd.read_parquet(stage_outputs_path)
    features_df = pd.read_parquet(features_path)

    if "created_dt" in features_df.columns:
        created_col = "created_dt"
        features_df[created_col] = pd.to_datetime(features_df[created_col])
    elif "created_timestamp" in features_df.columns:
        created_col = "created_timestamp"
        features_df[created_col] = pd.to_datetime(
            features_df[created_col], unit="s", utc=True
        )
        features_df = features_df.rename(columns={created_col: "created_dt"})
        created_col = "created_dt"
    else:  # pragma: no cover - defensive
        raise KeyError(
            "Features parquet must contain created_dt or created_timestamp column"
        )

    merged = (
        stage_df.merge(
            features_df[["post_id", created_col]], on="post_id", how="left"
        )
        .dropna(subset=["title", "R", created_col])
        .rename(columns={created_col: "created_dt"})
    )

    merged["created_dt"] = pd.to_datetime(merged["created_dt"], utc=True)
    return merged


def _train_test_split(
    frame: pd.DataFrame, split_quantile: float
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Timestamp]:
    split_time = frame["created_dt"].quantile(split_quantile)
    train_mask = frame["created_dt"] <= split_time
    train_df = frame.loc[train_mask].copy()
    test_df = frame.loc[~train_mask].copy()
    y_train = train_df["R"].astype(float)
    y_test = test_df["R"].astype(float)
    return train_df, test_df, y_train, y_test, split_time


def _embed_titles(
    titles: Iterable[str],
    model_name: str,
    batch_size: int,
) -> np.ndarray:
    model = SentenceTransformer(model_name)
    embeddings = model.encode(
        list(titles),
        batch_size=batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=False,
    )
    return np.asarray(embeddings, dtype=np.float32)


def _prepare_features(
    train_titles: pd.Series,
    test_titles: pd.Series,
    config: StageBEmbeddingConfig,
) -> tuple[np.ndarray, np.ndarray]:
    train_vecs = _embed_titles(train_titles.tolist(), config.model_name, config.batch_size)
    test_vecs = _embed_titles(test_titles.tolist(), config.model_name, config.batch_size)

    if config.use_pca:
        pca = PCA(
            n_components=config.pca_components,
            random_state=config.random_state,
        )
        train_vecs = pca.fit_transform(train_vecs)
        test_vecs = pca.transform(test_vecs)

    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_vecs)
    test_scaled = scaler.transform(test_vecs)
    return train_scaled, test_scaled


def _build_regressor(config: StageBEmbeddingConfig):
    regressor_name = config.regressor.lower()
    if regressor_name == "ridge":
        return RidgeCV(alphas=tuple(config.ridge_alphas))
    if regressor_name == "elasticnet":
        return ElasticNetCV(
            l1_ratio=list(config.elasticnet_l1_ratios),
            alphas=list(config.elasticnet_alphas),
            max_iter=5000,
            cv=5,
            random_state=config.random_state,
        )
    if regressor_name == "mlp":
        return MLPRegressor(
            hidden_layer_sizes=tuple(config.mlp_hidden_layers),
            max_iter=config.mlp_max_iter,
            random_state=config.random_state,
        )
    raise ValueError(
        "Unsupported regressor '{name}'. Choose from 'ridge', 'elasticnet', 'mlp'."
        .format(name=config.regressor)
    )


def _compute_pairwise_accuracy(
    df: pd.DataFrame, actual_col: str, pred_col: str
) -> tuple[float | None, int | None]:
    if df.empty:
        return None, None
    temp = df[["subreddit", "created_dt", actual_col, pred_col]].dropna().copy()
    if temp.empty:
        return None, None
    temp["hour_bucket"] = temp["created_dt"].dt.floor("h")
    pairwise_correct = 0
    pairwise_total = 0
    for (_, group) in temp.groupby(["subreddit", "hour_bucket"], dropna=True):
        values = group[[actual_col, pred_col]].to_numpy()
        n = len(values)
        if n < 2:
            continue
        for i in range(n):
            for j in range(i + 1, n):
                actual_diff = values[i, 0] - values[j, 0]
                pred_diff = values[i, 1] - values[j, 1]
                if actual_diff == 0 or pred_diff == 0:
                    continue
                pairwise_total += 1
                if actual_diff * pred_diff > 0:
                    pairwise_correct += 1
    if pairwise_total == 0:
        return None, None
    return pairwise_correct / pairwise_total, pairwise_total


def _compute_metrics(
    y_true_train: np.ndarray,
    y_pred_train: np.ndarray,
    y_true_test: np.ndarray,
    y_pred_test: np.ndarray,
) -> Dict[str, float]:
    return {
        "train_rmse": float(np.sqrt(mean_squared_error(y_true_train, y_pred_train))),
        "test_rmse": float(np.sqrt(mean_squared_error(y_true_test, y_pred_test))),
        "train_mae": float(mean_absolute_error(y_true_train, y_pred_train)),
        "test_mae": float(mean_absolute_error(y_true_test, y_pred_test)),
        "train_r2": float(r2_score(y_true_train, y_pred_train)),
        "test_r2": float(r2_score(y_true_test, y_pred_test)),
    }


def build_stage_b_config_grid(
    parameter_grid: Dict[str, Sequence[Any]],
    base_config: StageBEmbeddingConfig | None = None,
) -> List[StageBEmbeddingConfig]:
    base = asdict(base_config or StageBEmbeddingConfig())
    keys = list(parameter_grid.keys())
    values_product = list(product(*(parameter_grid[key] for key in keys)))
    configs: List[StageBEmbeddingConfig] = []
    for combo in values_product:
        payload = base.copy()
        for key, value in zip(keys, combo):
            payload[key] = value
        configs.append(StageBEmbeddingConfig(**payload))
    return configs


def _score_result(result: StageBEmbeddingResult, scoring: str) -> float:
    if scoring == "pairwise_accuracy":
        return result.pairwise_accuracy if result.pairwise_accuracy is not None else float("-inf")
    if scoring == "test_rmse":
        return -result.test_rmse
    if scoring == "train_rmse":
        return -result.train_rmse
    if scoring == "test_r2":
        return result.test_r2
    if scoring == "train_r2":
        return result.train_r2
    if scoring == "test_mae":
        return -result.test_mae
    if scoring == "train_mae":
        return -result.train_mae
    raise ValueError(f"Unsupported scoring metric '{scoring}'.")


def _config_slug(config: StageBEmbeddingConfig, index: int) -> str:
    model_fragment = config.model_name.split("/")[-1]
    pca_part = f"pca{config.pca_components}" if config.use_pca else "nopca"
    return "_".join(
        [
            f"{index:03d}",
            model_fragment,
            config.regressor,
            pca_part,
            f"bs{config.batch_size}",
            f"q{config.split_quantile:.2f}".replace(".", ""),
        ]
    )


def run_stage_b_embedding_grid(
    stage_outputs_path: Path | str,
    features_path: Path | str,
    configs: Iterable[StageBEmbeddingConfig],
    output_dir: Path | str | None = None,
    predictions_dir: Path | str | None = None,
    scoring: str = "pairwise_accuracy",
    grid_spec: Dict[str, Any] | None = None,
    progress: bool = True,
    log_path: Path | str | None = None,
) -> StageBEmbeddingSearchResult:
    stage_outputs_path = Path(stage_outputs_path)
    features_path = Path(features_path)
    output_dir_path = Path(output_dir) if output_dir is not None else None
    predictions_dir_path = Path(predictions_dir) if predictions_dir is not None else None
    log_path_obj = Path(log_path) if log_path is not None else None

    if output_dir_path is not None:
        output_dir_path.mkdir(parents=True, exist_ok=True)
    if predictions_dir_path is not None:
        predictions_dir_path.mkdir(parents=True, exist_ok=True)
    log_handle = None
    if log_path_obj is not None:
        log_path_obj.parent.mkdir(parents=True, exist_ok=True)
        log_handle = log_path_obj.open("a", encoding="utf-8")

    configs_list = list(configs)
    if not configs_list:
        raise ValueError("No configurations provided for embedding grid search")

    progress_bar = None
    total_configs = len(configs_list)
    if progress:
        try:  # pragma: no cover - tqdm optional
            from tqdm.auto import tqdm as tqdm_auto  # type: ignore

            progress_bar = tqdm_auto(configs_list, desc="Stage B embeddings", unit="cfg")
        except ImportError:
            progress_bar = None

    results: List[StageBEmbeddingResult] = []
    iterator: Iterable[tuple[int, StageBEmbeddingConfig]]
    if progress_bar is not None:
        iterator = enumerate(progress_bar)  # type: ignore[arg-type]
    else:
        iterator = enumerate(configs_list)

    for idx, cfg in iterator:
        slug = _config_slug(cfg, idx)
        start_stamp = datetime.utcnow().isoformat()
        if log_handle is not None:
            log_handle.write(
                f"[{start_stamp}] START idx={idx} slug={slug} model={cfg.model_name} reg={cfg.regressor} pca={cfg.use_pca} split={cfg.split_quantile}\n"
            )
            log_handle.flush()
        output_path = (
            output_dir_path / f"{slug}.json" if output_dir_path is not None else None
        )
        predictions_path = (
            predictions_dir_path / f"{slug}.parquet"
            if predictions_dir_path is not None
            else None
        )
        result = run_stage_b_embedding_experiment(
            stage_outputs_path=stage_outputs_path,
            features_path=features_path,
            output_path=output_path,
            predictions_path=predictions_path,
            config=cfg,
        )
        results.append(result)
        if progress_bar is not None:
            postfix = {
                "cfg": f"{idx + 1}/{total_configs}",
                "pair": f"{(result.pairwise_accuracy or 0.0):.3f}",
                "rmse": f"{result.test_rmse:.3f}",
            }
            progress_bar.set_postfix(postfix)
        if log_handle is not None:
            done_stamp = datetime.utcnow().isoformat()
            log_handle.write(
                f"[{done_stamp}] DONE idx={idx} slug={slug} pair={result.pairwise_accuracy} rmse={result.test_rmse} r2={result.test_r2}\n"
            )
            log_handle.flush()

    if progress_bar is not None:
        progress_bar.close()
    if log_handle is not None:
        log_handle.close()

    scored_results = [(_score_result(result, scoring), result) for result in results]
    scored_results.sort(key=lambda item: item[0], reverse=True)
    best_score, best_result = scored_results[0]
    ranked_results = [item[1] for item in scored_results]

    if output_dir_path is not None:
        summary_payload = {
            "scoring": scoring,
            "best_score": best_score,
            "best_result": best_result.to_dict(),
            "ranked_results": [result.to_dict() for result in ranked_results],
        }
        if grid_spec is not None:
            summary_payload["grid_spec"] = grid_spec
        (output_dir_path / "search_summary.json").write_text(
            json.dumps(summary_payload, indent=2),
            encoding="utf-8",
        )

    return StageBEmbeddingSearchResult(
        best_result=best_result,
        ranked_results=ranked_results,
        scoring=scoring,
        best_score=best_score,
        grid_spec=grid_spec,
    )


def run_stage_b_embedding_experiment(
    stage_outputs_path: Path | str,
    features_path: Path | str,
    output_path: Path | str | None = None,
    predictions_path: Path | str | None = None,
    config: StageBEmbeddingConfig | None = None,
) -> StageBEmbeddingResult:
    cfg = config or StageBEmbeddingConfig()
    stage_outputs_path = Path(stage_outputs_path)
    features_path = Path(features_path)
    output_path = Path(output_path) if output_path is not None else None
    predictions_path = Path(predictions_path) if predictions_path is not None else None

    frame = _load_frame(stage_outputs_path, features_path)
    train_df, test_df, y_train, y_test, split_time = _train_test_split(
        frame, cfg.split_quantile
    )

    if train_df.empty or test_df.empty:
        raise ValueError("Not enough data to form train/test splits for embeddings experiment")

    X_train, X_test = _prepare_features(train_df["title"], test_df["title"], cfg)
    regressor = _build_regressor(cfg)
    regressor.fit(X_train, y_train)

    train_pred = regressor.predict(X_train)
    test_pred = regressor.predict(X_test)

    metrics = _compute_metrics(y_train.to_numpy(), train_pred, y_test.to_numpy(), test_pred)
    pairwise_accuracy, pairwise_pairs = _compute_pairwise_accuracy(
        pd.DataFrame(
            {
                "subreddit": test_df["subreddit"].to_numpy(),
                "created_dt": test_df["created_dt"].to_numpy(),
                "actual": y_test.to_numpy(),
                "pred": test_pred,
            }
        ),
        actual_col="actual",
        pred_col="pred",
    )

    result = StageBEmbeddingResult(
        train_rmse=metrics["train_rmse"],
        test_rmse=metrics["test_rmse"],
        train_mae=metrics["train_mae"],
        test_mae=metrics["test_mae"],
        train_r2=metrics["train_r2"],
        test_r2=metrics["test_r2"],
        pairwise_accuracy=pairwise_accuracy,
        pairwise_pairs=pairwise_pairs,
        n_train=int(len(train_df)),
        n_test=int(len(test_df)),
        split_time=split_time.isoformat(),
        config=asdict(cfg),
    )

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")

    if predictions_path is not None:
        predictions_path.parent.mkdir(parents=True, exist_ok=True)
        payload = test_df[["post_id", "subreddit", "created_dt"]].copy()
        payload["stage_b_actual"] = y_test.to_numpy()
        payload["stage_b_pred"] = test_pred
        payload.to_parquet(predictions_path, index=False)

    return result


__all__ = [
    "StageBEmbeddingConfig",
    "StageBEmbeddingResult",
    "StageBEmbeddingSearchResult",
    "run_stage_b_embedding_experiment",
    "run_stage_b_embedding_grid",
    "build_stage_b_config_grid",
]
