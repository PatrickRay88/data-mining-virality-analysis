from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from .stage_modeling import (
    ModelingReport,
    prepare_stage_dataframe,
    run_stage_modeling_with_indices,
)


def _flatten_record(record: Dict[str, object]) -> Dict[str, object]:
    flat: Dict[str, object] = {}
    for key, value in record.items():
        if key in {"stage_a", "stage_b"} and isinstance(value, dict):
            for metric_name, metric_value in value.items():
                flat[f"{key}_{metric_name}"] = metric_value
        elif key == "residual_summary" and isinstance(value, dict):
            for metric_name, metric_value in value.items():
                flat[f"residual_{metric_name}"] = metric_value
        else:
            flat[key] = value
    return flat


def _records_to_frame(records: Sequence[Dict[str, object]]) -> pd.DataFrame:
    if not records:
        return pd.DataFrame()
    flattened = [_flatten_record(record) for record in records]
    return pd.DataFrame(flattened)


def _trimmed_mean_std(values: Sequence[float], trim_fraction: float) -> Optional[Dict[str, float]]:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return None
    if 0 < trim_fraction < 0.5:
        k = int(arr.size * trim_fraction)
        if k > 0 and arr.size - 2 * k > 0:
            arr = np.sort(arr)[k:-k]
    mean = float(arr.mean())
    std = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
    return {"mean": mean, "std": std}


def _report_summary(report: ModelingReport) -> Dict[str, object]:
    return {
        "stage_a": asdict(report.stage_a),
        "stage_b": asdict(report.stage_b),
        "residual_summary": report.residual_summary,
        "pairwise_accuracy": report.pairwise_accuracy,
        "pairwise_pairs": report.pairwise_pairs,
    }


def _add_gap_fields(summary: Dict[str, object]) -> None:
    stage_a = summary.get("stage_a", {})
    stage_b = summary.get("stage_b", {})
    if stage_a:
        summary["stage_a_gap_rmse"] = stage_a["test_rmse"] - stage_a["train_rmse"]
    if stage_b:
        summary["stage_b_gap_rmse"] = stage_b["test_rmse"] - stage_b["train_rmse"]


def run_temporal_splits(
    df: pd.DataFrame,
    quantiles: Sequence[float],
) -> List[Dict[str, object]]:
    results: List[Dict[str, object]] = []
    for q in quantiles:
        if not 0 < q < 1:
            continue
        split_time = df["created_dt"].quantile(q)
        train_idx = np.where(df["created_dt"] <= split_time)[0]
        test_idx = np.where(df["created_dt"] > split_time)[0]
        if len(train_idx) == 0 or len(test_idx) == 0:
            continue
        report = run_stage_modeling_with_indices(df, train_idx, test_idx)
        summary = _report_summary(report)
        summary.update(
            {
                "split_quantile": float(q),
                "split_time": split_time.isoformat(),
                "train_count": int(len(train_idx)),
                "test_count": int(len(test_idx)),
            }
        )
        _add_gap_fields(summary)
        results.append(summary)
    return results


def run_blocked_cross_validation(
    df: pd.DataFrame,
    frequency: str = "D",
    min_train: int = 200,
    min_test: int = 100,
    min_target_std: float = 1e-6,
    min_stage_b_rmse: float = 0.05,
) -> List[Dict[str, object]]:
    results: List[Dict[str, object]] = []
    created_original = df["created_dt"]
    tz_info = created_original.dt.tz
    created_naive = created_original.dt.tz_convert(None) if tz_info is not None else created_original
    block_labels = created_naive.dt.to_period(frequency)
    unique_blocks = pd.PeriodIndex(block_labels.unique()).sort_values()
    for block in unique_blocks:
        block_start = block.to_timestamp()
        block_end = (block + 1).to_timestamp()
        train_idx = np.where(created_naive < block_start)[0]
        test_mask = (created_naive >= block_start) & (created_naive < block_end)
        test_idx = np.where(test_mask)[0]
        if len(train_idx) < min_train or len(test_idx) < min_test:
            continue
        train_target = df.iloc[train_idx]["stage_a_target"]
        test_target = df.iloc[test_idx]["stage_a_target"]
        if train_target.std() <= min_target_std or test_target.std() <= min_target_std:
            continue
        report = run_stage_modeling_with_indices(df, train_idx, test_idx)
        if min(report.stage_b.train_rmse, report.stage_b.test_rmse) < min_stage_b_rmse:
            continue
        summary = _report_summary(report)
        summary.update(
            {
                "block": str(block),
                "block_start": block_start.isoformat(),
                "block_end": block_end.isoformat(),
                "train_count": int(len(train_idx)),
                "test_count": int(len(test_idx)),
            }
        )
        _add_gap_fields(summary)
        results.append(summary)
    return results


def run_bootstrap_resampling(
    df: pd.DataFrame,
    iterations: int,
    random_state: int = 42,
    min_oob_ratio: float = 0.1,
    max_abs_skew: float = 10.0,
) -> List[Dict[str, object]]:
    rng = np.random.default_rng(random_state)
    n = len(df)
    results: List[Dict[str, object]] = []
    min_oob = max(int(n * min_oob_ratio), 50)
    for iteration in range(iterations):
        train_idx = rng.integers(0, n, size=n)
        unique_train = np.unique(train_idx)
        test_mask = np.ones(n, dtype=bool)
        test_mask[unique_train] = False
        test_idx = np.where(test_mask)[0]
        if len(test_idx) < min_oob:
            continue
        report = run_stage_modeling_with_indices(df, train_idx, test_idx)
        summary = _report_summary(report)
        summary.update(
            {
                "iteration": int(iteration),
                "train_count": int(len(train_idx)),
                "test_count": int(len(test_idx)),
            }
        )
        residual_skew = summary.get("residual_summary", {}).get("residual_skew")
        if residual_skew is not None and abs(residual_skew) > max_abs_skew:
            continue
        _add_gap_fields(summary)
        results.append(summary)
    return results


def summarize_bootstrap(
    records: Sequence[Dict[str, object]],
    trim_fraction: float = 0.1,
) -> Dict[str, object]:
    if not records:
        return {}
    metrics: Dict[str, List[float]] = {
        "stage_a_train_rmse": [],
        "stage_a_test_rmse": [],
        "stage_b_train_rmse": [],
        "stage_b_test_rmse": [],
        "stage_b_test_r2": [],
        "pairwise_accuracy": [],
    }
    for record in records:
        stage_a = record["stage_a"]
        stage_b = record["stage_b"]
        metrics["stage_a_train_rmse"].append(stage_a["train_rmse"])
        metrics["stage_a_test_rmse"].append(stage_a["test_rmse"])
        metrics["stage_b_train_rmse"].append(stage_b["train_rmse"])
        metrics["stage_b_test_rmse"].append(stage_b["test_rmse"])
        metrics["stage_b_test_r2"].append(stage_b["test_r2"])
        if record["pairwise_accuracy"] is not None:
            metrics["pairwise_accuracy"].append(record["pairwise_accuracy"])
    summary: Dict[str, object] = {
        "iterations": len(records),
        "trim_fraction": float(trim_fraction),
    }
    for name, values in metrics.items():
        if not values:
            continue
        stats = _trimmed_mean_std(values, trim_fraction)
        if stats is not None:
            summary[name] = stats
    return summary


def run_learning_curve(
    df: pd.DataFrame,
    base_quantile: float,
    fractions: Sequence[float],
    min_train: int = 500,
) -> Dict[str, object]:
    if not 0 < base_quantile < 1:
        raise ValueError("base_quantile must be between 0 and 1")
    split_time = df["created_dt"].quantile(base_quantile)
    base_train_idx = np.where(df["created_dt"] <= split_time)[0]
    base_test_idx = np.where(df["created_dt"] > split_time)[0]
    results: List[Dict[str, object]] = []
    for frac in fractions:
        if frac <= 0 or frac > 1:
            continue
        cutoff = int(len(base_train_idx) * frac)
        if cutoff < min_train:
            continue
        train_idx = base_train_idx[:cutoff]
        report = run_stage_modeling_with_indices(df, train_idx, base_test_idx)
        summary = _report_summary(report)
        summary.update(
            {
                "fraction": float(frac),
                "train_count": int(len(train_idx)),
                "test_count": int(len(base_test_idx)),
            }
        )
        _add_gap_fields(summary)
        results.append(summary)
    return {
        "base_quantile": float(base_quantile),
        "base_split_time": split_time.isoformat(),
        "fractions": results,
    }


def _write_tables(report: Dict[str, object], tables_dir: Path) -> None:
    tables_dir.mkdir(parents=True, exist_ok=True)

    temporal_df = _records_to_frame(report.get("temporal_splits", []))
    if not temporal_df.empty:
        temporal_df = temporal_df.sort_values("split_quantile")
        temporal_df.to_csv(tables_dir / "stage_model_temporal_splits.csv", index=False)

    blocked_df = _records_to_frame(report.get("blocked_cross_validation", []))
    if not blocked_df.empty:
        blocked_df["block_start_dt"] = pd.to_datetime(blocked_df["block_start"], errors="coerce")
        blocked_df = blocked_df.sort_values("block_start_dt").drop(columns=["block_start_dt"])
        blocked_df.to_csv(tables_dir / "stage_model_blocked_cv.csv", index=False)

    bootstrap_records = report.get("bootstrap", {}).get("records", [])
    bootstrap_df = _records_to_frame(bootstrap_records)
    if not bootstrap_df.empty:
        bootstrap_df.sort_values("iteration").to_csv(
            tables_dir / "stage_model_bootstrap_records.csv", index=False
        )

    bootstrap_summary = report.get("bootstrap", {}).get("summary")
    if isinstance(bootstrap_summary, dict) and bootstrap_summary:
        rows: List[Dict[str, object]] = []
        for key, value in bootstrap_summary.items():
            if isinstance(value, dict):
                row: Dict[str, object] = {"metric": key}
                mean_value = value.get("mean")
                if mean_value is not None:
                    row["value"] = mean_value
                row.update(value)
                rows.append(row)
            else:
                rows.append({"metric": key, "value": value})
        pd.DataFrame(rows).to_csv(
            tables_dir / "stage_model_bootstrap_summary.csv", index=False
        )

    learning_fractions = report.get("learning_curve", {}).get("fractions", [])
    learning_df = _records_to_frame(learning_fractions)
    if not learning_df.empty:
        learning_df = learning_df.sort_values("fraction")
        learning_df.to_csv(tables_dir / "stage_model_learning_curve.csv", index=False)


def _write_figures(report: Dict[str, object], figures_dir: Path) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)
    import matplotlib.pyplot as plt

    temporal_df = _records_to_frame(report.get("temporal_splits", []))
    if not temporal_df.empty:
        temporal_df = temporal_df.sort_values("split_quantile")
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(
            temporal_df["split_quantile"],
            temporal_df["stage_a_test_rmse"],
            marker="o",
            label="Stage A Test RMSE",
        )
        ax.plot(
            temporal_df["split_quantile"],
            temporal_df["stage_b_test_rmse"],
            marker="o",
            label="Stage B Test RMSE",
        )
        ax.set_xlabel("Temporal split quantile")
        ax.set_ylabel("RMSE")
        ax.set_title("Temporal Split Performance")
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.4)
        fig.tight_layout()
        fig.savefig(figures_dir / "stage_model_temporal_rmse.png", dpi=200)
        plt.close(fig)

    blocked_df = _records_to_frame(report.get("blocked_cross_validation", []))
    if not blocked_df.empty:
        blocked_df["block_start_dt"] = pd.to_datetime(blocked_df["block_start"], errors="coerce")
        blocked_df = blocked_df.sort_values("block_start_dt")
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(
            blocked_df["block_start_dt"],
            blocked_df["stage_a_test_rmse"],
            marker="o",
            label="Stage A Test RMSE",
        )
        ax.plot(
            blocked_df["block_start_dt"],
            blocked_df["stage_b_test_rmse"],
            marker="o",
            label="Stage B Test RMSE",
        )
        ax.set_xlabel("Block start")
        ax.set_ylabel("RMSE")
        ax.set_title("Blocked Cross-Validation Performance")
        ax.legend()
        fig.autofmt_xdate()
        ax.grid(True, linestyle="--", alpha=0.4)
        fig.tight_layout()
        fig.savefig(figures_dir / "stage_model_blocked_rmse.png", dpi=200)
        plt.close(fig)

    bootstrap_records = report.get("bootstrap", {}).get("records", [])
    bootstrap_df = _records_to_frame(bootstrap_records)
    if not bootstrap_df.empty:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].hist(bootstrap_df["stage_b_test_rmse"], bins=20, color="#1f77b4", alpha=0.8)
        axes[0].set_title("Stage B Test RMSE (Bootstrap)")
        axes[0].set_xlabel("RMSE")
        axes[0].set_ylabel("Count")
        axes[1].hist(bootstrap_df["stage_b_test_r2"], bins=20, color="#ff7f0e", alpha=0.8)
        axes[1].set_title("Stage B Test R² (Bootstrap)")
        axes[1].set_xlabel("R²")
        axes[1].set_ylabel("Count")
        fig.tight_layout()
        fig.savefig(figures_dir / "stage_model_bootstrap_distributions.png", dpi=200)
        plt.close(fig)

    learning_fractions = report.get("learning_curve", {}).get("fractions", [])
    learning_df = _records_to_frame(learning_fractions)
    if not learning_df.empty:
        learning_df = learning_df.sort_values("fraction")
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(
            learning_df["fraction"],
            learning_df["stage_a_test_rmse"],
            marker="o",
            label="Stage A Test RMSE",
        )
        ax.plot(
            learning_df["fraction"],
            learning_df["stage_b_test_rmse"],
            marker="o",
            label="Stage B Test RMSE",
        )
        ax.set_xlabel("Train fraction")
        ax.set_ylabel("RMSE")
        ax.set_title("Learning Curve (Temporal Split)")
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.4)
        fig.tight_layout()
        fig.savefig(figures_dir / "stage_model_learning_curve.png", dpi=200)
        plt.close(fig)


def _write_artifacts(
    report: Dict[str, object],
    tables_dir: Optional[Path],
    figures_dir: Optional[Path],
) -> None:
    if tables_dir is not None:
        _write_tables(report, tables_dir)
    if figures_dir is not None:
        _write_figures(report, figures_dir)


def generate_diagnostics_report(
    data_path: Path,
    output_path: Path,
    temporal_quantiles: Sequence[float] = (0.6, 0.65, 0.7, 0.75, 0.8),
    bootstrap_iterations: int = 20,
    bootstrap_trim_fraction: float = 0.1,
    bootstrap_max_abs_skew: float = 10.0,
    learning_curve_quantile: float = 0.7,
    learning_curve_fractions: Sequence[float] = (0.3, 0.5, 0.7, 1.0),
    blocked_min_train: int = 200,
    blocked_min_test: int = 100,
    blocked_min_target_std: float = 1e-6,
    blocked_min_stage_b_rmse: float = 0.05,
    tables_dir: Optional[Path] = None,
    figures_dir: Optional[Path] = None,
) -> Dict[str, object]:
    raw_df = pd.read_parquet(data_path)
    prepared_df = prepare_stage_dataframe(raw_df)
    if prepared_df.empty:
        raise ValueError("No Reddit records available for diagnostics")

    temporal_results = run_temporal_splits(prepared_df, temporal_quantiles)
    blocked_results = run_blocked_cross_validation(
        prepared_df,
        min_train=blocked_min_train,
        min_test=blocked_min_test,
        min_target_std=blocked_min_target_std,
        min_stage_b_rmse=blocked_min_stage_b_rmse,
    )
    bootstrap_records = run_bootstrap_resampling(
        prepared_df,
        iterations=bootstrap_iterations,
        max_abs_skew=bootstrap_max_abs_skew,
    )
    learning_curve_result = run_learning_curve(
        prepared_df,
        base_quantile=learning_curve_quantile,
        fractions=learning_curve_fractions,
    )

    report: Dict[str, object] = {
        "metadata": {
            "rows": int(len(prepared_df)),
            "generated_at": pd.Timestamp.utcnow().isoformat(),
            "data_path": str(data_path),
        },
        "temporal_splits": temporal_results,
        "blocked_cross_validation": blocked_results,
        "bootstrap": {
            "records": bootstrap_records,
            "summary": summarize_bootstrap(
                bootstrap_records,
                trim_fraction=bootstrap_trim_fraction,
            ),
        },
        "learning_curve": learning_curve_result,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2))
    _write_artifacts(report, tables_dir, figures_dir)
    return report
