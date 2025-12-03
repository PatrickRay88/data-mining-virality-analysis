"""Export a human-readable dataset for course submission.

This CLI loads the feature-engineered table produced by ``bin/make_features.py``
and emits a trimmed CSV with descriptive column names and derived growth
metrics. The output is meant for human review (e.g., sharing with faculty)
so it prioritises readability over model-ready structure.
"""

from __future__ import annotations

import sys
from datetime import UTC, datetime
from pathlib import Path
import click
import pandas as pd


DEFAULT_INPUT = Path("data/features.parquet")
DEFAULT_OUTPUT = Path("docs/title_lift_human_readable.csv")

# Columns we prefer to surface (will keep the subset that exists in the input).
PREFERRED_COLUMNS: tuple[str, ...] = (
    "platform",
    "post_id",
    "title",
    "subreddit",
    "domain",
    "created_timestamp",
    "score",
    "comment_count",
    "upvote_ratio",
    "collection_type",
    "hour_of_day",
    "day_of_week",
    "is_weekend",
    "is_self_post",
    "is_image_post",
    "is_external_link",
    "score_5m",
    "score_15m",
    "score_30m",
    "score_60m",
    "early_momentum",
    "sustained_growth",
    "velocity_5_to_30m",
    "sentiment_compound",
    "sentiment_positive",
    "sentiment_negative",
    "sentiment_neutral",
    "title_length",
    "title_words",
    "title_chars_per_word",
    "has_question",
    "has_exclamation",
    "number_count",
    "has_numbers",
    "has_clickbait",
    "clickbait_keywords",
    "clickbait_patterns",
    "flesch_kincaid_grade",
    "avg_word_length",
    "type_token_ratio",
    "person_entities",
    "org_entities",
    "date_entities",
)

COLUMN_RENAMES: dict[str, str] = {
    "created_timestamp": "created_utc",
    "score": "score_final",
    "comment_count": "comment_count_final",
    "is_self_post": "is_text_post",
    "hour_of_day": "hour_of_day_utc",
    "day_of_week": "day_of_week_index",
    "has_clickbait": "clickbait_flag",
    "has_question": "question_mark_in_title",
    "has_exclamation": "exclamation_in_title",
    "has_numbers": "contains_digit",
    "number_count": "numeric_token_count",
    "title_chars_per_word": "avg_chars_per_word",
    "avg_word_length": "avg_word_length_chars",
    "velocity_5_to_30m": "velocity_score_5m_to_30m",
}

BOOL_COLUMNS: tuple[str, ...] = (
    "is_text_post",
    "is_image_post",
    "is_external_link",
    "is_weekend",
    "clickbait_flag",
    "question_mark_in_title",
    "exclamation_in_title",
    "contains_digit",
)

DAY_LABELS = {
    0: "Monday",
    1: "Tuesday",
    2: "Wednesday",
    3: "Thursday",
    4: "Friday",
    5: "Saturday",
    6: "Sunday",
}


def _load_table(input_path: Path) -> pd.DataFrame:
    """Read parquet/csv; fallback to sibling extensions if necessary."""
    candidates: list[Path] = []
    if input_path.exists():
        candidates.append(input_path)
    if not input_path.exists():
        candidates.extend(input_path.with_suffix(sfx) for sfx in (".parquet", ".csv"))
    for candidate in candidates:
        if not candidate.exists():
            continue
        if candidate.suffix == ".parquet":
            return pd.read_parquet(candidate)
        if candidate.suffix == ".csv":
            return pd.read_csv(candidate)
    raise FileNotFoundError(f"Could not locate dataset at {input_path} (tried parquet/csv variants)")


def _select_columns(df: pd.DataFrame) -> pd.DataFrame:
    available = [col for col in PREFERRED_COLUMNS if col in df.columns]
    if not available:
        return df.copy()
    return df.loc[:, available].copy()


def _prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = _select_columns(df)

    # Normalise timestamps for readability.
    if "created_timestamp" in df.columns:
        created_series = df["created_timestamp"]
        numeric = pd.to_numeric(created_series, errors="coerce")
        if numeric.notna().any():
            max_abs = numeric.abs().max()
            # Values on the order of 1e9 imply Unix seconds; larger magnitudes likely ns.
            unit = "s" if max_abs < 1e12 else "ns"
            created = pd.to_datetime(numeric, unit=unit, errors="coerce", utc=True)
        else:
            created = pd.to_datetime(created_series, errors="coerce", utc=True)
        df["created_timestamp"] = created.dt.strftime("%Y-%m-%d %H:%M:%S")

    # Stage additional derived metrics when the underlying columns exist.
    if {"score_5m", "score_60m"}.issubset(df.columns):
        df["score_delta_60m"] = df["score_60m"] - df["score_5m"]
    if {"score_5m", "score_15m"}.issubset(df.columns):
        df["score_delta_15m"] = df["score_15m"] - df["score_5m"]
    if {"score_15m", "score_30m"}.issubset(df.columns):
        df["score_delta_30m"] = df["score_30m"] - df["score_15m"]

    if "day_of_week" in df.columns and "day_of_week_label" not in df.columns:
        df["day_of_week_label"] = df["day_of_week"].map(DAY_LABELS)

    # Apply user-friendly column names.
    rename_map = {col: new for col, new in COLUMN_RENAMES.items() if col in df.columns}
    df = df.rename(columns=rename_map)

    # Convert binary indicators to boolean for readability.
    for col in BOOL_COLUMNS:
        if col in df.columns:
            df[col] = df[col].astype(bool)

    # Order columns with metadata first, then metrics.
    preferred_order: list[str] = []
    for key in (
        "platform",
        "post_id",
        "title",
        "subreddit",
        "domain",
        "created_utc",
        "score_final",
        "comment_count_final",
        "upvote_ratio",
        "collection_type",
        "hour_of_day_utc",
        "day_of_week_index",
        "day_of_week_label",
        "is_text_post",
        "is_image_post",
        "is_external_link",
        "is_weekend",
    ):
        if key in df.columns:
            preferred_order.append(key)

    remaining = [col for col in df.columns if col not in preferred_order]
    return df.loc[:, preferred_order + remaining]


def _validate_output_path(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _summarise(df: pd.DataFrame) -> None:
    print(f"Exporting {len(df):,} rows with {df.shape[1]} columns.")
    sample_cols = df.columns[:12].tolist()
    print("Sample columns:", sample_cols)
    print("First row:")
    with pd.option_context("display.max_columns", 12):
        print(df.head(1).to_string(index=False))


@click.command()
@click.option(
    "--input-path",
    type=click.Path(path_type=Path),
    default=DEFAULT_INPUT,
    show_default=True,
    help="Feature dataset to load (parquet or csv).",
)
@click.option(
    "--output-path",
    type=click.Path(path_type=Path),
    default=DEFAULT_OUTPUT,
    show_default=True,
    help="Destination CSV path.",
)
@click.option(
    "--limit",
    type=int,
    default=None,
    help="Optional row cap for quick previews (omit to export all rows).",
)
@click.option(
    "--sort-desc/--no-sort-desc",
    default=True,
    show_default=True,
    help="Sort by creation time (newest first) when timestamp is available.",
)
def main(input_path: Path, output_path: Path, limit: int | None, sort_desc: bool) -> None:
    """Generate the human-readable CSV."""
    try:
        df = _load_table(input_path)
    except FileNotFoundError as exc:
        raise click.ClickException(str(exc)) from exc

    readable_df = _prepare_dataframe(df)
    if limit is not None and limit > 0:
        readable_df = readable_df.head(limit)

    if sort_desc and "created_utc" in readable_df.columns:
        readable_df = readable_df.sort_values("created_utc", ascending=False)

    _validate_output_path(output_path)
    readable_df.to_csv(output_path, index=False)
    _summarise(readable_df)

    timestamp = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp} UTC] Wrote human-readable dataset -> {output_path}")


if __name__ == "__main__":
    # Delegate to Click for CLI handling.
    try:
        main(standalone_mode=True)
    except click.ClickException as exc:  # pragma: no cover - user-facing error
        click.echo(f"Error: {exc}", err=True)
        sys.exit(1)
