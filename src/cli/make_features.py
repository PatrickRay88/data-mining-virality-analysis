#!/usr/bin/env python3
"""Feature engineering CLI tool.

Usage:
    python -m src.cli.make_features --input-dir ./data --output data/features.parquet
"""

import glob
import logging
import sys
from pathlib import Path

import click
import numpy as np
import pandas as pd

from src.preprocess.features_context import ContextFeatureExtractor
from src.preprocess.features_titles import TitleFeatureExtractor
from src.preprocess.normalize import DataNormalizer


@click.command()
@click.option("--input-dir", "-i", default="./data", help="Input directory with parquet files")
@click.option("--output", "-o", default="./data/features.parquet", help="Output file for features")
@click.option(
    "--platform",
    type=click.Choice(["reddit", "hackernews", "both"]),
    default="both",
    help="Which platform data to process",
)
def main(input_dir, output, platform):
    """Extract features from collected data."""

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    input_path = Path(input_dir)
    if not input_path.exists():
        logger.error("Input directory %s does not exist", input_path)
        sys.exit(1)

    title_extractor = TitleFeatureExtractor()
    context_extractor = ContextFeatureExtractor()
    normalizer = DataNormalizer()

    logger.info("Processing %s data from %s", platform, input_path)

    try:
        if platform == "reddit":
            pattern = str(input_path / "reddit_*.parquet")
        elif platform == "hackernews":
            pattern = str(input_path / "hackernews_*.parquet")
        else:
            pattern = str(input_path / "*.parquet")

        data_files = glob.glob(pattern)

        if not data_files:
            logger.error("No data files found matching pattern: %s", pattern)
            sys.exit(1)

        logger.info("Found %d data files", len(data_files))

        all_data = []
        for file in data_files:
            if (
                "snapshots" not in file
                and "features" not in file
                and "stage_predictions" not in file
            ):
                df = pd.read_parquet(file)
                all_data.append(df)
                logger.info("Loaded %d records from %s", len(df), file)

        if not all_data:
            logger.error("No valid data files found")
            sys.exit(1)

        combined_df = pd.concat(all_data, ignore_index=True)
        logger.info("Combined dataset: %d records", len(combined_df))

        original_len = len(combined_df)
        combined_df = combined_df.drop_duplicates(subset=["post_id"])
        logger.info("Removed %d duplicates", original_len - len(combined_df))

        if "collection_type" not in combined_df.columns:
            combined_df["collection_type"] = "top"
        combined_df["is_new_collection"] = (combined_df["collection_type"] == "new").astype(int)

        if "subreddit" in combined_df.columns:
            subreddit_stats = (
                combined_df.groupby("subreddit")["score"].agg(["mean", "median", "count", "std"])
                .rename(
                    columns={
                        "mean": "subreddit_avg_score",
                        "median": "subreddit_median_score",
                        "count": "subreddit_post_count_global",
                        "std": "subreddit_score_std",
                    }
                )
            )
            combined_df = combined_df.merge(subreddit_stats, on="subreddit", how="left")
            combined_df["subreddit_score_std"] = combined_df["subreddit_score_std"].fillna(0)

        if "author_hash" in combined_df.columns:
            author_stats = (
                combined_df.groupby("author_hash")["score"].agg(["mean", "count"])
                .rename(columns={"mean": "author_avg_score", "count": "author_post_count_global"})
            )
            combined_df = combined_df.merge(author_stats, on="author_hash", how="left")
            combined_df["author_avg_score"] = combined_df["author_avg_score"].fillna(0)
            combined_df["author_post_count_global"] = combined_df["author_post_count_global"].fillna(0)

        logger.info("Extracting title features...")
        title_features = title_extractor.extract_features(combined_df["title"])

        logger.info("Extracting context features...")
        context_features = context_extractor.extract_features(combined_df)

        features_df = pd.concat([combined_df, title_features, context_features], axis=1)

        snapshot_files = glob.glob(str(input_path / "**/*snapshots*.parquet"), recursive=True)
        if snapshot_files:
            logger.info("Processing snapshot data for early score features...")
            all_snapshots = []
            total_snapshots = len(snapshot_files)
            with click.progressbar(
                snapshot_files,
                length=total_snapshots,
                label="Loading snapshot parquet files",
            ) as iterator:
                for file in iterator:
                    snapshots = pd.read_parquet(file)
                    all_snapshots.append(snapshots)

            if all_snapshots:
                combined_snapshots = pd.concat(all_snapshots, ignore_index=True)
                early_features = context_extractor.add_early_score_features(
                    combined_df, combined_snapshots
                )
                features_df = pd.concat([features_df, early_features], axis=1)
                logger.info("Added early score trajectory features")

                score_column_map = {
                    "score_at_5min": "score_5m",
                    "score_at_15min": "score_15m",
                    "score_at_30min": "score_30m",
                    "score_at_60min": "score_60m",
                }

                for source_col, target_col in score_column_map.items():
                    if source_col in features_df.columns:
                        features_df[target_col] = features_df[source_col]

                if "score_5m" not in features_df.columns:
                    features_df["score_5m"] = 0.0
                if "score_30m" not in features_df.columns:
                    features_df["score_30m"] = features_df["score_5m"]

                features_df["velocity_5_to_30m"] = (
                    features_df["score_30m"] - features_df["score_5m"]
                ) / 25.0
                features_df["velocity_5_to_30m"] = (
                    features_df["velocity_5_to_30m"].replace([np.inf, -np.inf], np.nan).fillna(0.0)
                )

                for minutes in (5, 15, 30, 60):
                    src_col = f"score_at_{minutes}min"
                    alias = f"score_{minutes}m"
                    if src_col in features_df.columns:
                        features_df[alias] = features_df[src_col].astype(float)

                if "score_at_5min" in features_df.columns and "score_at_30min" in features_df.columns:
                    score_5 = features_df["score_at_5min"].astype(float)
                    score_30 = features_df["score_at_30min"].astype(float)
                    velocity = (score_30 - score_5) / 25.0
                    features_df["velocity_5_to_30m"] = velocity.fillna(0.0)

                if "score_60m" not in features_df.columns:
                    base_60 = features_df.get("score_at_60min")
                    if base_60 is not None:
                        features_df["score_60m"] = base_60.astype(float)
                features_df["score_60m"] = features_df.get(
                    "score_60m", pd.Series(index=features_df.index, dtype=float)
                )
                features_df["score_60m"] = (
                    features_df["score_60m"].fillna(features_df.get("score", 0)).astype(float)
                )

        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        features_df.to_parquet(output_path, index=False)
        logger.info("Features saved to %s", output_path)

        print("\nFeature Engineering Summary:")
        print(f"Total records: {len(features_df)}")
        print(f"Total features: {len(features_df.columns)}")
        print(f"Platforms: {features_df['platform'].value_counts().to_dict()}")
        print(f"Output file: {output_path}")

        title_cols = [
            col
            for col in features_df.columns
            if any(x in col for x in ["title", "sentiment", "clickbait", "length"])
        ]
        context_cols = [
            col
            for col in features_df.columns
            if any(x in col for x in ["hour", "day", "age", "author", "content"])
        ]

        print(f"\nTitle features ({len(title_cols)}): {title_cols[:10]}...")
        print(f"Context features ({len(context_cols)}): {context_cols[:10]}...")

    except Exception as exc:  # pylint: disable=broad-except
        logger.error("Error during feature engineering: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
