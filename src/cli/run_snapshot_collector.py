#!/usr/bin/env python3
"""Continuous Reddit snapshot collector.

Run with::

    python -m src.cli.run_snapshot_collector -s technology -s science \
        --new-limit 75 --loop-seconds 300
"""

from __future__ import annotations

import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Set

import click
import pandas as pd

from src.ingest.reddit_client import RedditClient
from src.ingest.snapshot_manager import SnapshotConfig, SnapshotStateManager
from src.preprocess.normalize import DataNormalizer


def _load_reddit_client() -> RedditClient:
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass

    client_id = os.getenv("REDDIT_CLIENT_ID")
    client_secret = os.getenv("REDDIT_CLIENT_SECRET")
    user_agent = os.getenv("REDDIT_USER_AGENT")

    missing = [
        name
        for name, value in {
            "REDDIT_CLIENT_ID": client_id,
            "REDDIT_CLIENT_SECRET": client_secret,
            "REDDIT_USER_AGENT": user_agent,
        }.items()
        if not value
    ]

    if missing:
        raise click.ClickException(
            f"Missing Reddit credentials: {', '.join(missing)}. Update your .env file."
        )

    return RedditClient(client_id, client_secret, user_agent)


def _ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _collect_new_posts(
    client: RedditClient,
    normalizer: DataNormalizer,
    subreddits: Iterable[str],
    new_limit: int,
    known_ids: Set[str],
    run_id: str,
    run_timestamp: float,
    output_dir: Path,
) -> pd.DataFrame:
    all_new_frames: List[pd.DataFrame] = []

    for subreddit in subreddits:
        raw_new = client.collect_new_posts(subreddit, limit=new_limit)
        if raw_new.empty:
            continue

        raw_new["id"] = raw_new["id"].astype(str)
        raw_new = raw_new.drop_duplicates(subset="id")

        unseen_mask = ~raw_new["id"].isin(known_ids)
        unseen_posts = raw_new.loc[unseen_mask].copy()
        if unseen_posts.empty:
            continue

        normalized = normalizer.normalize_reddit_data(unseen_posts)
        normalized["collection_type"] = "new"
        normalized["collection_run_id"] = run_id
        normalized["ingested_timestamp"] = run_timestamp
        normalized["collection_type_detail"] = "stream"
        normalized["subreddit"] = normalized["subreddit"].fillna(subreddit)

        all_new_frames.append(normalized)

    if not all_new_frames:
        return pd.DataFrame()

    batch_df = pd.concat(all_new_frames, ignore_index=True)
    batch_df = batch_df.drop_duplicates(subset="post_id")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"reddit_stream_{timestamp}.parquet"
    normalizer.save_to_parquet(batch_df, str(output_path))

    return batch_df


def _initial_snapshots(new_posts: pd.DataFrame, snapshot_time: float) -> List[Dict[str, float]]:
    snapshots: List[Dict[str, float]] = []
    if new_posts.empty:
        return snapshots

    for row in new_posts.itertuples(index=False):
        created_ts = float(row.created_timestamp)
        age_minutes = max((snapshot_time - created_ts) / 60.0, 0.0)
        snapshots.append(
            {
                "post_id": str(row.post_id),
                "subreddit": getattr(row, "subreddit", ""),
                "interval_minutes": 0,
                "snapshot_time": snapshot_time,
                "post_age_minutes": age_minutes,
                "score": float(row.score),
                "upvote_ratio": getattr(row, "upvote_ratio", float("nan")),
                "num_comments": float(getattr(row, "comment_count", 0)),
            }
        )
    return snapshots


def _collect_due_snapshots(
    client: RedditClient,
    state: pd.DataFrame,
    due_masks: Dict[int, pd.Series],
    snapshot_time: float,
) -> List[Dict[str, float]]:
    if state.empty:
        return []

    due_post_ids: Set[str] = set()
    for mask in due_masks.values():
        if mask.any():
            due_post_ids.update(state.loc[mask, "post_id"].astype(str))

    if not due_post_ids:
        return []

    details = client.fetch_posts_by_ids(sorted(due_post_ids))
    if details.empty:
        return []

    details["id"] = details["id"].astype(str)
    details = details.set_index("id", drop=False)

    snapshots: List[Dict[str, float]] = []
    for minutes, mask in due_masks.items():
        if not mask.any():
            continue
        target_ids = state.loc[mask, "post_id"].astype(str)
        for post_id in target_ids:
            if post_id not in details.index:
                continue
            row = details.loc[post_id]
            created_ts = float(row["created_utc"])
            age_minutes = max((snapshot_time - created_ts) / 60.0, 0.0)
            snapshots.append(
                {
                    "post_id": post_id,
                    "subreddit": str(row["subreddit"]),
                    "interval_minutes": minutes,
                    "snapshot_time": snapshot_time,
                    "post_age_minutes": age_minutes,
                    "score": float(row["score"]),
                    "upvote_ratio": float(row.get("upvote_ratio", float("nan"))),
                    "num_comments": float(row.get("num_comments", 0)),
                }
            )

    return snapshots


@click.command()
@click.option(
    "--subreddit",
    "-s",
    multiple=True,
    required=True,
    help="Subreddits to poll for new posts (multiple allowed).",
)
@click.option(
    "--new-limit",
    default=75,
    show_default=True,
    help="Maximum number of /new posts to fetch per subreddit each cycle.",
)
@click.option(
    "--loop-seconds",
    default=300,
    show_default=True,
    help="Seconds to sleep between polling cycles (ignored when --iterations=1).",
)
@click.option(
    "--iterations",
    default=0,
    show_default=True,
    help="Number of cycles to run (0 = infinite until interrupted).",
)
@click.option(
    "--output-dir",
    default="./data",
    show_default=True,
    help="Directory to store normalized Reddit parquet slices.",
)
@click.option(
    "--snapshots-dir",
    default="./data/snapshots",
    show_default=True,
    help="Directory for snapshot parquet outputs.",
)
@click.option(
    "--state-file",
    default="./data/state/reddit_snapshot_state.parquet",
    show_default=True,
    help="Path to persistent snapshot state parquet.",
)
@click.option(
    "--intervals",
    default="5,15,30,60",
    show_default=True,
    help="Comma-separated snapshot intervals (minutes).",
)
@click.option(
    "--grace-minutes",
    default=120,
    show_default=True,
    help="Grace period after last interval before pruning posts (minutes).",
)
@click.option(
    "--tolerance-minutes",
    default=2.0,
    show_default=True,
    help="Tolerance window when considering an interval collected (minutes).",
)
def main(
    subreddit: Iterable[str],
    new_limit: int,
    loop_seconds: int,
    iterations: int,
    output_dir: str,
    snapshots_dir: str,
    state_file: str,
    intervals: str,
    grace_minutes: int,
    tolerance_minutes: float,
) -> None:
    """Run the continuous Reddit snapshot collector."""

    subreddits = [s.lower() for s in subreddit]
    interval_list = [int(x.strip()) for x in intervals.split(",") if x.strip()]
    if not interval_list:
        raise click.ClickException("At least one snapshot interval must be specified.")

    output_path = Path(output_dir)
    snapshots_path = Path(snapshots_dir)
    state_path = Path(state_file)

    _ensure_directory(output_path)
    _ensure_directory(snapshots_path)
    _ensure_directory(state_path.parent)

    client = _load_reddit_client()
    normalizer = DataNormalizer()
    config = SnapshotConfig(
        intervals=interval_list,
        grace_minutes=grace_minutes,
        tolerance_minutes=tolerance_minutes,
    )
    state_manager = SnapshotStateManager(state_path, config)

    loop_forever = iterations <= 0
    cycle = 0

    try:
        while loop_forever or cycle < iterations:
            cycle += 1
            run_dt = datetime.now(timezone.utc)
            run_timestamp = run_dt.timestamp()
            run_id = run_dt.strftime("%Y%m%d_%H%M%S")

            state = state_manager.load()
            known_ids = set(state["post_id"].astype(str)) if not state.empty else set()

            new_posts = _collect_new_posts(
                client,
                normalizer,
                subreddits,
                new_limit,
                known_ids,
                run_id,
                run_timestamp,
                output_path,
            )

            if not new_posts.empty:
                state = state_manager.merge_new_posts(state, new_posts, run_timestamp)

            due_masks = state_manager.get_due_snapshot_masks(state, run_timestamp)

            snapshots: List[Dict[str, float]] = []
            if not new_posts.empty:
                snapshots.extend(_initial_snapshots(new_posts, run_timestamp))

            due_snapshots = _collect_due_snapshots(client, state, due_masks, run_timestamp)
            snapshots.extend(due_snapshots)

            collected_by_interval: Dict[int, Set[str]] = {}
            for snapshot in due_snapshots:
                minutes = int(snapshot["interval_minutes"])
                collected_by_interval.setdefault(minutes, set()).add(str(snapshot["post_id"]))

            for minutes, post_ids in collected_by_interval.items():
                state = state_manager.mark_collected(state, post_ids, minutes, run_timestamp)

            state = state_manager.prune(state, run_timestamp)
            state_manager.save(state)

            if snapshots:
                snapshots_df = pd.DataFrame(snapshots)
                snapshot_file = snapshots_path / f"reddit_snapshots_{run_id}.parquet"
                snapshots_df.to_parquet(snapshot_file, index=False)
                click.echo(f"[{run_id}] Wrote {len(snapshots_df)} snapshots to {snapshot_file}")
            else:
                click.echo(f"[{run_id}] No snapshots to record this cycle.")

            click.echo(
                f"[{run_id}] State rows: {len(state)} | New posts this cycle: {len(new_posts)}"
            )

            if not loop_forever and cycle >= iterations:
                break

            if loop_seconds > 0:
                time.sleep(loop_seconds)

    except KeyboardInterrupt:
        click.echo("Interrupted by user. Saving state...")
        state = locals().get("state") if "state" in locals() else state_manager.load()
        state_manager.save(state)


if __name__ == "__main__":
    main()
