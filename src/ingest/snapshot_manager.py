"""Utilities for tracking Reddit post snapshot state.

This module keeps a lightweight state table for posts that require
scheduled score snapshots (e.g., 5/15/30/60 minute checkpoints).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd


@dataclass
class SnapshotConfig:
    """Configuration for snapshot tracking."""

    intervals: List[int]
    grace_minutes: int = 120
    tolerance_minutes: float = 2.0

    def __post_init__(self) -> None:
        self.intervals = sorted(int(x) for x in self.intervals)


class SnapshotStateManager:
    """Persisted state helper for Reddit snapshot collection."""

    STATE_COLUMNS = [
        "post_id",
        "subreddit",
        "created_timestamp",
        "first_seen_timestamp",
        "last_snapshot_timestamp",
        "drop_after_timestamp",
    ]

    def __init__(self, state_path: Path, config: SnapshotConfig) -> None:
        self.state_path = Path(state_path)
        self.config = config
        self.state_path.parent.mkdir(parents=True, exist_ok=True)

    @property
    def interval_columns(self) -> List[str]:
        return [f"collected_{minutes}" for minutes in self.config.intervals]

    def load(self) -> pd.DataFrame:
        if self.state_path.exists():
            state = pd.read_parquet(self.state_path)
        else:
            state = pd.DataFrame(columns=self.STATE_COLUMNS + self.interval_columns)
        return self._ensure_schema(state)

    def save(self, state: pd.DataFrame) -> None:
        state_to_save = self._ensure_schema(state).copy()
        state_to_save.to_parquet(self.state_path, index=False)

    def merge_new_posts(
        self,
        state: pd.DataFrame,
        new_posts: pd.DataFrame,
        observed_timestamp: float,
    ) -> pd.DataFrame:
        if new_posts is None or new_posts.empty:
            return self._ensure_schema(state)

        updated = self._ensure_schema(state).set_index("post_id", drop=False)

        for row in new_posts.itertuples(index=False):
            post_id = str(row.post_id)
            created_ts = float(row.created_timestamp)
            drop_after = created_ts + 60 * (
                self.config.intervals[-1] + self.config.grace_minutes
            )

            if post_id in updated.index:
                updated.loc[post_id, "created_timestamp"] = min(
                    created_ts, updated.loc[post_id, "created_timestamp"]
                )
                updated.loc[post_id, "first_seen_timestamp"] = min(
                    observed_timestamp, updated.loc[post_id, "first_seen_timestamp"]
                )
                updated.loc[post_id, "drop_after_timestamp"] = max(
                    drop_after, updated.loc[post_id, "drop_after_timestamp"]
                )
            else:
                new_entry = {
                    "post_id": post_id,
                    "subreddit": getattr(row, "subreddit", ""),
                    "created_timestamp": created_ts,
                    "first_seen_timestamp": observed_timestamp,
                    "last_snapshot_timestamp": np.nan,
                    "drop_after_timestamp": drop_after,
                }
                for col in self.interval_columns:
                    new_entry[col] = False
                updated.loc[post_id] = new_entry

        return self._ensure_schema(updated.reset_index(drop=True))

    def get_due_snapshot_masks(
        self, state: pd.DataFrame, current_timestamp: float
    ) -> Dict[int, pd.Series]:
        normalized_state = self._ensure_schema(state)
        if normalized_state.empty:
            return {minutes: pd.Series(dtype=bool) for minutes in self.config.intervals}

        age_minutes = (current_timestamp - normalized_state["created_timestamp"]) / 60.0
        masks: Dict[int, pd.Series] = {}

        for minutes in self.config.intervals:
            collected_col = f"collected_{minutes}"
            due_mask = (
                ~normalized_state[collected_col].astype(bool)
                & (age_minutes >= minutes - self.config.tolerance_minutes)
                & (age_minutes <= minutes + self.config.grace_minutes)
            )
            masks[minutes] = due_mask

        return masks

    def mark_collected(
        self,
        state: pd.DataFrame,
        post_ids: Iterable[str],
        minutes: int,
        snapshot_timestamp: float,
    ) -> pd.DataFrame:
        normalized_state = self._ensure_schema(state)
        if normalized_state.empty:
            return normalized_state

        collected_col = f"collected_{minutes}"
        if collected_col not in normalized_state.columns:
            normalized_state[collected_col] = False

        id_set = {str(pid) for pid in post_ids}
        mask = normalized_state["post_id"].astype(str).isin(id_set)
        if mask.any():
            normalized_state.loc[mask, collected_col] = True
            normalized_state.loc[mask, "last_snapshot_timestamp"] = snapshot_timestamp
        return normalized_state

    def prune(self, state: pd.DataFrame, current_timestamp: float) -> pd.DataFrame:
        normalized_state = self._ensure_schema(state)
        if normalized_state.empty:
            return normalized_state

        interval_cols = self.interval_columns
        all_collected = normalized_state[interval_cols].all(axis=1)
        expired = current_timestamp >= normalized_state["drop_after_timestamp"]
        keep_mask = ~(all_collected | expired)
        pruned = normalized_state.loc[keep_mask].reset_index(drop=True)
        return self._ensure_schema(pruned)

    def _ensure_schema(self, state: pd.DataFrame) -> pd.DataFrame:
        state = state.copy()
        for col in self.STATE_COLUMNS:
            if col not in state.columns:
                state[col] = np.nan

        for col in self.interval_columns:
            if col not in state.columns:
                state[col] = False
            state[col] = state[col].astype(bool)

        if not state.empty:
            state["post_id"] = state["post_id"].astype(str)
            state["subreddit"] = state["subreddit"].fillna("").astype(str)
            numeric_cols = [
                "created_timestamp",
                "first_seen_timestamp",
                "last_snapshot_timestamp",
                "drop_after_timestamp",
            ]
            for col in numeric_cols:
                state[col] = pd.to_numeric(state[col], errors="coerce")

        desired_cols = self.STATE_COLUMNS + self.interval_columns
        for col in desired_cols:
            if col not in state.columns:
                state[col] = np.nan

        return state[desired_cols]