"""
Context feature extraction for virality analysis.

This module extracts contextual features including:
- Time-based features (hour, weekday, recency)
- Content type detection
- Author activity proxies
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Optional
import math
import click


class ContextFeatureExtractor:
    """Extract contextual features from post metadata."""
    
    def __init__(self):
        """Initialize context feature extractor."""
        # Domain patterns for content type detection
        self.image_domains = {
            'i.redd.it', 'imgur.com', 'i.imgur.com', 'gfycat.com', 
            'v.redd.it', 'giphy.com', 'flickr.com', 'instagram.com'
        }
        
        self.news_domains = {
            'bbc.com', 'cnn.com', 'nytimes.com', 'washingtonpost.com',
            'reuters.com', 'apnews.com', 'theguardian.com', 'npr.org'
        }
        
    def extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract contextual features from post/story data.
        
        Args:
            df: DataFrame with post data including created_utc/time, url, author info
            
        Returns:
            DataFrame with extracted contextual features
        """
        features = pd.DataFrame(index=df.index)
        
        # Time features
        if 'created_timestamp' in df.columns:
            timestamp_col = 'created_timestamp'
        elif 'created_utc' in df.columns:
            timestamp_col = 'created_utc'
        elif 'time' in df.columns:
            timestamp_col = 'time'
        else:
            raise ValueError("DataFrame must contain 'created_timestamp', 'created_utc', or 'time' column")
            
        timestamps = pd.to_numeric(df[timestamp_col], errors='coerce')
        datetimes = pd.to_datetime(timestamps, unit='s', utc=True)
        
        # Hour of day features
        features['hour_of_day'] = datetimes.dt.hour
        features['is_morning'] = (features['hour_of_day'].between(6, 11)).astype(int)
        features['is_afternoon'] = (features['hour_of_day'].between(12, 17)).astype(int)
        features['is_evening'] = (features['hour_of_day'].between(18, 23)).astype(int)
        features['is_night'] = (~features['hour_of_day'].between(6, 23)).astype(int)
        
        # Day of week features
        features['day_of_week'] = datetimes.dt.dayofweek  # Monday=0
        features['is_weekend'] = (features['day_of_week'].isin([5, 6])).astype(int)
        features['is_monday'] = (features['day_of_week'] == 0).astype(int)
        features['is_friday'] = (features['day_of_week'] == 4).astype(int)
        
        # Recency buckets (useful for quality modeling)
        current_time = datetime.utcnow().timestamp()
        age_hours = (current_time - timestamps) / 3600
        features['age_hours'] = age_hours
        features['is_very_new'] = (age_hours <= 1).astype(int)
        features['is_recent'] = (age_hours <= 6).astype(int)
        features['is_old'] = (age_hours >= 24).astype(int)
        features['log_age_hours'] = np.log1p(np.clip(age_hours, a_min=0, a_max=None))

        # Cyclical encoding for smoother hour-of-day modeling
        features['hour_sin'] = np.sin(2 * math.pi * features['hour_of_day'] / 24.0)
        features['hour_cos'] = np.cos(2 * math.pi * features['hour_of_day'] / 24.0)
        
        # Content type features (mainly for Reddit)
        self_col = None
        if 'is_self_post' in df.columns:
            self_col = 'is_self_post'
        elif 'is_self' in df.columns:
            self_col = 'is_self'

        if 'url' in df.columns and self_col:
            is_self = df[self_col].fillna(False).astype(bool)
            features['is_text_post'] = is_self.astype(int)
            features['is_image_post'] = self._detect_image_content(df['url']).astype(int)
            features['is_news_post'] = self._detect_news_content(df['url']).astype(int)
            features['is_external_link'] = (~is_self).astype(int)
        elif 'url' in df.columns:  # Hacker News
            features['has_url'] = df['url'].notna().astype(int)
            features['is_image_post'] = self._detect_image_content(df['url']).astype(int)
            features['is_news_post'] = self._detect_news_content(df['url']).astype(int)
            features['is_text_post'] = df['url'].isna().astype(int)  # HN Ask/Show posts
        
        # Author activity features (if available)
        if 'author' in df.columns or 'by' in df.columns or 'author_hash' in df.columns:
            if 'author' in df.columns:
                author_col = 'author'
            elif 'by' in df.columns:
                author_col = 'by'
            else:
                author_col = 'author_hash'

            author_series = df[author_col]
            features['has_author'] = author_series.notna().astype(int)
            
            # Author activity proxy (count posts by same author in dataset)
            author_counts = author_series.value_counts()
            features['author_post_count'] = author_series.map(author_counts).fillna(1)
            features['is_frequent_poster'] = (features['author_post_count'] >= 5).astype(int)
            features['author_post_count_log'] = np.log1p(features['author_post_count'])

        if 'subreddit' in df.columns:
            subreddit_counts = df['subreddit'].value_counts()
            features['subreddit_post_count'] = df['subreddit'].map(subreddit_counts).fillna(0)
            
        return features
    
    def _detect_image_content(self, urls: pd.Series) -> pd.Series:
        """Detect if URLs point to image/media content."""
        if urls.isna().all():
            return pd.Series(False, index=urls.index)
            
        # Extract domain from URL
        domains = urls.str.extract(r'https?://(?:www\.)?([^/]+)')[0]
        
        # Check against known image domains
        is_image = domains.isin(self.image_domains)
        
        # Also check file extensions
        is_image_ext = urls.str.contains(
            r'\.(?:jpg|jpeg|png|gif|webp|mp4|webm)(?:\?|$)',
            case=False,
            na=False,
        )

        return is_image | is_image_ext
    
    def _detect_news_content(self, urls: pd.Series) -> pd.Series:
        """Detect if URLs point to news content."""
        if urls.isna().all():
            return pd.Series(False, index=urls.index)
            
        domains = urls.str.extract(r'https?://(?:www\.)?([^/]+)')[0]
        return domains.isin(self.news_domains)
    
    def add_early_score_features(self, df: pd.DataFrame,
                                snapshots_df: pd.DataFrame) -> pd.DataFrame:
        """
        Add early score trajectory features for quality modeling.
        
        Args:
            df: Main posts DataFrame
            snapshots_df: Early score snapshots DataFrame
            
        Returns:
            DataFrame with early score features added
        """
        features = pd.DataFrame(index=df.index)

        if 'post_id' not in df.columns:
            raise ValueError("DataFrame must contain 'post_id' column for snapshot alignment")
        if snapshots_df.empty:
            return features

        # Ensure snapshots have consistent types
        snapshots_local = snapshots_df.copy()
        snapshots_local['post_id'] = snapshots_local['post_id'].astype(str)
        df_post_ids = df['post_id'].astype(str)

        for minutes in [5, 15, 30, 60]:
            snapshot_scores = self._get_scores_at_time(
                df_post_ids,
                df['score'],
                snapshots_local,
                target_minutes=minutes,
                progress_label=f"Aligning {len(df_post_ids)} posts at {minutes} minutes",
            )
            features[f'score_at_{minutes}min'] = snapshot_scores.reindex(df.index)

            if minutes > 5:
                prev_scores = features.get('score_at_5min', pd.Series(0.0, index=df.index))
                growth = (features[f'score_at_{minutes}min'] - prev_scores) / (minutes - 5)
                features[f'growth_rate_{minutes}min'] = growth.fillna(0)

        features['early_momentum'] = (
            features.get('score_at_15min', pd.Series(0.0, index=df.index)) /
            (features.get('score_at_5min', pd.Series(0.0, index=df.index)) + 1)
        )
        features['sustained_growth'] = (
            features.get('growth_rate_30min', pd.Series(0.0, index=df.index)) > 0
        ).astype(int)

        return features

    def _get_scores_at_time(
        self,
        post_ids: pd.Series,
        fallback_scores: pd.Series,
        snapshots_df: pd.DataFrame,
        target_minutes: int,
        progress_label: Optional[str] = None,
    ) -> pd.Series:
        """Get post scores at specific time after creation."""
        scores = pd.Series(index=post_ids.index, dtype=float)

        if progress_label:
            indices = post_ids.index.to_list()
            values = post_ids.to_list()
            with click.progressbar(
                range(len(values)), length=len(values), label=progress_label
            ) as bar:
                for pos in bar:
                    idx = indices[pos]
                    post_id = values[pos]
                    post_snapshots = snapshots_df[snapshots_df['post_id'] == post_id]

                    if not post_snapshots.empty:
                        time_diff = (
                            post_snapshots['post_age_minutes'] - target_minutes
                        ).abs()
                        closest_idx = time_diff.idxmin()
                        scores.loc[idx] = post_snapshots.loc[closest_idx, 'score']
                    else:
                        scores.loc[idx] = fallback_scores.loc[idx]
        else:
            for idx, post_id in post_ids.items():
                post_snapshots = snapshots_df[snapshots_df['post_id'] == post_id]

                if not post_snapshots.empty:
                    time_diff = (
                        post_snapshots['post_age_minutes'] - target_minutes
                    ).abs()
                    closest_idx = time_diff.idxmin()
                    scores.loc[idx] = post_snapshots.loc[closest_idx, 'score']
                else:
                    scores.loc[idx] = fallback_scores.loc[idx]

        return scores