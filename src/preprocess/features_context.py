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
        if 'created_utc' in df.columns:
            timestamp_col = 'created_utc'
        elif 'time' in df.columns:
            timestamp_col = 'time'
        else:
            raise ValueError("DataFrame must contain 'created_utc' or 'time' column")
            
        # Convert timestamp to datetime
        datetimes = pd.to_datetime(df[timestamp_col], unit='s', utc=True)
        
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
        age_hours = (current_time - df[timestamp_col]) / 3600
        features['age_hours'] = age_hours
        features['is_very_new'] = (age_hours <= 1).astype(int)
        features['is_recent'] = (age_hours <= 6).astype(int)
        features['is_old'] = (age_hours >= 24).astype(int)
        
        # Content type features (mainly for Reddit)
        if 'url' in df.columns and 'is_self' in df.columns:
            features['is_text_post'] = df['is_self'].astype(int)
            features['is_image_post'] = self._detect_image_content(df['url']).astype(int)
            features['is_news_post'] = self._detect_news_content(df['url']).astype(int)
            features['is_external_link'] = (~df['is_self']).astype(int)
        elif 'url' in df.columns:  # Hacker News
            features['has_url'] = df['url'].notna().astype(int)
            features['is_image_post'] = self._detect_image_content(df['url']).astype(int)
            features['is_news_post'] = self._detect_news_content(df['url']).astype(int)
            features['is_text_post'] = df['url'].isna().astype(int)  # HN Ask/Show posts
        
        # Author activity features (if available)
        if 'author' in df.columns or 'by' in df.columns:
            author_col = 'author' if 'author' in df.columns else 'by'
            features['has_author'] = df[author_col].notna().astype(int)
            
            # Author activity proxy (count posts by same author in dataset)
            author_counts = df[author_col].value_counts()
            features['author_post_count'] = df[author_col].map(author_counts).fillna(1)
            features['is_frequent_poster'] = (features['author_post_count'] >= 5).astype(int)
            
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
        is_image_ext = urls.str.contains(r'\.(jpg|jpeg|png|gif|webp|mp4|webm)(\?|$)', 
                                       case=False, na=False)
        
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
        
        # Calculate early growth metrics from snapshots
        for minutes in [5, 15, 30, 60]:
            # Get scores at specific time points
            snapshot_scores = self._get_scores_at_time(
                df, snapshots_df, target_minutes=minutes
            )
            features[f'score_at_{minutes}min'] = snapshot_scores
            
            # Calculate growth rates
            if minutes > 5:
                prev_scores = features[f'score_at_5min']
                growth = (snapshot_scores - prev_scores) / (minutes - 5)
                features[f'growth_rate_{minutes}min'] = growth.fillna(0)
        
        # Early momentum indicators
        features['early_momentum'] = features['score_at_15min'] / (features['score_at_5min'] + 1)
        features['sustained_growth'] = (features['growth_rate_30min'] > 0).astype(int)
        
        return features
    
    def _get_scores_at_time(self, posts_df: pd.DataFrame, 
                           snapshots_df: pd.DataFrame, 
                           target_minutes: int) -> pd.Series:
        """Get post scores at specific time after creation."""
        scores = pd.Series(index=posts_df.index, dtype=float)
        
        for post_id in posts_df.index:
            # Find snapshot closest to target time
            post_snapshots = snapshots_df[
                snapshots_df['post_id'] == post_id
            ].copy()
            
            if not post_snapshots.empty:
                # Find closest snapshot to target time
                time_diff = abs(post_snapshots['post_age_minutes'] - target_minutes)
                closest_idx = time_diff.idxmin()
                scores.loc[post_id] = post_snapshots.loc[closest_idx, 'score']
            else:
                # Fallback to final score if no snapshots available
                scores.loc[post_id] = posts_df.loc[post_id, 'score']
                
        return scores