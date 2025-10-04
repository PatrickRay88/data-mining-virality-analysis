"""
Data normalization and harmonization between Reddit and Hacker News.

This module provides utilities to convert platform-specific data
into a unified schema for cross-platform analysis.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import hashlib


class DataNormalizer:
    """Normalize and harmonize data from different platforms."""
    
    def __init__(self):
        """Initialize data normalizer."""
        pass
        
    def normalize_reddit_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize Reddit data to unified schema.
        
        Args:
            df: DataFrame with Reddit post data
            
        Returns:
            DataFrame with normalized schema
        """
        normalized = pd.DataFrame()
        
        # Core fields
        normalized['platform'] = 'reddit'
        normalized['post_id'] = df['id']
        normalized['title'] = df['title']
        normalized['created_timestamp'] = df['created_utc']
        normalized['score'] = df['score']
        normalized['comment_count'] = df['num_comments']
        normalized['author_hash'] = df['author'].apply(self._hash_author)
        
        # Platform-specific fields
        normalized['upvote_ratio'] = df['upvote_ratio']
        normalized['subreddit'] = df['subreddit']
        normalized['is_self_post'] = df['is_self']
        normalized['url'] = df['url']
        normalized['domain'] = df.get('domain', '')
        
        # Optional fields
        if 'selftext' in df.columns:
            normalized['text_content'] = df['selftext']
        else:
            normalized['text_content'] = ''
            
        # Content flags
        normalized['is_nsfw'] = df.get('over_18', False)
        normalized['is_spoiler'] = df.get('spoiler', False)
        normalized['is_stickied'] = df.get('stickied', False)
        
        return normalized
    
    def normalize_hn_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize Hacker News data to unified schema.
        
        Args:
            df: DataFrame with HN story data
            
        Returns:
            DataFrame with normalized schema
        """
        normalized = pd.DataFrame()
        
        # Core fields
        normalized['platform'] = 'hackernews'
        normalized['post_id'] = df['id'].astype(str)
        normalized['title'] = df['title']
        normalized['created_timestamp'] = df['time']  # Already Unix timestamp
        normalized['score'] = df['score']
        normalized['comment_count'] = df['descendants']
        normalized['author_hash'] = df['by'].apply(self._hash_author)
        
        # HN doesn't have upvote ratio, so we estimate it
        normalized['upvote_ratio'] = np.nan
        normalized['subreddit'] = 'hackernews'  # Uniform platform identifier
        
        # Content detection
        normalized['is_self_post'] = df['url'].isna()  # Ask HN / Show HN posts
        normalized['url'] = df['url'].fillna('')
        normalized['domain'] = df['url'].str.extract(r'https?://(?:www\.)?([^/]+)')[0].fillna('')
        normalized['text_content'] = df.get('text', '')
        
        # HN doesn't have these flags
        normalized['is_nsfw'] = False
        normalized['is_spoiler'] = False
        normalized['is_stickied'] = False
        
        return normalized
    
    def combine_platforms(self, reddit_df: pd.DataFrame, 
                         hn_df: pd.DataFrame) -> pd.DataFrame:
        """
        Combine normalized data from both platforms.
        
        Args:
            reddit_df: Normalized Reddit DataFrame
            hn_df: Normalized Hacker News DataFrame
            
        Returns:
            Combined DataFrame with platform indicator
        """
        # Ensure both DataFrames have the same columns
        all_columns = set(reddit_df.columns) | set(hn_df.columns)
        
        for col in all_columns:
            if col not in reddit_df.columns:
                reddit_df[col] = np.nan
            if col not in hn_df.columns:
                hn_df[col] = np.nan
                
        # Reorder columns to match
        reddit_df = reddit_df.reindex(columns=sorted(all_columns))
        hn_df = hn_df.reindex(columns=sorted(all_columns))
        
        # Combine DataFrames
        combined = pd.concat([reddit_df, hn_df], ignore_index=True)
        
        return combined
    
    def save_to_parquet(self, df: pd.DataFrame, filepath: str, 
                       partition_cols: Optional[List[str]] = None) -> None:
        """
        Save DataFrame to Parquet format with optional partitioning.
        
        Args:
            df: DataFrame to save
            filepath: Output file path
            partition_cols: Columns to partition by (e.g., ['platform', 'date'])
        """
        # Ensure proper data types for Parquet
        df = df.copy()
        
        # Convert object columns to string where appropriate
        for col in df.select_dtypes(include=['object']).columns:
            if col not in ['created_timestamp']:  # Keep timestamp as numeric
                df[col] = df[col].astype(str)
        
        # Add date partition if timestamp available
        if 'created_timestamp' in df.columns and partition_cols and 'date' in partition_cols:
            df['date'] = pd.to_datetime(df['created_timestamp'], unit='s').dt.date
            
        # Save to Parquet
        if partition_cols:
            df.to_parquet(filepath, partition_cols=partition_cols, index=False)
        else:
            df.to_parquet(filepath, index=False)
    
    def load_from_parquet(self, filepath: str) -> pd.DataFrame:
        """Load DataFrame from Parquet file."""
        return pd.read_parquet(filepath)
    
    def _hash_author(self, author: str) -> str:
        """
        Create anonymized hash of author username for privacy.
        
        Args:
            author: Original username
            
        Returns:
            SHA256 hash of username (first 8 characters)
        """
        if pd.isna(author) or author in ['[deleted]', '']:
            return 'anonymous'
            
        # Create consistent hash
        hash_object = hashlib.sha256(str(author).encode())
        return hash_object.hexdigest()[:8]
    
    def add_quality_scores(self, df: pd.DataFrame, 
                          quality_scores: pd.Series) -> pd.DataFrame:
        """
        Add quality scores from Stage A modeling to dataset.
        
        Args:
            df: Main dataset
            quality_scores: Series with quality scores indexed by post_id
            
        Returns:
            DataFrame with quality scores added
        """
        df = df.copy()
        df['quality_score'] = df['post_id'].map(quality_scores)
        df['quality_score'] = df['quality_score'].fillna(df['quality_score'].median())
        
        return df
    
    def create_residual_target(self, df: pd.DataFrame) -> pd.Series:
        """
        Create residual lift target variable for Stage B analysis.
        
        Args:
            df: DataFrame with 'score' and 'quality_score' columns
            
        Returns:
            Series with residual lift values
        """
        if 'quality_score' not in df.columns:
            raise ValueError("DataFrame must contain 'quality_score' column from Stage A")
            
        # Calculate residual as log(observed) - log(predicted quality)
        observed_log = np.log1p(df['score'])  # log(1 + score) for stability
        predicted_log = np.log1p(df['quality_score'])
        
        residual_lift = observed_log - predicted_log
        
        return residual_lift