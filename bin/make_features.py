#!/usr/bin/env python3
"""
Feature engineering CLI tool.

Usage:
    python bin/make_features.py --input-dir ./data --output features.parquet
"""

import os
import sys
import click
import pandas as pd
import logging
from pathlib import Path
import glob

# Add src to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from preprocess.features_titles import TitleFeatureExtractor
from preprocess.features_context import ContextFeatureExtractor
from preprocess.normalize import DataNormalizer


@click.command()
@click.option('--input-dir', '-i', default='./data', help='Input directory with parquet files')
@click.option('--output', '-o', default='./data/features.parquet', help='Output file for features')
@click.option('--platform', type=click.Choice(['reddit', 'hackernews', 'both']), default='both', help='Which platform data to process')
def main(input_dir, output, platform):
    """Extract features from collected data."""
    
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    input_path = Path(input_dir)
    if not input_path.exists():
        logger.error(f"Input directory {input_path} does not exist")
        sys.exit(1)
    
    # Initialize feature extractors
    title_extractor = TitleFeatureExtractor()
    context_extractor = ContextFeatureExtractor()
    normalizer = DataNormalizer()
    
    logger.info(f"Processing {platform} data from {input_path}")
    
    try:
        # Find data files
        if platform == 'reddit':
            pattern = str(input_path / "reddit_*.parquet")
        elif platform == 'hackernews':
            pattern = str(input_path / "hackernews_*.parquet")
        else:
            pattern = str(input_path / "*.parquet")
        
        data_files = glob.glob(pattern)
        
        if not data_files:
            logger.error(f"No data files found matching pattern: {pattern}")
            sys.exit(1)
        
        logger.info(f"Found {len(data_files)} data files")
        
        # Load and combine all data
        all_data = []
        for file in data_files:
            if 'snapshots' not in file and 'features' not in file:  # Skip snapshot and feature files
                df = pd.read_parquet(file)
                all_data.append(df)
                logger.info(f"Loaded {len(df)} records from {file}")
        
        if not all_data:
            logger.error("No valid data files found")
            sys.exit(1)
        
        # Combine all data
        combined_df = pd.concat(all_data, ignore_index=True)
        logger.info(f"Combined dataset: {len(combined_df)} records")
        
        # Remove duplicates
        original_len = len(combined_df)
        combined_df = combined_df.drop_duplicates(subset=['post_id'])
        logger.info(f"Removed {original_len - len(combined_df)} duplicates")
        
        # Extract title features
        logger.info("Extracting title features...")
        title_features = title_extractor.extract_features(combined_df['title'])
        
        # Extract context features
        logger.info("Extracting context features...")
        context_features = context_extractor.extract_features(combined_df)
        
        # Combine all features
        features_df = pd.concat([
            combined_df,
            title_features,
            context_features
        ], axis=1)
        
        # Add snapshot features if available
        snapshot_files = glob.glob(str(input_path / "*snapshots*.parquet"))
        if snapshot_files:
            logger.info("Processing snapshot data for early score features...")
            all_snapshots = []
            for file in snapshot_files:
                snapshots = pd.read_parquet(file)
                all_snapshots.append(snapshots)
            
            if all_snapshots:
                combined_snapshots = pd.concat(all_snapshots, ignore_index=True)
                early_features = context_extractor.add_early_score_features(
                    combined_df, combined_snapshots
                )
                features_df = pd.concat([features_df, early_features], axis=1)
                logger.info("Added early score trajectory features")
        
        # Save features
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        features_df.to_parquet(output_path, index=False)
        logger.info(f"Features saved to {output_path}")
        
        # Print feature summary
        print(f"\nFeature Engineering Summary:")
        print(f"Total records: {len(features_df)}")
        print(f"Total features: {len(features_df.columns)}")
        print(f"Platforms: {features_df['platform'].value_counts().to_dict()}")
        print(f"Output file: {output_path}")
        
        # Show feature columns
        title_cols = [col for col in features_df.columns if any(x in col for x in ['title', 'sentiment', 'clickbait', 'length'])]
        context_cols = [col for col in features_df.columns if any(x in col for x in ['hour', 'day', 'age', 'author', 'content'])]
        
        print(f"\nTitle features ({len(title_cols)}): {title_cols[:10]}...")
        print(f"Context features ({len(context_cols)}): {context_cols[:10]}...")
        
    except Exception as e:
        logger.error(f"Error during feature engineering: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()