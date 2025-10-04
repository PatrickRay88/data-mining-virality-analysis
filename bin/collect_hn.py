#!/usr/bin/env python3
"""
Hacker News data collection CLI tool.

Usage:
    python bin/collect_hn.py --days 30 --limit 1000
"""

import os
import sys
import click
import pandas as pd
from datetime import datetime, timedelta
import logging
from pathlib import Path

# Add src to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from ingest.hn_client import HackerNewsClient
from preprocess.normalize import DataNormalizer


@click.command()
@click.option('--days', '-d', default=7, help='Number of days of data to collect')
@click.option('--limit', '-l', default=1000, help='Maximum stories to collect')
@click.option('--output-dir', '-o', default='./data', help='Output directory for data files')
@click.option('--story-type', default='new', type=click.Choice(['new', 'top']), help='Type of stories to collect')
@click.option('--snapshots', is_flag=True, help='Collect early score snapshots')
def main(days, limit, output_dir, story_type, snapshots):
    """Collect Hacker News data for virality analysis."""
    
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Initialize client and normalizer
    hn_client = HackerNewsClient()
    normalizer = DataNormalizer()
    
    logger.info(f"Collecting {story_type} stories from Hacker News for {days} days (limit: {limit})")
    
    try:
        # Collect stories
        if story_type == 'new':
            stories_df = hn_client.collect_new_stories(limit=limit)
        else:
            stories_df = hn_client.collect_top_stories(limit=limit)
        
        logger.info(f"Collected {len(stories_df)} stories")
        
        if len(stories_df) == 0:
            logger.warning("No stories collected")
            return
        
        # Filter by date if specified
        if days < 365:  # Only filter for reasonable time periods
            cutoff_time = (datetime.now() - timedelta(days=days)).timestamp()
            stories_df = stories_df[stories_df['time'] >= cutoff_time]
            logger.info(f"Filtered to {len(stories_df)} stories from last {days} days")
        
        # Normalize data
        normalized_df = normalizer.normalize_hn_data(stories_df)
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"hackernews_{story_type}_{timestamp}.parquet"
        output_file = output_path / filename
        
        # Save to Parquet
        normalizer.save_to_parquet(normalized_df, str(output_file))
        logger.info(f"Data saved to {output_file}")
        
        # Collect snapshots if requested
        if snapshots and len(normalized_df) > 0:
            logger.info("Collecting early score snapshots...")
            
            # Get recent stories (created in last 2 hours)
            current_time = datetime.now().timestamp()
            recent_stories = normalized_df[
                (current_time - normalized_df['created_timestamp']) / 3600 <= 2
            ]
            
            if len(recent_stories) > 0:
                story_ids = recent_stories['post_id'].astype(int).tolist()
                snapshots_df = hn_client.get_story_snapshots(story_ids)
                
                if len(snapshots_df) > 0:
                    snapshot_filename = f"hn_snapshots_{story_type}_{timestamp}.parquet"
                    snapshot_file = output_path / snapshot_filename
                    snapshots_df.to_parquet(snapshot_file, index=False)
                    logger.info(f"Snapshots saved to {snapshot_file}")
                else:
                    logger.warning("No snapshots collected")
            else:
                logger.info("No recent stories found for snapshot collection")
        
        # Print summary statistics
        print(f"\nCollection Summary:")
        print(f"Stories collected: {len(normalized_df)}")
        print(f"Story type: {story_type}")
        print(f"Score range: {normalized_df['score'].min()} - {normalized_df['score'].max()}")
        print(f"Average score: {normalized_df['score'].mean():.1f}")
        print(f"Time range: {pd.to_datetime(normalized_df['created_timestamp'], unit='s').min()} to {pd.to_datetime(normalized_df['created_timestamp'], unit='s').max()}")
        print(f"Output file: {output_file}")
        
    except Exception as e:
        logger.error(f"Error during collection: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()