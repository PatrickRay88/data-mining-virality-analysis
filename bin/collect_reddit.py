#!/usr/bin/env python3
"""
Reddit data collection CLI tool.

Usage:
    python bin/collect_reddit.py --subreddit technology --days 30 --limit 1000
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

from ingest.reddit_client import RedditClient
from preprocess.normalize import DataNormalizer


@click.command()
@click.option('--subreddit', '-s', default='technology', help='Subreddit to collect from')
@click.option('--days', '-d', default=7, help='Number of days of data to collect')
@click.option('--limit', '-l', default=1000, help='Maximum posts to collect')
@click.option('--output-dir', '-o', default='./data', help='Output directory for data files')
@click.option('--snapshots', is_flag=True, help='Collect early score snapshots')
def main(subreddit, days, limit, output_dir, snapshots):
    """Collect Reddit data for virality analysis."""
    
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load environment variables
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        logger.warning("python-dotenv not installed. Make sure environment variables are set.")
    
    # Get Reddit API credentials
    client_id = os.getenv('REDDIT_CLIENT_ID')
    client_secret = os.getenv('REDDIT_CLIENT_SECRET')
    user_agent = os.getenv('REDDIT_USER_AGENT')
    
    if not all([client_id, client_secret, user_agent]):
        logger.error("Missing Reddit API credentials. Check your .env file.")
        sys.exit(1)
    
    # Initialize client and normalizer
    reddit_client = RedditClient(client_id, client_secret, user_agent)
    normalizer = DataNormalizer()
    
    logger.info(f"Collecting posts from r/{subreddit} for {days} days (limit: {limit})")
    
    try:
        # Determine time filter based on days
        if days <= 1:
            time_filter = 'day'
        elif days <= 7:
            time_filter = 'week'
        elif days <= 30:
            time_filter = 'month'
        else:
            time_filter = 'year'
        
        # Collect posts
        posts_df = reddit_client.collect_posts(
            subreddit=subreddit,
            limit=limit,
            time_filter=time_filter
        )
        
        logger.info(f"Collected {len(posts_df)} posts")
        
        # Normalize data
        normalized_df = normalizer.normalize_reddit_data(posts_df)
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"reddit_{subreddit}_{timestamp}.parquet"
        output_file = output_path / filename
        
        # Save to Parquet
        normalizer.save_to_parquet(normalized_df, str(output_file))
        logger.info(f"Data saved to {output_file}")
        
        # Collect snapshots if requested
        if snapshots and len(normalized_df) > 0:
            logger.info("Collecting early score snapshots...")
            
            # Get recent posts (created in last 2 hours)
            current_time = datetime.now().timestamp()
            recent_posts = normalized_df[
                (current_time - normalized_df['created_timestamp']) / 3600 <= 2
            ]
            
            if len(recent_posts) > 0:
                post_ids = recent_posts['post_id'].tolist()
                snapshots_df = reddit_client.get_post_snapshots(post_ids)
                
                if len(snapshots_df) > 0:
                    snapshot_filename = f"reddit_snapshots_{subreddit}_{timestamp}.parquet"
                    snapshot_file = output_path / snapshot_filename
                    snapshots_df.to_parquet(snapshot_file, index=False)
                    logger.info(f"Snapshots saved to {snapshot_file}")
                else:
                    logger.warning("No snapshots collected")
            else:
                logger.info("No recent posts found for snapshot collection")
        
        # Print summary statistics
        print(f"\nCollection Summary:")
        print(f"Posts collected: {len(normalized_df)}")
        print(f"Subreddit: r/{subreddit}")
        print(f"Score range: {normalized_df['score'].min()} - {normalized_df['score'].max()}")
        print(f"Average score: {normalized_df['score'].mean():.1f}")
        print(f"Output file: {output_file}")
        
    except Exception as e:
        logger.error(f"Error during collection: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()