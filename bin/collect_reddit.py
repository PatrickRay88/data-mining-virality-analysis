#!/usr/bin/env python3
"""Reddit data collection CLI tool.

Usage examples::

    python bin/collect_reddit.py --subreddit technology --days 30 --limit 1000
    python bin/collect_reddit.py -s technology -s science -s worldnews --days 14
"""

import os
import sys
import click
import pandas as pd
from datetime import datetime
import logging
from pathlib import Path

# Ensure project root on path for local package imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.ingest.reddit_client import RedditClient
from src.preprocess.normalize import DataNormalizer


@click.command()
@click.option(
    '--subreddit',
    '-s',
    multiple=True,
    default=('technology',),
    show_default=True,
    help='One or more subreddits to collect from (use multiple -s flags).',
)
@click.option('--days', '-d', default=7, help='Number of days of data to collect')
@click.option('--limit', '-l', default=1000, help='Maximum top posts to collect per subreddit')
@click.option('--output-dir', '-o', default='./data', help='Output directory for data files')
@click.option('--snapshots', is_flag=True, help='Collect early score snapshots')
@click.option('--include-new', is_flag=True, help='Also collect freshest posts from /new for snapshot coverage')
@click.option('--new-limit', default=100, show_default=True, help='Maximum new posts to collect per subreddit when --include-new is set')
def main(subreddit, days, limit, output_dir, snapshots, include_new, new_limit):
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

    subreddits = subreddit or ('technology',)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    combined_frames = []
    summaries = []

    logger.info(
        "Collecting posts from %s for %d days (limit per subreddit: %d)",
        ', '.join(f"r/{name}" for name in subreddits),
        days,
        limit,
    )

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

        total_collected = 0

        for sub_name in subreddits:
            logger.info("Collecting r/%s with time filter '%s'", sub_name, time_filter)

            posts_df = reddit_client.collect_posts(
                subreddit=sub_name,
                limit=limit,
                time_filter=time_filter,
            )

            new_posts_df = pd.DataFrame()
            if include_new:
                logger.info("Collecting newest posts from r/%s (limit %d)", sub_name, new_limit)
                new_posts_df = reddit_client.collect_new_posts(sub_name, limit=new_limit)
                logger.info("Collected %d new posts from r/%s", len(new_posts_df), sub_name)

            logger.info("Collected %d posts from r/%s", len(posts_df), sub_name)

            if posts_df.empty and new_posts_df.empty:
                summaries.append(
                    {
                        'subreddit': sub_name,
                        'posts': 0,
                        'min_score': None,
                        'max_score': None,
                        'avg_score': None,
                        'output_file': None,
                    }
                )
                continue

            normalized_frames = []
            if not posts_df.empty:
                normalized_top = normalizer.normalize_reddit_data(posts_df)
                normalized_top['collection_type'] = 'top'
                normalized_frames.append(normalized_top)
            if include_new and not new_posts_df.empty:
                normalized_new = normalizer.normalize_reddit_data(new_posts_df)
                normalized_new['collection_type'] = 'new'
                normalized_frames.append(normalized_new)

            if normalized_frames:
                normalized_df = pd.concat(normalized_frames, ignore_index=True)
                normalized_df = normalized_df.drop_duplicates(subset=['post_id'])
                combined_frames.append(normalized_df)
                total_collected += len(normalized_df)

                filename = f"reddit_{sub_name}_{timestamp}.parquet"
                output_file = output_path / filename
                normalizer.save_to_parquet(normalized_df, str(output_file))
                logger.info("Data saved to %s", output_file)

                min_score = normalized_df['score'].min()
                max_score = normalized_df['score'].max()
                avg_score = normalized_df['score'].mean()
                new_count = int((normalized_df['collection_type'] == 'new').sum())
                top_count = len(normalized_df) - new_count
            else:
                normalized_df = pd.DataFrame()
                min_score = None
                max_score = None
                avg_score = None
                new_count = 0
                top_count = 0

            summaries.append(
                {
                    'subreddit': sub_name,
                    'posts': len(normalized_df),
                    'top_posts': top_count,
                    'new_posts': new_count,
                    'min_score': min_score,
                    'max_score': max_score,
                    'avg_score': avg_score,
                    'output_file': output_path / f"reddit_{sub_name}_{timestamp}.parquet" if not normalized_df.empty else None,
                }
            )

            if snapshots and not normalized_df.empty:
                logger.info("Collecting early score snapshots for r/%s", sub_name)

                current_time = datetime.now().timestamp()
                recent_posts = normalized_df[
                    (current_time - normalized_df['created_timestamp']) / 3600 <= 2
                ]

                if len(recent_posts) > 0:
                    post_ids = recent_posts['post_id'].tolist()
                    snapshots_df = reddit_client.get_post_snapshots(post_ids)

                    if len(snapshots_df) > 0:
                        snapshot_filename = f"reddit_snapshots_{sub_name}_{timestamp}.parquet"
                        snapshot_file = output_path / snapshot_filename
                        snapshots_df.to_parquet(snapshot_file, index=False)
                        logger.info("Snapshots saved to %s", snapshot_file)
                    else:
                        logger.warning("No snapshots collected for r/%s", sub_name)
                else:
                    logger.info("No recent posts in r/%s for snapshot collection", sub_name)

        combined_file = None
        if len(combined_frames) > 1:
            combined_df = pd.concat(combined_frames, ignore_index=True)
            combined_df = combined_df.drop_duplicates(subset=['post_id'])
            combined_file = output_path / f"reddit_multi_{len(combined_frames)}subs_{timestamp}.parquet"
            normalizer.save_to_parquet(combined_df, str(combined_file))
            logger.info("Combined dataset saved to %s", combined_file)

        # Print summary statistics
        print("\nCollection Summary:")
        for item in summaries:
            print(f"- Subreddit: r/{item['subreddit']}")
            print(f"  Posts collected: {item['posts']}")
            if item['posts'] > 0 and include_new:
                print(f"    Top posts: {item['top_posts']} | New posts: {item['new_posts']}")
            if item['posts'] > 0:
                print(f"  Score range: {item['min_score']} - {item['max_score']}")
                print(f"  Average score: {item['avg_score']:.1f}")
                print(f"  Output file: {item['output_file']}")
            else:
                print("  No posts collected")

        if combined_file is not None:
            print(f"\nCombined dataset: {combined_file}")
            print(f"Total posts across subreddits: {total_collected}")

    except Exception as e:
        logger.error(f"Error during collection: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()