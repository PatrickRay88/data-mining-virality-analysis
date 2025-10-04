"""
Reddit API client for collecting posts and early score snapshots.

This module uses PRAW to collect Reddit data with proper rate limiting
and ethical data handling practices.
"""

import time
import logging
from typing import List, Dict, Optional, Iterator
from datetime import datetime, timedelta
import praw
import pandas as pd
from tqdm import tqdm


class RedditClient:
    """Client for collecting Reddit data via PRAW API."""
    
    def __init__(self, client_id: str, client_secret: str, user_agent: str):
        """Initialize Reddit client with API credentials."""
        self.reddit = praw.Reddit(
            client_id=client_id,
            client_secret=client_secret,
            user_agent=user_agent
        )
        self.logger = logging.getLogger(__name__)
        
    def collect_posts(self, 
                     subreddit: str, 
                     limit: int = 1000,
                     time_filter: str = "week") -> pd.DataFrame:
        """
        Collect posts from a subreddit.
        
        Args:
            subreddit: Subreddit name (without r/)
            limit: Maximum number of posts to collect
            time_filter: Time filter for posts (hour, day, week, month, year, all)
            
        Returns:
            DataFrame with post data
        """
        posts_data = []
        
        try:
            sub = self.reddit.subreddit(subreddit)
            
            # Get top posts from the specified time period
            posts = sub.top(time_filter=time_filter, limit=limit)
            
            for post in tqdm(posts, desc=f"Collecting r/{subreddit} posts"):
                post_data = self._extract_post_data(post)
                posts_data.append(post_data)
                
                # Respectful rate limiting
                time.sleep(0.1)
                
        except Exception as e:
            self.logger.error(f"Error collecting posts from r/{subreddit}: {e}")
            raise
            
        return pd.DataFrame(posts_data)
    
    def collect_new_posts(self, subreddit: str, limit: int = 100) -> pd.DataFrame:
        """Collect new/rising posts for real-time monitoring."""
        posts_data = []
        
        try:
            sub = self.reddit.subreddit(subreddit)
            posts = sub.new(limit=limit)
            
            for post in posts:
                post_data = self._extract_post_data(post)
                post_data['collection_time'] = datetime.utcnow().timestamp()
                posts_data.append(post_data)
                
        except Exception as e:
            self.logger.error(f"Error collecting new posts: {e}")
            raise
            
        return pd.DataFrame(posts_data)
    
    def _extract_post_data(self, post) -> Dict:
        """Extract relevant data from a Reddit post object."""
        return {
            'id': post.id,
            'subreddit': str(post.subreddit),
            'title': post.title,
            'created_utc': post.created_utc,
            'author': str(post.author) if post.author else '[deleted]',
            'is_self': post.is_self,
            'url': post.url,
            'domain': post.domain if hasattr(post, 'domain') else None,
            'upvote_ratio': post.upvote_ratio,
            'score': post.score,
            'num_comments': post.num_comments,
            'selftext': post.selftext[:500] if post.selftext else '',  # Truncate long text
            'over_18': post.over_18,
            'spoiler': post.spoiler,
            'stickied': post.stickied
        }
        
    def get_post_snapshots(self, 
                          post_ids: List[str], 
                          intervals: List[int] = [5, 15, 30, 60]) -> pd.DataFrame:
        """
        Collect score snapshots for posts at specified intervals (minutes).
        
        Args:
            post_ids: List of Reddit post IDs
            intervals: Minutes after creation to take snapshots
            
        Returns:
            DataFrame with snapshot data
        """
        snapshots = []
        
        for post_id in tqdm(post_ids, desc="Collecting snapshots"):
            try:
                post = self.reddit.submission(id=post_id)
                
                # Get current timestamp
                now = datetime.utcnow().timestamp()
                post_age_minutes = (now - post.created_utc) / 60
                
                # Only collect if post is still within our monitoring window
                if post_age_minutes <= max(intervals) + 10:  # 10 min buffer
                    snapshot_data = {
                        'post_id': post_id,
                        'snapshot_time': now,
                        'post_age_minutes': post_age_minutes,
                        'score': post.score,
                        'upvote_ratio': post.upvote_ratio,
                        'num_comments': post.num_comments
                    }
                    snapshots.append(snapshot_data)
                    
                time.sleep(0.2)  # Rate limiting
                    
            except Exception as e:
                self.logger.warning(f"Could not get snapshot for post {post_id}: {e}")
                continue
                
        return pd.DataFrame(snapshots)