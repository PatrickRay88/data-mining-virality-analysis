"""
Reddit API client for collecting posts and early score snapshots.

This module uses PRAW to collect Reddit data with proper rate limiting
and ethical data handling practices.
"""

import time
import logging
from typing import List, Dict
from datetime import datetime, timezone

import pandas as pd
import praw
from prawcore import exceptions as praw_exceptions
from tqdm import tqdm
from requests import exceptions as requests_exceptions


class RedditClient:
    """Client for collecting Reddit data via PRAW API."""
    
    def __init__(self, client_id: str, client_secret: str, user_agent: str,
                 retry_attempts: int = 3, retry_backoff: float = 2.0):
        """Initialize Reddit client with API credentials."""
        self.reddit = praw.Reddit(
            client_id=client_id,
            client_secret=client_secret,
            user_agent=user_agent,
        )
        self.logger = logging.getLogger(__name__)
        self.retry_attempts = max(1, retry_attempts)
        self.retry_backoff = max(1.0, retry_backoff)
        
    def collect_posts(
        self,
        subreddit: str,
        limit: int = 1000,
        time_filter: str = "week",
    ) -> pd.DataFrame:
        """Collect posts from a subreddit."""
        posts_data: List[Dict] = []

        for attempt in range(1, self.retry_attempts + 1):
            try:
                sub = self.reddit.subreddit(subreddit)
                posts = sub.top(time_filter=time_filter, limit=limit)

                for post in tqdm(posts, desc=f"Collecting r/{subreddit} posts"):
                    post_data = self._extract_post_data(post)
                    posts_data.append(post_data)
                    time.sleep(0.1)

                break
            except (
                praw_exceptions.ServerError,
                praw_exceptions.RequestException,
                praw_exceptions.ResponseException,
                requests_exceptions.RequestException,
            ) as exc:
                self.logger.warning(
                    "Attempt %s/%s failed for r/%s top feed: %s",
                    attempt,
                    self.retry_attempts,
                    subreddit,
                    exc,
                )
                if attempt == self.retry_attempts:
                    self.logger.error(
                        "Exceeded retry budget while collecting r/%s top posts; returning partial results.",
                        subreddit,
                    )
                else:
                    time.sleep(self.retry_backoff * attempt)
            except Exception as exc:  # pragma: no cover - unexpected
                self.logger.error("Error collecting posts from r/%s: %s", subreddit, exc)
                raise

        return pd.DataFrame(posts_data)
    
    def collect_new_posts(self, subreddit: str, limit: int = 100) -> pd.DataFrame:
        """Collect new/rising posts for real-time monitoring."""
        posts_data: List[Dict] = []

        for attempt in range(1, self.retry_attempts + 1):
            try:
                sub = self.reddit.subreddit(subreddit)
                posts = sub.new(limit=limit)

                for post in posts:
                    post_data = self._extract_post_data(post)
                    post_data["collection_time"] = datetime.now(timezone.utc).timestamp()
                    posts_data.append(post_data)

                break
            except (
                praw_exceptions.ServerError,
                praw_exceptions.RequestException,
                praw_exceptions.ResponseException,
                requests_exceptions.RequestException,
            ) as exc:
                self.logger.warning(
                    "Attempt %s/%s failed for r/%s new feed: %s",
                    attempt,
                    self.retry_attempts,
                    subreddit,
                    exc,
                )
                if attempt == self.retry_attempts:
                    self.logger.error(
                        "Exceeded retry budget while collecting r/%s new posts; returning partial results.",
                        subreddit,
                    )
                else:
                    time.sleep(self.retry_backoff * attempt)
            except Exception as exc:  # pragma: no cover - unexpected
                self.logger.error("Error collecting new posts: %s", exc)
                raise

        return pd.DataFrame(posts_data)

    def fetch_posts_by_ids(self, post_ids: List[str]) -> pd.DataFrame:
        """Fetch the latest metadata for a list of post IDs."""
        if not post_ids:
            return pd.DataFrame()

        records: List[Dict] = []
        fullname_iter = [f"t3_{post_id}" for post_id in post_ids]

        try:
            for submission in self.reddit.info(fullnames=fullname_iter):
                post_record = self._extract_post_data(submission)
                post_record['fetched_utc'] = datetime.now(timezone.utc).timestamp()
                records.append(post_record)
                time.sleep(0.1)
        except Exception as exc:
            self.logger.error(f"Error fetching posts by id: {exc}")
            raise

        return pd.DataFrame(records)
    
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
                now = datetime.now(timezone.utc).timestamp()
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