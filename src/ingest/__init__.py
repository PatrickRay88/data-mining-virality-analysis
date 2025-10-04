"""
Data ingestion module for Reddit and Hacker News APIs.
"""

from .reddit_client import RedditClient
from .hn_client import HackerNewsClient

__all__ = ["RedditClient", "HackerNewsClient"]