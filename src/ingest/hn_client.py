"""
Hacker News API client for collecting stories and comments.

This module uses the free Hacker News API to collect story data
for cross-platform analysis.
"""

import requests
import time
import logging
from typing import List, Dict, Optional
from datetime import datetime
import pandas as pd
from tqdm import tqdm


class HackerNewsClient:
    """Client for collecting Hacker News data via public API."""
    
    BASE_URL = "https://hacker-news.firebaseio.com/v0"
    
    def __init__(self):
        """Initialize Hacker News client."""
        self.session = requests.Session()
        self.logger = logging.getLogger(__name__)
        
    def collect_new_stories(self, limit: int = 500) -> pd.DataFrame:
        """
        Collect recent stories from Hacker News.
        
        Args:
            limit: Maximum number of stories to collect
            
        Returns:
            DataFrame with story data
        """
        stories_data = []
        
        try:
            # Get list of new story IDs
            response = self.session.get(f"{self.BASE_URL}/newstories.json")
            response.raise_for_status()
            story_ids = response.json()[:limit]
            
            # Collect detailed data for each story
            for story_id in tqdm(story_ids, desc="Collecting HN stories"):
                story_data = self._get_story_details(story_id)
                if story_data:
                    stories_data.append(story_data)
                    
                # Respectful rate limiting
                time.sleep(0.05)
                
        except Exception as e:
            self.logger.error(f"Error collecting HN stories: {e}")
            raise
            
        return pd.DataFrame(stories_data)
    
    def collect_top_stories(self, limit: int = 500) -> pd.DataFrame:
        """Collect top stories from Hacker News."""
        stories_data = []
        
        try:
            response = self.session.get(f"{self.BASE_URL}/topstories.json")
            response.raise_for_status()
            story_ids = response.json()[:limit]
            
            for story_id in tqdm(story_ids, desc="Collecting HN top stories"):
                story_data = self._get_story_details(story_id)
                if story_data:
                    stories_data.append(story_data)
                    
                time.sleep(0.05)
                
        except Exception as e:
            self.logger.error(f"Error collecting HN top stories: {e}")
            raise
            
        return pd.DataFrame(stories_data)
    
    def _get_story_details(self, story_id: int) -> Optional[Dict]:
        """
        Get detailed information for a specific story.
        
        Args:
            story_id: Hacker News story ID
            
        Returns:
            Dictionary with story data or None if error
        """
        try:
            response = self.session.get(f"{self.BASE_URL}/item/{story_id}.json")
            response.raise_for_status()
            item = response.json()
            
            # Only process stories (not comments, jobs, etc.)
            if not item or item.get('type') != 'story':
                return None
                
            return {
                'id': item.get('id'),
                'title': item.get('title', ''),
                'url': item.get('url', ''),
                'text': item.get('text', '')[:500] if item.get('text') else '',  # Truncate
                'score': item.get('score', 0),
                'by': item.get('by', ''),  # Author username
                'time': item.get('time', 0),  # Unix timestamp
                'descendants': item.get('descendants', 0),  # Comment count
                'kids': len(item.get('kids', [])),  # Direct replies
                'dead': item.get('dead', False),
                'deleted': item.get('deleted', False)
            }
            
        except Exception as e:
            self.logger.warning(f"Could not get details for story {story_id}: {e}")
            return None
    
    def get_story_snapshots(self, 
                           story_ids: List[int], 
                           intervals: List[int] = [5, 15, 30, 60]) -> pd.DataFrame:
        """
        Collect score snapshots for stories at specified intervals.
        
        Args:
            story_ids: List of HN story IDs
            intervals: Minutes after creation to take snapshots
            
        Returns:
            DataFrame with snapshot data
        """
        snapshots = []
        
        for story_id in tqdm(story_ids, desc="Collecting HN snapshots"):
            story_data = self._get_story_details(story_id)
            
            if story_data:
                now = datetime.utcnow().timestamp()
                story_age_minutes = (now - story_data['time']) / 60
                
                # Only collect if story is within monitoring window
                if story_age_minutes <= max(intervals) + 10:
                    snapshot_data = {
                        'story_id': story_id,
                        'snapshot_time': now,
                        'story_age_minutes': story_age_minutes,
                        'score': story_data['score'],
                        'descendants': story_data['descendants']
                    }
                    snapshots.append(snapshot_data)
                    
            time.sleep(0.1)
            
        return pd.DataFrame(snapshots)
    
    def collect_historical_data(self, 
                               start_date: datetime, 
                               end_date: datetime,
                               max_stories: int = 10000) -> pd.DataFrame:
        """
        Collect historical stories within a date range.
        Note: This is a best-effort approach since HN API doesn't have date filtering.
        
        Args:
            start_date: Start date for collection
            end_date: End date for collection
            max_stories: Maximum stories to check
            
        Returns:
            DataFrame with stories from the specified period
        """
        stories_data = []
        start_ts = start_date.timestamp()
        end_ts = end_date.timestamp()
        
        try:
            # Get a large sample of story IDs
            response = self.session.get(f"{self.BASE_URL}/topstories.json")
            response.raise_for_status()
            story_ids = response.json()[:max_stories]
            
            for story_id in tqdm(story_ids, desc="Filtering historical stories"):
                story_data = self._get_story_details(story_id)
                
                if story_data and start_ts <= story_data['time'] <= end_ts:
                    stories_data.append(story_data)
                    
                time.sleep(0.05)
                
                # Stop if we have enough data
                if len(stories_data) >= 1000:
                    break
                    
        except Exception as e:
            self.logger.error(f"Error collecting historical HN data: {e}")
            raise
            
        return pd.DataFrame(stories_data)