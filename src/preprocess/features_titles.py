"""
Title feature extraction for virality analysis.

This module implements comprehensive title feature engineering including:
- Surface features (length, punctuation, capitalization)
- Sentiment analysis (VADER)
- Readability metrics (Flesch-Kincaid)
- Clickbait detection
- Named entity recognition
"""

import re
import string
from typing import Dict, List
import pandas as pd
import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import textstat
import spacy


class TitleFeatureExtractor:
    """Extract comprehensive features from post/story titles."""
    
    def __init__(self):
        """Initialize feature extractor with required models."""
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        
        # Load spaCy model (download with: python -m spacy download en_core_web_sm)
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            print("Warning: spaCy model not found. Install with: python -m spacy download en_core_web_sm")
            self.nlp = None
            
        # Clickbait keywords and patterns
        self.clickbait_keywords = [
            "you won't believe", "what happens next", "shocking", "amazing",
            "unbelievable", "incredible", "mind-blowing", "jaw-dropping",
            "this will", "you need to", "everyone should", "nobody talks about",
            "the truth about", "what nobody tells you", "secret", "revealed",
            "exposed", "conspiracy", "hidden", "leaked"
        ]
        
        self.clickbait_patterns = [
            r'\d+\s+(?:reasons|ways|things|facts|tips|tricks|secrets)',
            r'(?:how|why|what|when|where)\s+(?:to|you|we|they)',
            r'this\s+(?:is|will|might|could)',
            r'you\s+(?:won\'t|can\'t|shouldn\'t|must|should|need)',
        ]
        
    def extract_features(self, titles: pd.Series) -> pd.DataFrame:
        """
        Extract comprehensive features from titles.
        
        Args:
            titles: Series of title strings
            
        Returns:
            DataFrame with extracted features
        """
        features = pd.DataFrame()
        
        # Surface features
        features['title_length'] = titles.str.len()
        features['title_words'] = titles.str.split().str.len()
        features['title_chars_per_word'] = features['title_length'] / features['title_words']
        
        # Punctuation features
        features['has_question'] = titles.str.contains(r'\?', regex=True).astype(int)
        features['has_exclamation'] = titles.str.contains(r'!', regex=True).astype(int)
        features['punctuation_count'] = titles.apply(self._count_punctuation)
        
        # Capitalization features
        features['all_caps_words'] = titles.apply(self._count_all_caps_words)
        features['capitalization_ratio'] = titles.apply(self._capitalization_ratio)
        
        # Number and special character features
        features['number_count'] = titles.str.count(r'\d+')
        features['has_numbers'] = (features['number_count'] > 0).astype(int)
        
        # Sentiment features
        sentiment_scores = titles.apply(self._get_sentiment_scores)
        features['sentiment_compound'] = sentiment_scores.apply(lambda x: x['compound'])
        features['sentiment_positive'] = sentiment_scores.apply(lambda x: x['pos'])
        features['sentiment_negative'] = sentiment_scores.apply(lambda x: x['neg'])
        features['sentiment_neutral'] = sentiment_scores.apply(lambda x: x['neu'])
        
        # Readability features
        features['flesch_kincaid_grade'] = titles.apply(textstat.flesch_kincaid_grade)
        features['avg_word_length'] = titles.apply(self._avg_word_length)
        features['type_token_ratio'] = titles.apply(self._type_token_ratio)
        
        # Clickbait features
        features['clickbait_keywords'] = titles.apply(self._count_clickbait_keywords)
        features['clickbait_patterns'] = titles.apply(self._count_clickbait_patterns)
        features['has_clickbait'] = ((features['clickbait_keywords'] > 0) | 
                                   (features['clickbait_patterns'] > 0)).astype(int)
        
        # Named entity features (if spaCy is available)
        if self.nlp:
            entity_counts = titles.apply(self._count_entities)
            features['person_entities'] = entity_counts.apply(lambda x: x.get('PERSON', 0))
            features['org_entities'] = entity_counts.apply(lambda x: x.get('ORG', 0))
            features['date_entities'] = entity_counts.apply(lambda x: x.get('DATE', 0))
            features['total_entities'] = (features['person_entities'] + 
                                        features['org_entities'] + 
                                        features['date_entities'])
        else:
            features['person_entities'] = 0
            features['org_entities'] = 0  
            features['date_entities'] = 0
            features['total_entities'] = 0
            
        return features
    
    def _count_punctuation(self, text: str) -> int:
        """Count punctuation marks in text."""
        return sum(1 for char in text if char in string.punctuation)
    
    def _count_all_caps_words(self, text: str) -> int:
        """Count words that are entirely in capital letters."""
        words = text.split()
        return sum(1 for word in words if word.isupper() and len(word) > 1)
    
    def _capitalization_ratio(self, text: str) -> float:
        """Calculate ratio of uppercase letters to total letters."""
        letters = [char for char in text if char.isalpha()]
        if not letters:
            return 0.0
        return sum(1 for char in letters if char.isupper()) / len(letters)
    
    def _get_sentiment_scores(self, text: str) -> Dict[str, float]:
        """Get VADER sentiment scores."""
        return self.sentiment_analyzer.polarity_scores(text)
    
    def _avg_word_length(self, text: str) -> float:
        """Calculate average word length."""
        words = text.split()
        if not words:
            return 0.0
        return sum(len(word) for word in words) / len(words)
    
    def _type_token_ratio(self, text: str) -> float:
        """Calculate type-token ratio (unique words / total words)."""
        words = text.lower().split()
        if not words:
            return 0.0
        return len(set(words)) / len(words)
    
    def _count_clickbait_keywords(self, text: str) -> int:
        """Count clickbait keywords in text."""
        text_lower = text.lower()
        return sum(1 for keyword in self.clickbait_keywords if keyword in text_lower)
    
    def _count_clickbait_patterns(self, text: str) -> int:
        """Count clickbait patterns in text using regex."""
        text_lower = text.lower()
        count = 0
        for pattern in self.clickbait_patterns:
            if re.search(pattern, text_lower):
                count += 1
        return count
    
    def _count_entities(self, text: str) -> Dict[str, int]:
        """Count named entities by type using spaCy."""
        if not self.nlp:
            return {}
            
        doc = self.nlp(text)
        entity_counts = {}
        
        for ent in doc.ents:
            entity_type = ent.label_
            entity_counts[entity_type] = entity_counts.get(entity_type, 0) + 1
            
        return entity_counts