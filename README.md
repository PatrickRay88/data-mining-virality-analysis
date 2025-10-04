# Content Virality Analysis

A data mining project that extends Weissburg 2022's research on content virality by analyzing Reddit and Hacker News data with enhanced features and residual-lift analysis.

## Research Objectives

This project replicates and extends Weissburg 2022 on Reddit with three key enhancements:

1. **Enhanced Features**: Time-based features (hour, weekday, recency), content-type classification, and author activity proxies
2. **Cross-Platform Analysis**: Hacker News data integration to test title effect transferability
3. **Residual-Lift Methodology**: Exposure-aware quality estimation followed by title-driven lift analysis

> **Academic Context**: This project is part of a graduate-level data mining course, implementing methodological extensions suggested in the original Weissburg 2022 paper.

## Project Structure

```
├── src/                    # Core Python package
│   ├── ingest/            # API clients for Reddit and Hacker News
│   ├── preprocess/        # Feature engineering and data normalization
│   ├── models/            # Quality modeling and residual-lift analysis
│   └── utils/             # Shared utilities (time bins, calibration, evaluation)
├── notebooks/             # Jupyter notebooks for exploration and visualization
├── bin/                   # CLI entry points for data collection and modeling
├── data/                  # Raw and processed data storage (Parquet format)
└── tests/                 # Unit tests for core functionality
```

## Quick Start

1. **Clone Repository**:
   ```bash
   git clone https://github.com/PatrickRay88/virality-analysis.git
   cd virality-analysis
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   python -m spacy download en_core_web_sm
   ```

3. **Configure API Access**:
   ```bash
   cp .env.example .env
   # Edit .env with your Reddit API credentials
   # Get credentials at: https://www.reddit.com/prefs/apps
   ```

4. **Collect Data**:
   ```bash
   python bin/collect_reddit.py --subreddit technology --days 30
   python bin/collect_hn.py --days 30
   ```

5. **Feature Engineering**:
   ```bash
   python bin/make_features.py
   ```

6. **Explore Results**:
   ```bash
   jupyter notebook notebooks/01_data_exploration.ipynb
   ```

## Research Pipeline

### Stage A - Quality Modeling (Exposure-aware baseline)
Estimate intrinsic content quality using Poisson regression on early score dynamics with time controls.

### Stage B - Residual Analysis (Core extension)  
Model title-driven lift above/below quality baseline using linear and gradient-boosted approaches.

### Stage C - Natural Experiments (Optional)
Analyze same-URL, different-title pairs for quasi-causal evidence.

### Stage D - Cross-platform Transfer
Test Reddit→HN and HN→Reddit model transferability.

## Data Collection Notes

- **Reddit**: Uses PRAW API with rate limiting and early snapshot collection
- **Hacker News**: Uses free JSON API for "newstories" stream
- **Ethics**: Public data only, anonymized authors, TOS compliant
- **Storage**: All data stored as Parquet files for efficient analysis

## Key Features

- **Title Analysis**: Length, sentiment (VADER), readability, clickbait markers, entity counts
- **Temporal Features**: Hour-of-day, weekday/weekend, recency buckets  
- **Context Features**: Content type flags, author account age, posting history
- **Quality Metrics**: Exposure-corrected scores, early growth trajectories

## API Setup

### Reddit API Configuration
1. Visit [Reddit App Preferences](https://www.reddit.com/prefs/apps)
2. Click "Create App" or "Create Another App"
3. Choose "script" as the app type
4. Note your `client_id` (under app name) and `client_secret`
5. Add these credentials to your `.env` file

### Data Collection Ethics
- All data collected is from public APIs and publicly available posts
- Author usernames are hashed for privacy protection
- Rate limiting implemented to respect platform terms of service
- Research conducted under academic fair use guidelines

## Contributing

This is an academic research project. For questions or collaboration inquiries, please open an issue.

## License

This project is for academic research purposes. Data collection follows platform Terms of Service and academic research guidelines.