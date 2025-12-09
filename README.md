> **Research Question**: Do certain title features push Reddit posts above or below their "intrinsic quality" baseline on Reddit? Cross-platform generalization is deferred to future work pending Hacker News data collection.
## Project Structure

```
├── src/                    # Core Python package
│   ├── ingest/            # API clients for Reddit (plus Hacker News scaffolding)
│   ├── preprocess/        # Feature engineering and data normalization
│   ├── models/            # Quality modeling and residual-lift analysis
│   └── utils/             # Shared utilities (time bins, calibration, evaluation)
├── notebooks/             # Jupyter notebooks for exploration and visualization
├── src/cli/               # CLI entry points for data collection and modeling
├── data/                  # Local raw/processed data (omitted from repo; available on request)
└── scripts/               # Automation helpers (PowerShell wrappers, etc.)
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
   # Single subreddit
   python -m src.cli.collect_reddit --subreddit technology --days 30

   # Multiple subreddits in one run (repeat -s flag)
   python -m src.cli.collect_reddit -s technology -s science -s worldnews --days 30

   # Optional future work (not executed in this repo snapshot)
   # python -m src.cli.collect_hn --days 30

   # Continuous /new polling + 5/15/30/60 minute snapshots
   python -m src.cli.run_snapshot_collector -s technology -s science -s worldnews \
       --new-limit 75 --loop-seconds 300
   ```

5. **Feature Engineering**:
   ```bash
   python -m src.cli.make_features
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

## Data Collection Notes

- **Reddit**: Uses PRAW API with rate limiting and early snapshot collection. `python -m src.cli.run_snapshot_collector` polls `/new` feeds on a schedule, persists normalized slices, and records 5/15/30/60-minute score snapshots for exposure modeling.
- **Hacker News**: Collector script targets the JSON API for "newstories" stream, but no HN snapshots are included yet
- **Ethics**: Public data only, anonymized authors, TOS compliant
- **Storage**: All data stored as Parquet files for efficient analysis
- **Availability**: Datasets are excluded from version control; request access or regenerate via the collectors

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

## References

- Weissburg, E., Kumar, A., & Dhillon, P. S. (2022). "Judging a Book by Its Cover: Predicting the Marginal Impact of Title on Reddit Post Popularity." *Proceedings of the Sixteenth International AAAI Conference on Web and Social Media (ICWSM 2022)*.
- Stoddard, G. (2015). "Popularity dynamics and intrinsic quality in Reddit and Hacker News." *Proceedings of the International AAAI Conference on Web and Social Media*, 9(1), 416-425.
- Berger, J., & Milkman, K. L. (2012). "What makes online content viral?" *Journal of Marketing Research*, 49(2), 192-205.
- Cheng, J., Adamic, L., Dow, P. A., Kleinberg, J. M., & Leskovec, J. (2014). "Can cascades be predicted?" *Proceedings of the 23rd International Conference on World Wide Web*, 925-936.
- Goel, S., Anderson, A., Hofman, J., & Watts, D. J. (2016). "The structural virality of online diffusion." *Management Science*, 62(1), 180-196.
- Guerini, M., Strapparava, C., & Özbal, G. (2011). "Exploring text virality in social networks." *Proceedings of the International AAAI Conference on Web and Social Media*, 5(1), 82-89.
- Lakkaraju, H., McAuley, J., & Leskovec, J. (2013). "What's in a name? Understanding the interplay between titles, content, and communities in social media." *Proceedings of the International AAAI Conference on Web and Social Media*, 7(1), 311-320.
