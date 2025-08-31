# XFounderFinder V2

AI-powered X/Twitter network analysis tool for systematic founder and candidate identification.

## Overview

XFounderFinder V2 analyzes Twitter networks to identify and rank potential founders, candidates, or collaborators based on your specific requirements. It combines social network analysis with AI to provide systematic candidate evaluation with detailed evidence and scoring.

## Features

- **Network Collection**: Fetches 1st and 2nd degree Twitter connections
- **AI Candidate Analysis**: Uses Google Gemini to analyze profiles against your criteria
- **Systematic Scoring**: 4-category scoring system with detailed evidence
- **Community Detection**: Auto-groups accounts into professional communities
- **Tweet Analysis**: Processes recent tweets for professional insights
- **Influence Measurement**: CloutRank algorithm for network influence scoring

## How It Works

1. **Data Collection**: Enter target username, app fetches their network connections
2. **Community Detection**: AI groups accounts into professional categories
3. **AI Analysis**: You provide search criteria, AI scores each person on relevance
4. **Ranking**: Systematic ranking with evidence and outreach recommendations

## Scoring System

- **Role Fit (40 points)**: Match between bio/experience and your requirements
- **Influence Network (25 points)**: CloutRank influence within this specific network
- **Technical Evidence (25 points)**: Skills demonstrated in recent tweets
- **Accessibility (10 points)**: Likelihood of responding to outreach

Results are ranked A-D tiers with specific evidence for each recommendation.

## Prerequisites

- Python 3.8+
- RapidAPI account (for Twitter data)
- Google Gemini API key

## Installation

1. Clone repository:
```bash
git clone https://github.com/1saadahmed1/XFounderFinder-V2.git
cd XFounderFinder-V2
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Configure API keys:
Create `config.py`:
```python
RAPIDAPI_KEY = "your_rapidapi_key"
GEMINI_API_KEY = "your_gemini_key"
RAPIDAPI_HOST = "twitter-api45.p.rapidapi.com"
```

4. Run application:
```bash
streamlit run app.py
```

## Usage

1. **Network Collection**: Enter Twitter username and set collection limits
2. **Community Detection**: Run AI community detection to group accounts
3. **Candidate Search**: Enter search criteria (e.g., "AI startup founders with ML background")
4. **Review Results**: Get ranked candidates with detailed analysis and evidence
5. **Export**: Download CSV with complete analysis and outreach suggestions

## Configuration

Key settings in `config.py`:

- `MAX_CONCURRENT_REQUESTS`: API request concurrency (default: 10)
- Collection limits: Pages to fetch for 1st/2nd degree connections
- Community detection: Target communities and minimum sizes
- CloutRank parameters: Damping factor and convergence settings

## Project Structure

```
XFounderFinder-V2/
├── api/                 # API clients (Twitter, Gemini)
├── data/               # Network data structures and analysis
├── utils/              # Helper functions and state management
├── visualization/      # Table displays and UI components
├── app.py             # Main Streamlit application
├── config.py          # API keys and settings
└── requirements.txt   # Dependencies
```

## API Keys Setup

**RapidAPI (Twitter Data)**:
1. Sign up at rapidapi.com
2. Subscribe to Twitter API service
3. Copy your API key

**Google Gemini**:
1. Visit Google AI Studio
2. Generate API key
3. Enable Gemini API access

## Performance Notes

- Network collection limited to prevent memory issues
- Tweet processing prioritizes most influential accounts
- AI analysis uses batch processing for efficiency
- Concurrent requests managed to respect rate limits

## License

See LICENSE file for details.

## Support

For issues or questions, please use GitHub Issues.
