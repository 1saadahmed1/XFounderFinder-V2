# X Network Visualization

An interactive 3D visualization tool for exploring X (formerly Twitter) following networks. This application allows users to analyze the network structure, identify influential accounts, detect communities, and generate summaries of tweet content.

## Features

- **3D Force-Directed Graph Visualization** with permanent node labels, directional arrows, and interactive controls
- **Topic Nodes** that connect to accounts discussing similar topics
- **Network Analysis** using CloutRank (PageRank) or In-Degree metrics
- **Community Detection** powered by AI to group accounts by similar topics and interests
- **Tweet Summarization** to understand account content without manual browsing
- **Comprehensive Filtering** by followers, following, tweet count, and community
- **Data Export** for further analysis in other tools
- **High Performance** with parallel processing for API requests and data processing

## Setup

### Prerequisites

- Python 3.8 or higher
- RapidAPI key for Twitter API access
- Google Gemini API key for AI features

### Installation

1. Clone this repository:
   ```
   git clone https://github.com/BrianMills2718/x-network-viz.git
   cd x-network-viz
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Create a `.streamlit/secrets.toml` file with your API keys:
   ```toml
   RAPIDAPI_KEY = "your_rapidapi_key_here"
   GEMINI_API_KEY = "your_gemini_api_key_here"
   ```

### Running the Application

Start the Streamlit app:
```
streamlit run app.py
```

## Usage Guide

1. **Enter an X username** (without the @ symbol) in the text input field
2. Adjust **Data Fetch Options** in the sidebar:
   - Pages of Following for Original Account (each page = 20 accounts)
   - Pages of Following for Second Degree (connections of connections)
3. Click **Generate Network** to retrieve and analyze the network data
4. Use the **Display Options** to configure visualization:
   - Importance Metric: Choose between In-Degree and CloutRank
   - Account/Label Size: Adjust visual appearance
   - Max Accounts: Control how many nodes are displayed
   - Node Spacing: Control the distance between nodes
5. Apply **Filters** in the sidebar to focus on accounts of interest:
   - Numeric ranges for tweets, followers, following
   - Degree filtering (show/hide original, first-degree, second-degree)
   - Community filtering (when communities are generated)
6. Click **Summarize Tweets & Generate Communities** to:
   - Fetch and analyze tweets from top accounts
   - Generate AI-powered community classifications
   - Extract and visualize common topics
7. Explore the **Network Analysis** tables showing:
   - Top accounts by importance
   - Top accounts within each community
   - Topics discussed across accounts
8. **Export Data** using the download button in the sidebar

## Visualization Features

- **Directional Arrows**: Clearly see relationship directions between accounts with arrow heads and curved edges
- **Topic Nodes**: Diamond-shaped nodes represent topics with connections to related accounts
- **Interactive Controls**: Click nodes to focus the camera, drag to rotate, scroll to zoom
- **Customizable Display**: Adjust node sizes, colors, and filtering to create the perfect visualization

## Project Structure

```
network_viz_app/
├── app.py                  # Main Streamlit application
├── config.py               # Configuration and constants
├── api/                    # API client modules
│   ├── twitter_client.py   # X/Twitter API client
│   └── ai_client.py        # Google Gemini API client
├── data/                   # Data handling modules
│   ├── network.py          # Network data structures
│   ├── analysis.py         # Network analysis algorithms
│   ├── processing.py       # Data processing utilities
│   └── communities.py      # Community detection
├── visualization/          # Visualization components
│   ├── network_3d.py       # 3D network visualization
│   └── tables.py           # Table generation
└── utils/                  # Utility functions
    ├── helpers.py          # State management & helpers
    └── logging_config.py   # Logging configuration
```

## Performance Optimizations

The application includes several performance enhancements:
- Parallel fetching of X/Twitter API data
- Concurrent processing of tweets and community detection
- Optimized TCP connection handling
- Automatic retry mechanisms for API failures

## License

This project is available under a dual license:
- **Non-Commercial Use**: Free to use, modify, and distribute for non-commercial purposes
- **Commercial Use**: Requires a license agreement with revenue sharing (8% of gross revenue)

See the [LICENSE.md](LICENSE.md) file for full details.

## Acknowledgements

- Force Graph 3D for the visualization engine
- Streamlit for the web application framework
- NetworkX for graph algorithms
- Google Gemini API for AI-powered analysis