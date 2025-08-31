"""
Enhanced XFounderFinder application - FINAL FIXED VERSION.
"""
import asyncio
import logging
import io
from typing import Dict, List, Any, Optional, Set
import time

import streamlit as st
import pandas as pd
import networkx as nx

from api.twitter_client import TwitterClient
from api.ai_client import AIClient
from data.network import NetworkData
from data.processing import DataProcessor
from data.communities import CommunityManager
from data.analysis import compute_cloutrank, compute_in_degree
from utils.helpers import StateManager
from config import (
    RAPIDAPI_KEY, RAPIDAPI_HOST, GEMINI_API_KEY
)

# --- Logging Setup ---
log_stream = io.StringIO()
stream_handler = logging.StreamHandler(log_stream)
stream_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
stream_handler.setFormatter(formatter)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logging.getLogger().addHandler(stream_handler)

# --- Helper Functions ---
def run_async(coro):
    """Helper to run async functions in Streamlit."""
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop.run_until_complete(coro)

def calculate_mutual_connections(network: NetworkData, target_id: str) -> int:
    """Calculate how many first-degree connections also follow this target."""
    if not network.original_id:
        return 0
    
    mutual_count = 0
    first_degree = network.get_first_degree_nodes()
    
    for edge in network.edges:
        if edge[1] == target_id and edge[0] in first_degree:
            mutual_count += 1
    
    return mutual_count

def display_structured_candidate_results(ai_response: Dict[str, Any]) -> None:
    """Display structured candidate results with color coding and ranking."""
    if not ai_response or not ai_response.get("candidates"):
        st.info("No candidates found matching your criteria.")
        return
    
    candidates = ai_response["candidates"]
    analysis_summary = ai_response.get("analysis_summary", "")
    
    # Apply custom styling
    st.markdown("""
    <style>
        .tier-a { background: linear-gradient(135deg, #4CAF50, #45a049) !important; color: white !important; }
        .tier-b { background: linear-gradient(135deg, #2196F3, #1976D2) !important; color: white !important; }
        .tier-c { background: linear-gradient(135deg, #FF9800, #F57C00) !important; color: white !important; }
        .tier-d { background: linear-gradient(135deg, #f44336, #d32f2f) !important; color: white !important; }
        .candidate-header { font-size: 18px; font-weight: bold; margin-bottom: 15px; padding: 10px; border-radius: 8px; text-align: center; }
        .score-breakdown { display: flex; justify-content: space-between; margin: 10px 0; padding: 10px; background-color: rgba(255, 255, 255, 0.1); border-radius: 5px; }
        .evidence-section { margin: 10px 0; padding: 10px; background-color: rgba(0, 0, 0, 0.05); border-radius: 5px; border-left: 4px solid #2196F3; }
        .strength-item { margin: 5px 0; padding: 5px 10px; background-color: rgba(76, 175, 80, 0.1); border-radius: 3px; border-left: 3px solid #4CAF50; }
        .concern-item { margin: 5px 0; padding: 5px 10px; background-color: rgba(255, 152, 0, 0.1); border-radius: 3px; border-left: 3px solid #FF9800; }
    </style>
    """, unsafe_allow_html=True)
    
    st.subheader("AI Candidate Analysis Results")
    
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        if analysis_summary:
            st.info(f"**Analysis Summary:** {analysis_summary}")
    with col2:
        st.metric("Candidates Found", len(candidates))
    with col3:
        if candidates:
            avg_score = sum(c.get("total_score", 0) for c in candidates) / len(candidates)
            st.metric("Average Score", f"{avg_score:.1f}/100")
    
    # Display tier distribution
    tier_counts = {"A": 0, "B": 0, "C": 0, "D": 0}
    for candidate in candidates:
        tier = candidate.get("tier", "D")
        tier_counts[tier] += 1
    
    tier_cols = st.columns(4)
    tier_colors = {"A": "#4CAF50", "B": "#2196F3", "C": "#FF9800", "D": "#f44336"}
    for i, (tier, count) in enumerate(tier_counts.items()):
        with tier_cols[i]:
            if count > 0:
                st.markdown(f'''
                <div style="background-color: {tier_colors[tier]}; color: white; padding: 10px; border-radius: 5px; text-align: center;">
                    <strong>Tier {tier}</strong><br>{count} candidates
                </div>
                ''', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Display each candidate
    for candidate in candidates:
        username = candidate.get("username", "Unknown")
        total_score = candidate.get("total_score", 0)
        rank = candidate.get("rank", 0)
        tier = candidate.get("tier", "D")
        reasoning = candidate.get("reasoning", "")
        scores = candidate.get("scores", {})
        
        tier_class = f"tier-{tier.lower()}"
        
        st.markdown(f"""
        <div class="candidate-header {tier_class}">
            #{rank} @{username} - Tier {tier} - Score: {total_score}/100
        </div>
        """, unsafe_allow_html=True)
        
        # Score breakdown if available
        if scores:
            st.markdown(f"""
            <div class="score-breakdown">
                <div style="text-align: center; flex: 1;">
                    <div style="font-size: 12px; color: #666;">Role Fit</div>
                    <div style="font-size: 16px; font-weight: bold;">{scores.get("role_fit", 0)}/40</div>
                </div>
                <div style="text-align: center; flex: 1;">
                    <div style="font-size: 12px; color: #666;">Influence</div>
                    <div style="font-size: 16px; font-weight: bold;">{scores.get("influence_network", 0)}/25</div>
                </div>
                <div style="text-align: center; flex: 1;">
                    <div style="font-size: 12px; color: #666;">Technical</div>
                    <div style="font-size: 16px; font-weight: bold;">{scores.get("technical_evidence", 0)}/25</div>
                </div>
                <div style="text-align: center; flex: 1;">
                    <div style="font-size: 12px; color: #666;">Access</div>
                    <div style="font-size: 16px; font-weight: bold;">{scores.get("accessibility", 0)}/10</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # Display reasoning
        st.markdown(f"""
        <div class="evidence-section">
            <strong>Analysis:</strong><br>
            {reasoning}
        </div>
        """, unsafe_allow_html=True)
        
        # Display strengths
        if candidate.get("key_strengths"):
            st.write("**Key Strengths:**")
            for strength in candidate["key_strengths"]:
                st.markdown(f'<div class="strength-item">• {strength}</div>', unsafe_allow_html=True)
        
        # Display concerns
        if candidate.get("concerns"):
            st.write("**Concerns:**")
            for concern in candidate["concerns"]:
                st.markdown(f'<div class="concern-item">• {concern}</div>', unsafe_allow_html=True)
        
        if candidate.get("outreach_approach"):
            st.write(f"**Outreach Strategy:** {candidate['outreach_approach']}")
        
        st.markdown("---")

def generate_enhanced_csv(profiles: Dict[str, Any], 
                          importance_scores: Dict[str, float],
                          mutual_connections: Dict[str, int],
                          communities: Dict[str, str] = None) -> str:
    """Generate enhanced CSV with all metrics."""
    if not profiles:
        return "Username,Name,Connection Path,CloutRank,Mutual Connections,Followers,Bio,Community\n"
    
    rows = []
    for profile_id, profile_data in profiles.items():
        community = ""
        if communities:
            username = profile_data.get('screen_name', '')
            community = communities.get(username, 'Uncategorized')
            
        rows.append({
            'Username': profile_data.get('screen_name', 'N/A'),
            'Name': profile_data.get('name', 'N/A'),
            'Connection Path': profile_data.get('connection_path', 'N/A'),
            'CloutRank Score': importance_scores.get(profile_id, 0),
            'Mutual Connections': mutual_connections.get(profile_id, 0),
            'Followers': profile_data.get('followers_count', 0),
            'Bio': profile_data.get('description', ''),
            'Community': community,
            'Tweet Summary': profile_data.get('tweet_summary', '')
        })
    
    rows = sorted(rows, key=lambda x: x['CloutRank Score'], reverse=True)
    df = pd.DataFrame(rows)
    return df.to_csv(index=False)

# --- Main Streamlit App ---
st.set_page_config(
    page_title="AI-Powered X Profile Scout - Enhanced",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🚀 XFounderFinder - Enhanced Edition")
st.write("Advanced AI-powered candidate discovery with network analysis, tweet insights, and community detection.")

if not RAPIDAPI_KEY or not GEMINI_API_KEY:
    st.error("API keys not found. Please set RAPIDAPI_KEY and GEMINI_API_KEY in environment variables.")
    st.stop()

# Initialize clients and managers
twitter_client = TwitterClient(RAPIDAPI_KEY, RAPIDAPI_HOST)
ai_client = AIClient(GEMINI_API_KEY)
data_processor = DataProcessor(twitter_client, ai_client)
community_manager = CommunityManager(ai_client)
state_manager = StateManager()

# Initialize session state
if "network_data" not in st.session_state:
    st.session_state.network_data = None
if "first_degree_profiles" not in st.session_state:
    st.session_state.first_degree_profiles = {}
if "second_degree_profiles" not in st.session_state:
    st.session_state.second_degree_profiles = {}
if "ai_results" not in st.session_state:
    st.session_state.ai_results = None
if "importance_scores" not in st.session_state:
    st.session_state.importance_scores = {}
if "communities" not in st.session_state:
    st.session_state.communities = {}
if "community_labels" not in st.session_state:
    st.session_state.community_labels = {}
if "tweet_data_fetched" not in st.session_state:
    st.session_state.tweet_data_fetched = False

# --- Sidebar Configuration ---
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Data Fetching Section
    st.subheader("1. Network Discovery")
    input_username = st.text_input(
        "Enter X username:",
        value=state_manager.get("input_username", ""),
        help="The starting point for network exploration"
    )
    
    col1, col2 = st.columns(2)
    with col1:
        following_pages = st.slider(
            "1st-degree pages",
            min_value=1, max_value=10,
            value=state_manager.get("following_pages", 2),
            help="Each page = ~20 accounts"
        )
    with col2:
        second_degree_pages = st.slider(
            "2nd-degree pages",
            min_value=1, max_value=5,
            value=state_manager.get("second_degree_pages", 1),
            help="Per 1st-degree account"
        )
    
    # Feature Toggles
    st.subheader("2. Analysis Features")
    
    include_first_degree = st.toggle(
        "Include 1st-degree in search",
        value=False,
        help="Also analyze direct connections"
    )
    
    fetch_tweets = st.toggle(
        "Analyze tweet content",
        value=True,
        help="Fetch and analyze recent tweets (slower but more accurate)"
    )
    
    enable_communities = st.toggle(
        "Community detection",
        value=True,
        help="Group accounts by interests/topics"
    )
    
    if enable_communities:
        num_communities = st.slider(
            "Number of communities",
            min_value=3, max_value=12,
            value=6,
            help="More communities = specific groups, fewer = broader categories"
        )
        
        min_community_size = st.slider(
            "Minimum community size",
            min_value=3, max_value=20,
            value=8,
            help="Communities smaller than this will be filtered out"
        )
    
    calculate_influence = st.toggle(
        "Calculate influence scores",
        value=True,
        help="PageRank-based importance metrics"
    )
    
    fetch_button = st.button("🔍 Fetch Network", key="fetch_profiles", type="primary")

# --- Main Content Area ---
col1_main, col2_main = st.columns([2, 1])

with col1_main:
    # --- Step 1: Data Fetching ---
    if fetch_button and input_username:
        # Reset session state
        for key in ["network_data", "first_degree_profiles", "second_degree_profiles", 
                   "ai_results", "importance_scores", "communities", "community_labels", "tweet_data_fetched"]:
            if key in st.session_state:
                del st.session_state[key]
        
        st.subheader(f"📡 Fetching network for @{input_username}")
        
        with st.spinner("Collecting network data..."):
            try:
                start_time = time.time()
                network_data = run_async(data_processor.collect_network_data(
                    username=input_username,
                    following_pages=following_pages,
                    second_degree_pages=second_degree_pages,
                    max_first_degree=200,
                    max_second_degree_per_account=30
                ))
                end_time = time.time()

                st.session_state.network_data = network_data
                
                # Separate profiles by degree
                first_degree_ids = network_data.get_first_degree_nodes()
                second_degree_ids = network_data.get_second_degree_nodes()
                
                first_degree_profiles = {
                    node_id: network_data.nodes[node_id] 
                    for node_id in first_degree_ids
                }
                second_degree_profiles = {
                    node_id: network_data.nodes[node_id] 
                    for node_id in second_degree_ids
                }
                
                st.session_state.first_degree_profiles = first_degree_profiles
                st.session_state.second_degree_profiles = second_degree_profiles
                
                # Calculate mutual connections
                mutual_connections = {}
                for node_id in network_data.nodes:
                    mutual_connections[node_id] = calculate_mutual_connections(network_data, node_id)
                st.session_state.mutual_connections = mutual_connections
                
                # Calculate influence scores
                if calculate_influence:
                    with st.spinner("Calculating influence scores..."):
                        cloutrank_scores = compute_cloutrank(network_data.graph)
                        in_degree_scores = compute_in_degree(network_data.graph)
                        
                        st.session_state.cloutrank_scores = cloutrank_scores
                        st.session_state.in_degree_scores = in_degree_scores
                        
                        # Combined importance scores
                        importance_scores = {}
                        for node_id in network_data.nodes:
                            clout = cloutrank_scores.get(node_id, 0)
                            mutual = mutual_connections.get(node_id, 0) / max(len(first_degree_ids), 1) if first_degree_ids else 0
                            importance_scores[node_id] = (clout * 0.7) + (mutual * 0.3)
                        
                        st.session_state.importance_scores = importance_scores
                
                st.success(f"✅ Network collected: {len(first_degree_profiles)} 1st-degree, "
                          f"{len(second_degree_profiles)} 2nd-degree profiles in {end_time - start_time:.2f}s")
                
                # Fetch tweets if enabled
                if fetch_tweets:
                    with st.spinner("Fetching and analyzing tweets..."):
                        profiles_to_analyze = sorted(
                            st.session_state.importance_scores.items(),
                            key=lambda x: x[1],
                            reverse=True
                        )[:50]
                        
                        selected_nodes = {node_id for node_id, _ in profiles_to_analyze}
                        network_data = run_async(
                            data_processor.process_tweet_data(
                                network_data,
                                selected_nodes,
                                batch_size=10
                            )
                        )
                        st.session_state.network_data = network_data
                        st.session_state.tweet_data_fetched = True
                        st.success(f"✅ Analyzed tweets for {len(selected_nodes)} top accounts")
                
                # FIXED: Community detection with proper error handling
                if enable_communities and len(network_data.nodes) > 20:
                    with st.spinner("Detecting communities..."):
                        try:
                            node_descriptions = {}
                            for node_id, node in network_data.nodes.items():
                                if node_id.startswith("orig_"):
                                    continue
                                username = node.get("screen_name", "")
                                description = node.get("description", "")
                                tweet_summary = node.get("tweet_summary", "")
                                combined = f"{description} {tweet_summary}".strip()
                                if combined and username:
                                    node_descriptions[username] = combined
                            
                            if len(node_descriptions) >= 20:
                                # FIXED: Use run_async for all async calls
                                community_labels = run_async(ai_client.generate_community_labels(
                                    [{"screen_name": k, "description": v, "tweet_summary": ""} 
                                     for k, v in node_descriptions.items()],
                                    num_communities
                                ))
                                
                                node_communities = run_async(ai_client.classify_accounts(
                                    [{"screen_name": k, "description": v} 
                                     for k, v in node_descriptions.items()],
                                    community_labels
                                ))
                                
                                # Filter communities by size
                                community_counts = {}
                                for username, comm_id in node_communities.items():
                                    if comm_id in community_labels:
                                        community_counts[comm_id] = community_counts.get(comm_id, 0) + 1
                                
                                filtered_labels = {
                                    comm_id: label for comm_id, label in community_labels.items()
                                    if community_counts.get(comm_id, 0) >= min_community_size
                                }
                                
                                filtered_communities = {
                                    username: comm_id for username, comm_id in node_communities.items()
                                    if comm_id in filtered_labels
                                }
                                
                                st.session_state.communities = filtered_communities
                                st.session_state.community_labels = filtered_labels
                                
                                if filtered_labels:
                                    st.success(f"✅ Detected {len(filtered_labels)} meaningful communities")
                                else:
                                    st.warning("No communities met minimum size requirements")
                        except Exception as community_error:
                            st.warning(f"Community detection failed: {community_error}")
                            logger.error(f"Community detection error: {community_error}")
                
            except Exception as e:
                st.error(f"❌ Error: {e}")
                logger.error(f"Error during data collection: {e}")

with col2_main:
    # --- Network Statistics ---
    if st.session_state.network_data:
        st.subheader("📊 Network Statistics")
        
        metrics_col1, metrics_col2 = st.columns(2)
        with metrics_col1:
            st.metric("1st Degree", len(st.session_state.first_degree_profiles))
            if st.session_state.importance_scores:
                top_scorer = max(st.session_state.importance_scores.items(), key=lambda x: x[1])
                if top_scorer[0] in st.session_state.network_data.nodes:
                    top_name = st.session_state.network_data.nodes[top_scorer[0]].get("screen_name", "Unknown")
                    st.metric("Top Influence", f"@{top_name}")
        
        with metrics_col2:
            st.metric("2nd Degree", len(st.session_state.second_degree_profiles))
            if st.session_state.tweet_data_fetched:
                st.metric("Tweets Analyzed", "✅")
        
        # Display communities
        if st.session_state.community_labels:
            st.subheader("🏷️ Communities Found")
            community_counts = {}
            for username, comm_id in st.session_state.communities.items():
                community_counts[comm_id] = community_counts.get(comm_id, 0) + 1
            
            for comm_id, label in st.session_state.community_labels.items():
                count = community_counts.get(comm_id, 0)
                if count > 0:
                    st.write(f"• **{label}**: {count} accounts")

# --- Download Section ---
if st.session_state.get("second_degree_profiles") or st.session_state.get("first_degree_profiles"):
    st.markdown("---")
    st.subheader("💾 Export Network Data")
    
    col1_export, col2_export = st.columns(2)
    
    with col1_export:
        export_profiles = dict(st.session_state.second_degree_profiles)
        if include_first_degree:
            export_profiles.update(st.session_state.first_degree_profiles)
        
        csv_data = generate_enhanced_csv(
            export_profiles,
            st.session_state.get("importance_scores", {}),
            st.session_state.get("mutual_connections", {}),
            st.session_state.get("communities", {})
        )
        
        st.download_button(
            label="📥 Download Enhanced Network CSV",
            data=csv_data,
            file_name=f"{input_username}_network_analysis.csv",
            mime="text/csv",
        )
    
    with col2_export:
        st.info(f"📊 Export contains {len(export_profiles)} profiles")

# --- AI Analysis Section ---
if st.session_state.get("first_degree_profiles") or st.session_state.get("second_degree_profiles"):
    st.markdown("---")
    st.header("🤖 AI-Powered Candidate Search")
    
    # FIXED: Community filters OUTSIDE the form
    selected_communities_filter = {}
    if st.session_state.get("community_labels"):
        st.subheader("Filter by Communities")
        
        # Count members per community
        community_counts = {}
        for username, comm_id in st.session_state.communities.items():
            if comm_id in st.session_state.community_labels:
                community_counts[comm_id] = community_counts.get(comm_id, 0) + 1
        
        # Only show communities with members
        communities_with_members = [
            (comm_id, label, community_counts.get(comm_id, 0)) 
            for comm_id, label in st.session_state.community_labels.items()
            if community_counts.get(comm_id, 0) > 0
        ]
        
        if communities_with_members:
            communities_with_members.sort(key=lambda x: x[2], reverse=True)
            
            # Limit display
            max_display = 10
            if len(communities_with_members) > max_display:
                st.warning(f"Showing top {max_display} communities for performance")
                communities_with_members = communities_with_members[:max_display]
            
            # Display checkboxes
            for comm_id, label, count in communities_with_members:
                default_value = count >= 10
                selected = st.checkbox(
                    f"{label} ({count})",
                    value=st.session_state.get(f"comm_filter_{comm_id}", default_value),
                    key=f"comm_filter_{comm_id}"
                )
                selected_communities_filter[comm_id] = selected
    
    # FIXED: Simple form without buttons
    with st.form("ai_search_form"):
        user_prompt = st.text_area(
            "Describe your ideal candidate:",
            value="I need a technical co-founder with experience in AI/ML, strong Python skills, "
                  "startup experience, and ideally knowledge of B2B SaaS.",
            height=100
        )
        
        find_button = st.form_submit_button("🔍 Find Candidates", type="primary")

    if find_button and user_prompt:
        with st.spinner("Searching with AI..."):
            try:
                ai_results = run_async(
                    data_processor.find_candidates_with_ai(
                        st.session_state.network_data,
                        user_prompt
                    )
                )
                st.session_state.ai_results = ai_results
                
            except Exception as e:
                st.error(f"Error during AI analysis: {e}")
                logger.error(f"Error during AI analysis: {e}")

# --- Results Display ---
if st.session_state.get("ai_results"):
    st.markdown("---")
    display_structured_candidate_results(st.session_state.ai_results)

# Debug section
with st.sidebar:
    st.markdown("---")
    if st.checkbox("🔧 Show Debug Info"):
        st.text_area("Log Output", log_stream.getvalue(), height=200)
        if st.button("Clear Logs"):
            log_stream.truncate(0)
            log_stream.seek(0)
            st.rerun()