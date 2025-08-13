"""
Main application for AI-powered X profile search.
"""
import asyncio
import logging
import io
from typing import Dict, List, Any, Optional
import time
import platform

import streamlit as st
import pandas as pd

from api.twitter_client import TwitterClient
from api.ai_client import AIClient
from data.network import NetworkData
from data.processing import DataProcessor
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

# --- CSV Generation Helper Function ---
def generate_second_degree_csv(profiles: Dict[str, Any]) -> str:
    """Generates a CSV string from a dictionary of profile data."""
    if not profiles:
        return "Username,Connection Path,Reasoning\n"
    
    # Extract relevant data and connection paths
    rows = []
    for profile_id, profile_data in profiles.items():
        # Get the connection path from the profile data
        path = profile_data.get('connection_path', 'N/A')
        # We'll just put some placeholder data for reasoning here, as it's not available yet
        reasoning = "N/A"
        rows.append({
            'Username': profile_data.get('username', 'N/A'),
            'Connection Path': path,
            'Reasoning': reasoning
        })
    
    # Create a DataFrame and convert it to CSV
    df = pd.DataFrame(rows)
    return df.to_csv(index=False)

# Convert async function to sync for Streamlit
def run_async(coro):
    """Helper to run async functions in Streamlit."""
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop.run_until_complete(coro)

# --- Main Streamlit App (No async main) ---
st.set_page_config(
    page_title="AI-Powered X Profile Scout",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("AI-Powered X Profile Scout")
st.write("Find promising candidates for a specific role within a user's network.")

if not RAPIDAPI_KEY or not GEMINI_API_KEY:
    st.error("API keys not found. Please set `RAPIDAPI_KEY` and `GEMINI_API_KEY` in environment variables.")
    st.stop()

# Initialize clients and state manager
twitter_client = TwitterClient(RAPIDAPI_KEY, RAPIDAPI_HOST)
ai_client = AIClient(GEMINI_API_KEY)
data_processor = DataProcessor(twitter_client, ai_client)
state_manager = StateManager()

# Use Streamlit's session_state for storing fetched data across reruns
if "first_degree_profiles" not in st.session_state:
    st.session_state.first_degree_profiles = {}
if "second_degree_profiles" not in st.session_state:
    st.session_state.second_degree_profiles = {}
if "ai_results" not in st.session_state:
    st.session_state.ai_results = None

with st.sidebar:
    st.header("1. Fetch Data")
    input_username = st.text_input(
        "Enter an X username:",
        value=state_manager.get("input_username", "")
    )
    
    # Increased the range of the sliders as requested
    following_pages = st.slider(
        "Pages of following to fetch (original user)",
        min_value=1, max_value=10,
        value=state_manager.get("following_pages", 2)
    )
    second_degree_pages = st.slider(
        "Pages of following to fetch (for 1st-degree accounts)",
        min_value=1, max_value=5,
        value=state_manager.get("second_degree_pages", 1)
    )

    fetch_button = st.button("Fetch Network Profiles", key="fetch_profiles")

# --- Step 1: Data Fetching Logic ---
if fetch_button and input_username:
    # Clear previous results when fetching new data
    st.session_state.first_degree_profiles = {}
    st.session_state.second_degree_profiles = {}
    st.session_state.ai_results = None
    state_manager.set("input_username", input_username)
    state_manager.set("following_pages", following_pages)
    state_manager.set("second_degree_pages", second_degree_pages)

    st.subheader(f"Fetching profiles for @{input_username}")
    with st.spinner("Collecting network data... This may take several minutes."):
        try:
            start_time = time.time()
            # Use run_async helper to run the async function
            network_data_object = run_async(data_processor.collect_network_data(
                username=input_username,
                following_pages=following_pages,
                second_degree_pages=second_degree_pages
            ))
            end_time = time.time()

            # Annotate each node with its degree
            first_degree_ids = network_data_object.get_first_degree_nodes()
            second_degree_ids = network_data_object.get_second_degree_nodes()

            for node_id, node in network_data_object.nodes.items():
                if node_id == network_data_object.original_id:
                    node['degree'] = 0
                elif node_id in first_degree_ids:
                    node['degree'] = 1
                elif node_id in second_degree_ids:
                    node['degree'] = 2
                else:
                    node['degree'] = 3

            # Separate the profiles based on their degree
            first_degree_profiles = {
                node['id']: node for node in network_data_object.nodes.values() if node.get('degree') == 1
            }
            second_degree_profiles = {
                node['id']: node for node in network_data_object.nodes.values() if node.get('degree') == 2
            }
            
            st.session_state.first_degree_profiles = first_degree_profiles
            st.session_state.second_degree_profiles = second_degree_profiles
            
            total_profiles = len(first_degree_profiles) + len(second_degree_profiles)
            st.success(f"Network collection complete! Discovered a total of {total_profiles} profiles in {end_time - start_time:.2f} seconds.")
            st.write(f"This includes {len(first_degree_profiles)} first-degree profiles and {len(second_degree_profiles)} second-degree profiles.")
            
        except Exception as e:
            st.error(f"Error during data collection: {e}")
            logger.error(f"Error during data collection: {e}")

# --- CSV Download Button ---
if st.session_state.second_degree_profiles:
    st.markdown("---")
    st.subheader("Download Full Second-Degree Network")
    st.write("Download a CSV of all discovered second-degree profiles, including their connection paths, before AI analysis.")
    
    # We use a lambda to pass the profiles to the function for the download button
    csv_data = generate_second_degree_csv(st.session_state.second_degree_profiles)
    
    st.download_button(
        label="Download Second-Degree Connections CSV",
        data=csv_data,
        file_name=f"{input_username}_second_degree_network.csv",
        mime="text/csv",
    )
    st.markdown("---")

# --- Step 2: AI Analysis UI and Logic ---
# This section is only displayed after data has been fetched
if st.session_state.first_degree_profiles or st.session_state.second_degree_profiles:
    st.header("2. Find Candidates")
    st.write("Use AI to search through the **second-degree** profiles for promising candidates.")
    
    with st.form("ai_search_form"):
        user_prompt = st.text_area(
            "Describe the profile you're looking for:",
            value="I need a cofounder for a geospatial company. They should have experience in startups, be proficient in Python, and have a background in GIS or remote sensing."
        )
        find_candidates_button = st.form_submit_button("Find Candidates")

    if find_candidates_button and user_prompt:
        st.session_state.ai_results = None # Clear previous AI results
        with st.spinner("Searching for candidates with AI..."):
            try:
                # Use run_async helper for the AI search
                ai_results = run_async(ai_client.search_for_candidates(
                    profiles=list(st.session_state.second_degree_profiles.values()),
                    user_prompt=user_prompt
                ))
                st.session_state.ai_results = ai_results
            except Exception as e:
                st.error(f"Error during AI analysis: {e}")
                logger.error(f"Error during AI analysis: {e}")
        
# --- Step 3: Display AI Results ---
if st.session_state.ai_results:
    st.header("3. Promising Candidates from the Second-Degree Network")
    
    candidates_data = st.session_state.ai_results.get("candidates")
    if candidates_data:
        # Create a DataFrame for better display
        df = pd.DataFrame(candidates_data)
        
        # Build username to connection_path map from second_degree_profiles
        username_to_path = {
            profile.get('username', ''): profile.get('connection_path', 'N/A')
            for profile in st.session_state.second_degree_profiles.values()
        }

        # Map usernames to their connection paths using the mapping
        df['Connection Path'] = df['username'].map(username_to_path).fillna('N/A')
        
        # Add a clickable link for the profile
        df['Profile URL'] = df['username'].apply(
            lambda u: f"https://x.com/{u}"
        )
        
        # Select and rename columns for display
        display_df = df[['username', 'reasoning', 'Connection Path', 'Profile URL']]
        display_df.rename(columns={
            'username': 'Username',
            'reasoning': 'AI Reasoning',
            'Profile URL': 'Profile Link'
        }, inplace=True)
        
        # Display the DataFrame as an interactive table
        st.dataframe(
            display_df,
            use_container_width=True,
            column_config={
                "Profile Link": st.column_config.LinkColumn(
                    "Profile Link",
                    help="Link to the X profile",
                    display_text="View Profile"
                )
            }
        )
    else:
        st.warning("The AI did not find any promising candidates based on your prompt.")

# Keeping debug options for development
with st.sidebar:
    st.markdown("---\n_Debug Logs_")
    if st.checkbox("Show Debug Logs"):
        st.header("Debug Logs")
        st.text_area("Log Output", log_stream.getvalue(), height=300)
        if st.button("Clear Logs"):
            log_stream.truncate(0)
            log_stream.seek(0)
            st.rerun()

# REMOVED: if __name__ == "__main__": asyncio.run(main())
# This was causing the script to run instead of letting Streamlit handle it