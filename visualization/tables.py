"""
Table visualization utilities for X Network Visualization.
"""
import logging
from typing import Dict, List, Tuple, Optional, Any

import streamlit as st
import pandas as pd

# Assuming this is available and correctly implemented
from data.network import NetworkData

logger = logging.getLogger(__name__)

class TableVisualizer:
    """
    Visualizer for data tables.
    """
    
    @staticmethod
    def format_text_with_line_breaks(text: str, max_line_length: int = 80) -> str:
        """
        Format text with line breaks at word boundaries for better readability in tables.
        
        Args:
            text: Text to format
            max_line_length: Maximum length for each line
            
        Returns:
            Formatted text with line breaks
        """
        if not text:
            return ""
            
        if len(text) <= max_line_length:
            return text
            
        words = text.split()
        lines = []
        current_line = ""
        
        for word in words:
            if len(current_line + " " + word) > max_line_length:
                lines.append(current_line)
                current_line = word
            else:
                if current_line:
                    current_line += " " + word
                else:
                    current_line = word
        
        if current_line:
            lines.append(current_line)
        
        return "\n".join(lines)
    
    @staticmethod
    def apply_table_styles():
        """
        Apply custom CSS styling to make tables more readable, particularly in dark mode.
        """
        table_styles = """
        <style>
            .stTable {
                width: 100% !important;
                overflow-x: auto !important;
            }
            .stTable table {
                min-width: 1000px;
                padding: 1em;
                font-size: 14px;
            }
            .stTable thead tr {
                background-color: rgba(108, 166, 205, 0.3);
            }
            .stTable thead th {
                padding: 12px !important;
                font-weight: bold !important;
                color: white !important;
                text-align: left !important;
                position: sticky !important;
                top: 0 !important;
                z-index: 1 !important;
                background-color: rgb(38, 39, 48) !important;
                border-bottom: 2px solid rgba(108, 166, 205, 0.7) !important;
            }
            .stTable tbody tr {
                border-bottom: 1px solid rgba(255, 255, 255, 0.1) !important;
            }
            .stTable tbody tr:nth-child(even) {
                background-color: rgba(255, 255, 255, 0.05) !important;
            }
            .stTable tbody tr:hover {
                background-color: rgba(108, 166, 205, 0.2) !important;
            }
            .stTable tbody td {
                padding: 10px !important;
                text-align: left !important;
                white-space: pre-line !important;
                word-wrap: break-word !important;
                max-width: 300px !important;
                line-height: 1.4 !important;
            }
            .stTable tbody td:nth-child(8) {
                white-space: pre-line !important;
                max-height: 150px !important;
                overflow-y: auto !important;
            }
            .stTable tbody td:nth-child(9) {
                white-space: pre-line !important;
                max-height: 150px !important;
                overflow-y: auto !important;
            }
            .stTable tbody td:nth-child(2) {
                font-weight: bold !important;
                white-space: nowrap !important;
            }
            .connection-original {
                color: #ff9d00 !important;
                font-weight: bold !important;
            }
            .connection-first {
                color: #00c3ff !important;
                font-weight: bold !important;
            }
            .connection-second {
                color: #8bc34a !important;
                font-weight: bold !important;
            }
        </style>
        """
        st.markdown(table_styles, unsafe_allow_html=True)
        
        js = """
        <script>
            function styleConnectionCells() {
                const tables = document.querySelectorAll('.stTable table');
                if (!tables.length) return;
                tables.forEach(table => {
                    let connectionIdx = -1;
                    const headers = table.querySelectorAll('thead th');
                    headers.forEach((header, idx) => {
                        if (header.textContent.includes('Connection')) {
                            connectionIdx = idx;
                        }
                    });
                    if (connectionIdx === -1) return;
                    const rows = table.querySelectorAll('tbody tr');
                    rows.forEach(row => {
                        const cell = row.cells[connectionIdx];
                        if (!cell) return;
                        const text = cell.textContent.trim();
                        if (text.includes('Original')) {
                            cell.classList.add('connection-original');
                        } else if (text.includes('1st Degree')) {
                            cell.classList.add('connection-first');
                        } else if (text.includes('2nd Degree')) {
                            cell.classList.add('connection-second');
                        }
                    });
                });
            }
            document.addEventListener("DOMContentLoaded", function() {
                styleConnectionCells();
                const observer = new MutationObserver(function(mutations) {
                    styleConnectionCells();
                });
                observer.observe(document.body, { childList: true, subtree: true });
            });
        </script>
        """
        st.markdown(js, unsafe_allow_html=True)
    
    @staticmethod
    def display_top_accounts_table(
        nodes: Dict[str, Dict],
        importance_scores: Dict[str, float],
        cloutrank_scores: Dict[str, float],
        in_degrees: Dict[str, int],
        top_accounts: List[Dict[str, Any]],
        original_id: str,
        show_tweet_summaries: bool = False,
        importance_metric: str = "CloutRank"
    ) -> None:
        """
        Display table of top accounts based on importance scores.
        
        This method has been corrected to handle potential None values in all
        formatted numeric fields, providing a definitive fix for the error.
        
        This version is also corrected to properly identify and label the
        original seed account.
        """
        st.subheader(f"Top Accounts by {importance_metric}")
        
        TableVisualizer.apply_table_styles()
        
        rows_list = []
        
        # This will hold the normalized username, e.g., 'elonmusk'
        normalized_original_id = original_id.replace('@', '').lower()

        for idx, account_data in enumerate(top_accounts, 1):
            node_id = account_data.get('username')
            
            if not node_id:
                logger.warning(f"Skipping account data with missing username: {account_data}")
                continue

            node_details = nodes.get(node_id, {})
            if not node_details:
                logger.warning(f"Skipping account with node_id '{node_id}' as details were not found in network_nodes.")
                continue

            # --- CRITICAL FIX: Correctly identify the connection type
            # Check if the current node_id matches the original_id.
            if node_id.lower() == normalized_original_id:
                connection = "Original"
            else:
                connection = account_data.get('connection_type', 'Other')

            # --- CRITICAL FIX: Retrieve scores and check for None before formatting.
            cr_value = cloutrank_scores.get(node_id)
            cr_value_str = f"{cr_value:.4f}" if cr_value is not None else "N/A"
            
            id_value = in_degrees.get(node_id)
            id_value_str = str(id_value) if id_value is not None else "N/A"

            followers = node_details.get('followers_count')
            followers_str = f"{followers:,}" if followers is not None else "N/A"
            
            following = node_details.get('friends_count')
            following_str = f"{following:,}" if following is not None else "N/A"

            row = {
                "Rank": idx,
                "Username": f"@{node_details.get('screen_name', 'N/A')}",
                "Connection": connection,
                "CloutRank": cr_value_str,
                "In-Degree": id_value_str,
                "Followers": followers_str,
                "Following": following_str,
                "Description": TableVisualizer.format_text_with_line_breaks(node_details.get('description', '')),
            }
            
            if 'community_name' in account_data:
                 row['Community'] = account_data['community_name']

            if show_tweet_summaries:
                tweet_summary = account_data.get("tweet_summary", "No tweet summary available")
                row["Tweet Summary"] = TableVisualizer.format_text_with_line_breaks(tweet_summary)
            
            rows_list.append(row)
        
        df = pd.DataFrame(rows_list)
        
        if not df.empty:
            with st.expander(f"Top {len(rows_list)} Accounts by {importance_metric}", expanded=False):
                st.table(df)
        else:
            st.info("No accounts to display.")

    @staticmethod
    def display_community_tables(
        network_nodes: Dict[str, Dict],
        community_accounts: Dict[str, List[Dict[str, Any]]],
        community_colors: Dict[str, str],
        community_labels: Dict[str, str],
        cloutrank_scores: Dict[str, float],
        in_degrees: Dict[str, int],
        show_tweet_summaries: bool = False
    ) -> None:
        """
        Display tables of top accounts for each community.
        
        This method has been updated to safely handle potential None values in all
        formatted numeric fields.
        """
        st.header("Community Analysis")
        
        TableVisualizer.apply_table_styles()
        
        if not community_accounts:
            st.info("No communities found to display.")
            return

        sorted_communities = sorted(
            community_accounts.items(),
            key=lambda x: len(x[1]),
            reverse=True
        )
        
        for community_id, accounts in sorted_communities:
            label = community_labels.get(community_id, f"Community {community_id}")
            color = community_colors.get(community_id, "#6ca6cd")
            
            with st.expander(f"{label} ({len(accounts)} accounts)", expanded=False):
                st.markdown(f"<div style='width:100%; height:3px; background-color:{color}'></div>", 
                            unsafe_allow_html=True)
                
                rows_list = []
                for account_data in accounts:
                    node_id = account_data.get('username')
                    
                    if not node_id:
                        logger.warning(f"Skipping community account data with missing username: {account_data}")
                        continue
                        
                    node_details = network_nodes.get(node_id, {})
                    if not node_details:
                        logger.warning(f"Skipping community account with node_id '{node_id}' as details were not found.")
                        continue
                    
                    # CRITICAL FIX: Retrieve scores and check for None before formatting.
                    cr_value = cloutrank_scores.get(node_id)
                    cr_value_str = f"{cr_value:.4f}" if cr_value is not None else "N/A"
                    
                    id_value = in_degrees.get(node_id)
                    id_value_str = str(id_value) if id_value is not None else "N/A"

                    # CRITICAL FIX: Add checks for all other formatted numeric values
                    followers = node_details.get('followers_count')
                    followers_str = f"{followers:,}" if followers is not None else "N/A"
                    
                    following = node_details.get('friends_count')
                    following_str = f"{following:,}" if following is not None else "N/A"

                    row = {
                        "Username": f"@{node_details.get('screen_name', 'N/A')}",
                        "CloutRank": cr_value_str,
                        "In-Degree": id_value_str,
                        "Followers": followers_str,
                        "Following": following_str,
                        "Description": TableVisualizer.format_text_with_line_breaks(node_details.get('description', '')),
                    }
                    if show_tweet_summaries:
                        tweet_summary = account_data.get("tweet_summary", "No tweet summary available")
                        row["Tweet Summary"] = TableVisualizer.format_text_with_line_breaks(tweet_summary)
                    
                    rows_list.append(row)
                
                if rows_list:
                    df = pd.DataFrame(rows_list)
                    st.table(df)
                else:
                    st.info("No accounts in this community to display.")

    @staticmethod
    def display_community_color_key(
        community_labels: Dict[str, str],
        community_colors: Dict[str, str],
        node_communities: Dict[str, str]
    ) -> None:
        st.subheader("Community Color Key")
        community_counts = {}
        for username, comm_id in node_communities.items():
            community_counts[comm_id] = community_counts.get(comm_id, 0) + 1
        community_data = []
        for comm_id, color in community_colors.items():
            label = community_labels.get(comm_id, f"Community {comm_id}")
            count = community_counts.get(comm_id, 0)
            label_with_count = f"{label} ({count} accounts)"
            community_data.append((label_with_count, color, comm_id))
        community_data.sort(key=lambda x: x[0])
        num_communities = len(community_data)
        num_cols = min(4, max(2, 5 - (num_communities // 15)))
        st.markdown("""
        <style>
        .community-grid { max-height: 400px; overflow-y: auto; padding-right: 10px; }
        </style>
        """, unsafe_allow_html=True)
        with st.container():
            st.markdown('<div class="community-grid">', unsafe_allow_html=True)
            for i in range(0, num_communities, num_cols):
                cols = st.columns(num_cols)
                for j in range(num_cols):
                    idx = i + j
                    if idx < num_communities:
                        label, color, _ = community_data[idx]
                        with cols[j]:
                            st.markdown(
                                f'<div style="display:flex; align-items:center">'
                                f'<div style="width:15px; height:15px; background-color:{color}; '
                                f'border-radius:3px; margin-right:8px;"></div>'
                                f'<span style="font-size:0.9em; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;">{label}</span>'
                                f'</div>',
                                unsafe_allow_html=True
                            )
            st.markdown('</div>', unsafe_allow_html=True)

    @staticmethod
    def find_promising_candidates(
        network_nodes: Dict[str, Dict],
        cloutrank_scores: Dict[str, float],
        community_accounts: Dict[str, List[Dict[str, Any]]],
        user_purpose: str,
        top_n: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Analyzes network data to find and rank promising candidates based on a user's purpose.
        
        Args:
            network_nodes: A dictionary of all account nodes.
            cloutrank_scores: A dictionary of CloutRank scores for each node.
            community_accounts: A dictionary mapping community IDs to accounts.
            user_purpose: The user's purpose as a string (e.g., "looking for a co-founder for a sustainable tech company").
            top_n: The number of top candidates to return.
            
        Returns:
            A sorted list of dictionaries representing the most promising candidates.
        """
        logger.info(f"Finding candidates for purpose: '{user_purpose}'")
        
        # --- 1. Define Relevant Keywords and Communities ---
        # Normalize purpose for keyword extraction
        purpose_lower = user_purpose.lower()
        
        # Define keywords from the purpose, could be enhanced with NLP libraries
        keywords = {
            "cofounder": 3.0, "founder": 2.5, "entrepreneur": 2.0, "ceo": 1.5,
            "electric vehicle": 2.5, "ev": 2.5, "battery": 2.0, "tech": 1.5,
            "sustainability": 2.0, "clean energy": 1.5, "technology": 1.5,
            "investor": 1.0, "business": 1.0, "venture": 1.0, "startup": 2.5,
        }
        
        # Define communities that are likely to contain relevant candidates
        # This is based on hypothetical community labels from your AI client
        relevant_communities = {
            "Sustainable Tech": 3.0,
            "AI and Robotics": 2.0,
            "Startup Founders": 3.0,
            "Venture Capital": 1.5,
            "Technology Innovators": 2.0,
        }
        
        candidate_scores = {}

        # --- 2. Score Each Account in the Network ---
        for node_id, node_data in network_nodes.items():
            score = 0.0
            
            # --- Score based on CloutRank (Influence) ---
            cloutrank = cloutrank_scores.get(node_id, 0)
            score += cloutrank * 10  # Weight CloutRank heavily

            # --- Score based on Keyword Matches in Bio and Tweets ---
            bio_text = node_data.get('description', '').lower()
            tweet_summary = node_data.get('tweet_summary', '').lower()
            
            combined_text = bio_text + " " + tweet_summary
            
            for keyword, weight in keywords.items():
                if keyword in combined_text:
                    score += weight

            # --- Score based on Community Affiliation ---
            community_id = node_data.get('community')
            if community_id:
                community_name = TableVisualizer.get_community_name(community_id, community_accounts)
                if community_name in relevant_communities:
                    score += relevant_communities[community_name]
            
            if score > 0:
                candidate_scores[node_id] = score

        # --- 3. Rank and Format the Candidates ---
        sorted_candidates = sorted(candidate_scores.items(), key=lambda item: item[1], reverse=True)
        
        final_candidates = []
        for node_id, score in sorted_candidates[:top_n]:
            node_data = network_nodes[node_id]
            
            # Find the community name
            community_id = node_data.get('community')
            community_name = TableVisualizer.get_community_name(community_id, community_accounts)
            
            final_candidates.append({
                "Rank": len(final_candidates) + 1,
                "Username": f"@{node_data.get('screen_name', 'N/A')}",
                "Score": f"{score:.2f}",
                "Community": community_name,
                "Description": TableVisualizer.format_text_with_line_breaks(node_data.get('description', '')),
            })
            
        logger.info(f"Found {len(final_candidates)} promising candidates.")
        return final_candidates

    @staticmethod
    def get_community_name(community_id, community_accounts):
        """Helper to find community name from community_accounts dict."""
        for name, accounts_list in community_accounts.items():
            for account in accounts_list:
                if account.get('username') == community_id:
                    return name
        return "N/A"

    @staticmethod
    def display_topics_table(
        network_nodes: Dict[str, Dict],
        topic_to_accounts: Dict[str, List[str]],
        account_to_topics: Dict[str, List[str]],
        cloutrank_scores: Dict[str, float]
    ) -> None:
        """
        Display a table of topics and the accounts that discuss them.
        """
        if not topic_to_accounts or not account_to_topics:
            logger.warning("No topic data available for displaying topic table")
            st.info("🔍 No topic data available. Use the 'Summarize Tweets & Generate Communities' button to generate topics.")
            return
        
        logger.info(f"Generating topic table with {len(topic_to_accounts)} topics across {len(account_to_topics)} accounts")
        
        try:
            TableVisualizer.apply_table_styles()
            st.header("Topics")
            
            with st.expander(f"🔍 Topics from Twitter Accounts ({len(topic_to_accounts)} topics found)", expanded=False):
                st.write("Topics extracted from tweet content and account descriptions, showing connections between accounts and topics they discuss.")
                
                username_to_node_id = {node.get("screen_name", "").lower(): node_id for node_id, node in network_nodes.items() if node.get("screen_name")}
                
                topic_influence = {}
                for topic, usernames in topic_to_accounts.items():
                    topic_score = sum(cloutrank_scores.get(username_to_node_id.get(username.lower()), 0) for username in usernames)
                    topic_influence[topic] = topic_score
                
                top_topics = sorted([(topic, score) for topic, score in topic_influence.items()], key=lambda x: x[1], reverse=True)[:20]
                
                logger.info(f"Selected top {len(top_topics)} topics by influence score")
                
                table_data = {
                    "Topic": [],
                    "Accounts": [],
                    "Key Statements": []
                }
                
                if not top_topics:
                    st.warning("No topics with influence score were found. This may happen if account usernames don't match between tables.")
                    return
                    
                for topic_name, _ in top_topics:
                    usernames = topic_to_accounts.get(topic_name, [])
                    accounts_text = ", ".join([f"@{u}" for u in usernames])
                    table_data["Topic"].append(topic_name)
                    table_data["Accounts"].append(accounts_text)
                    
                    key_statements = []
                    for username in usernames[:3]:
                        node_id = username_to_node_id.get(username.lower())
                        if node_id and node_id in network_nodes:
                            node = network_nodes[node_id]
                            tweet_summary = node.get("tweet_summary", "")
                            
                            if tweet_summary:
                                sentences = tweet_summary.split(". ")
                                relevant_sentences = [s for s in sentences if topic_name.lower() in s.lower()]
                                statement = ""
                                if relevant_sentences:
                                    statement = relevant_sentences[0]
                                else:
                                    statement = sentences[0] if sentences else ""
                                
                                if statement:
                                    key_statements.append(f"@{username}: {statement}")

                    statements_text = "\n\n".join(key_statements)
                    formatted_statements = TableVisualizer.format_text_with_line_breaks(statements_text, max_line_length=100)
                    table_data["Key Statements"].append(formatted_statements)
                
                df = pd.DataFrame(table_data)
                
                logger.info("Displaying topic table with data rows: " + str(len(df)))
                st.table(df)
                
        except Exception as e:
            error_msg = f"Error generating topic table: {str(e)}"
            logger.error(error_msg)
            st.error(error_msg)