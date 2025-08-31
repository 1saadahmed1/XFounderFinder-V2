"""
Fixed community detection and management for X Network Visualization.
"""
import logging
import random
import asyncio
import colorsys
from typing import Dict, List, Tuple, Optional, Any

import streamlit as st

from api.ai_client import AIClient

logger = logging.getLogger(__name__)

class CommunityManager:
    """
    FIXED manager for community detection with proper filtering and limits.
    """
    
    def __init__(self, ai_client: AIClient):
        """Initialize the community manager."""
        self.ai_client = ai_client
        self.community_labels: Dict[str, str] = {}
        self.community_colors: Dict[str, str] = {}
        self.node_communities: Dict[str, str] = {}
    
    async def detect_communities_fixed(self, 
                                     node_descriptions: Dict[str, str], 
                                     target_communities: int = 8,
                                     min_community_size: int = 5) -> Tuple[Dict[str, str], Dict[str, str]]:
        """
        FIXED community detection that only returns communities with actual members.
        """
        if len(node_descriptions) < 20:
            st.warning("Need at least 20 accounts for community detection")
            return {}, {}
        
        progress = st.progress(0)
        status = st.empty()
        
        try:
            # Sample if too large
            if len(node_descriptions) > 300:
                status.text(f"Sampling 300 accounts from {len(node_descriptions)} for analysis...")
                sampled_items = random.sample(list(node_descriptions.items()), 300)
                sample_descriptions = dict(sampled_items)
            else:
                sample_descriptions = node_descriptions
                status.text(f"Analyzing {len(sample_descriptions)} accounts...")
            
            # Convert to format AI expects
            sample_accounts = [
                {
                    "screen_name": username,
                    "description": desc,
                    "tweet_summary": ""
                }
                for username, desc in sample_descriptions.items()
            ]
            
            progress.progress(0.3)
            status.text("Generating community labels...")
            
            # Generate labels
            community_labels = await self.ai_client.generate_community_labels(
                sample_accounts, target_communities
            )
            
            if not community_labels:
                st.error("Failed to generate community labels")
                return {}, {}
            
            progress.progress(0.6)
            status.text("Classifying all accounts...")
            
            # Classify ALL original accounts
            all_accounts = [
                {"screen_name": username, "description": desc}
                for username, desc in node_descriptions.items()
            ]
            
            # Process in chunks
            node_communities = {}
            chunk_size = 50
            chunks = [all_accounts[i:i+chunk_size] for i in range(0, len(all_accounts), chunk_size)]
            
            for i, chunk in enumerate(chunks):
                chunk_classifications = await self.ai_client.classify_accounts(chunk, community_labels)
                node_communities.update(chunk_classifications)
                
                progress.progress(0.6 + (0.3 * (i + 1) / len(chunks)))
                status.text(f"Classified chunk {i + 1}/{len(chunks)}")
            
            progress.progress(0.9)
            status.text("Filtering communities...")
            
            # CRITICAL FIX: Count actual community sizes and filter
            community_counts = {}
            for username, comm_id in node_communities.items():
                if comm_id in community_labels:  # Only count valid community IDs
                    community_counts[comm_id] = community_counts.get(comm_id, 0) + 1
            
            # Filter out communities that are too small or have zero members
            filtered_labels = {}
            filtered_communities = {}
            
            for comm_id, label in community_labels.items():
                actual_count = community_counts.get(comm_id, 0)
                if actual_count >= min_community_size:
                    filtered_labels[comm_id] = label
                    # Only keep classifications for communities we're keeping
                    for username, user_comm_id in node_communities.items():
                        if user_comm_id == comm_id:
                            filtered_communities[username] = comm_id
            
            # Store results
            self.community_labels = filtered_labels
            self.node_communities = filtered_communities
            
            # Generate colors only for communities we kept
            if filtered_labels:
                self.community_colors = self._generate_distinct_colors()
            
            progress.progress(1.0)
            
            # Show final results
            final_count = len(filtered_labels)
            total_classified = len(filtered_communities)
            
            if final_count > 0:
                status.text(f"Created {final_count} communities with {total_classified} total members")
                st.success(f"Community detection complete: {final_count} communities, {total_classified} accounts classified")
            else:
                status.text("No communities met minimum size requirements")
                st.warning("No communities created - try lowering minimum community size")
            
            return filtered_labels, filtered_communities
            
        except Exception as e:
            logger.error(f"Error in community detection: {e}")
            st.error(f"Community detection failed: {e}")
            return {}, {}
    
    def _generate_distinct_colors(self) -> Dict[str, str]:
        """Generate visually distinct colors for the communities we actually have."""
        colors = {}
        n_colors = len(self.community_labels)
        community_ids = list(self.community_labels.keys())
        
        for i, comm_id in enumerate(community_ids):
            hue = i / n_colors
            saturation = 0.7 + random.uniform(-0.1, 0.1)
            value = 0.8 + random.uniform(-0.1, 0.1)
            
            rgb = colorsys.hsv_to_rgb(hue, saturation, value)
            hex_color = '#{:02x}{:02x}{:02x}'.format(
                int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255)
            )
            colors[comm_id] = hex_color
            
        return colors
    
    def get_top_accounts_by_community(self, 
                                     nodes: Dict[str, Dict], 
                                     importance_scores: Dict[str, float], 
                                     top_n: int = 15) -> Dict[str, List]:
        """Get top accounts for each community that actually has members."""
        if not self.node_communities:
            return {}
            
        community_accounts = {}
        
        for node_id, node in nodes.items():
            if node_id.startswith("orig_"):
                continue
                
            username = node.get("screen_name", "")
            if not username:
                continue
                
            # Only process if this username is in our filtered communities
            if username in self.node_communities:
                community = self.node_communities[username]
                
                # Double-check the community still exists in our labels
                if community in self.community_labels:
                    if community not in community_accounts:
                        community_accounts[community] = []
                    
                    account_data = {
                        'username': node_id,
                        'screen_name': username,
                        'importance_score': importance_scores.get(node_id, 0),
                        'tweet_summary': node.get('tweet_summary', ''),
                        'description': node.get('description', ''),
                        'followers_count': node.get('followers_count', 0),
                        'friends_count': node.get('friends_count', 0)
                    }
                    
                    community_accounts[community].append(account_data)
        
        # Sort accounts within each community by importance
        top_accounts = {}
        for community, accounts in community_accounts.items():
            sorted_accounts = sorted(accounts, key=lambda x: x['importance_score'], reverse=True)[:top_n]
            top_accounts[community] = sorted_accounts
        
        return top_accounts

class FixedCommunityUI:
    """
    FIXED community filtering interface that only shows communities with members.
    """
    
    @staticmethod
    def display_fixed_community_filters(community_labels: Dict[str, str], 
                                       node_communities: Dict[str, str],
                                       key_prefix: str = "community") -> Dict[str, bool]:
        """
        Display FIXED community filters - only shows communities with actual members.
        """
        if not community_labels:
            st.info("No communities available. Run community detection first.")
            return {}
        
        # Count ACTUAL members per community
        community_counts = {}
        for username, comm_id in node_communities.items():
            if comm_id in community_labels:  # Only count if community label exists
                community_counts[comm_id] = community_counts.get(comm_id, 0) + 1
        
        # ONLY show communities that have members
        communities_with_members = [
            (comm_id, label, community_counts.get(comm_id, 0)) 
            for comm_id, label in community_labels.items()
            if community_counts.get(comm_id, 0) > 0  # CRITICAL FIX
        ]
        
        if not communities_with_members:
            st.warning("No communities have members assigned. Try re-running community detection.")
            return {}
        
        # Sort by member count (largest first)
        communities_with_members.sort(key=lambda x: x[2], reverse=True)
        
        st.subheader(f"Community Filters ({len(communities_with_members)} communities)")
        
        # Quick selection buttons
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("Select All", key=f"{key_prefix}_select_all"):
                for comm_id, _, _ in communities_with_members:
                    st.session_state[f"{key_prefix}_filter_{comm_id}"] = True
                st.rerun()
        
        with col2:
            if st.button("Select None", key=f"{key_prefix}_select_none"):
                for comm_id, _, _ in communities_with_members:
                    st.session_state[f"{key_prefix}_filter_{comm_id}"] = False
                st.rerun()
        
        with col3:
            if st.button("Top 5 Only", key=f"{key_prefix}_top_5"):
                top_5_ids = [comm_id for comm_id, _, _ in communities_with_members[:5]]
                for comm_id, _, _ in communities_with_members:
                    st.session_state[f"{key_prefix}_filter_{comm_id}"] = comm_id in top_5_ids
                st.rerun()
        
        # Display communities in a compact way
        selected_communities = {}
        
        if len(communities_with_members) <= 12:
            # Show all communities if reasonable number
            n_cols = 2 if len(communities_with_members) > 6 else 1
            cols = st.columns(n_cols)
            
            for i, (comm_id, label, count) in enumerate(communities_with_members):
                col_idx = i % n_cols
                with cols[col_idx]:
                    default_value = count >= 10  # Auto-select larger communities
                    
                    selected = st.checkbox(
                        f"{label} ({count})",
                        value=st.session_state.get(f"{key_prefix}_filter_{comm_id}", default_value),
                        key=f"{key_prefix}_filter_{comm_id}",
                        help=f"Community with {count} members"
                    )
                    selected_communities[comm_id] = selected
        else:
            # Use selectbox for many communities
            st.warning(f"Too many communities ({len(communities_with_members)}). Showing top 12.")
            communities_with_members = communities_with_members[:12]
            
            for comm_id, label, count in communities_with_members:
                default_value = count >= 10
                selected = st.checkbox(
                    f"{label} ({count})",
                    value=st.session_state.get(f"{key_prefix}_filter_{comm_id}", default_value),
                    key=f"{key_prefix}_filter_{comm_id}"
                )
                selected_communities[comm_id] = selected
        
        # Show summary
        selected_count = sum(1 for selected in selected_communities.values() if selected)
        total_members = sum(
            community_counts.get(comm_id, 0) 
            for comm_id, selected in selected_communities.items() 
            if selected
        )
        
        if selected_count > 0:
            st.success(f"Selected {selected_count} communities with {total_members} total members")
        else:
            st.warning("No communities selected!")
        
        return selected_communities

def get_optimized_community_settings():
    """Provide optimized community detection settings."""
    
    st.sidebar.subheader("Community Detection Settings")
    
    # Smart defaults
    if 'network_data' in st.session_state and st.session_state.network_data:
        network_size = len(st.session_state.network_data.nodes)
        
        if network_size < 100:
            default_communities = 4
            st.sidebar.info("Small network: 4 communities recommended")
        elif network_size < 500:
            default_communities = 6
            st.sidebar.info("Medium network: 6 communities recommended")
        else:
            default_communities = 8
            st.sidebar.info("Large network: 8 communities recommended")
    else:
        default_communities = 6
    
    num_communities = st.sidebar.slider(
        "Target communities",
        min_value=3,
        max_value=12,
        value=default_communities,
        help="More = specific groups, fewer = broader categories"
    )
    
    min_community_size = st.sidebar.slider(
        "Minimum community size",
        min_value=3,
        max_value=25,
        value=8,
        help="Communities smaller than this will be filtered out"
    )
    
    return num_communities, min_community_size