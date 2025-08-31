"""
Enhanced table visualization with structured candidate results, color coding, and systematic ranking.
"""
import logging
from typing import Dict, List, Tuple, Optional, Any

import streamlit as st
import pandas as pd

logger = logging.getLogger(__name__)

class TableVisualizer:
    """
    Enhanced visualizer with structured candidate analysis display.
    """
    
    @staticmethod
    def format_text_with_line_breaks(text: str, max_line_length: int = 80) -> str:
        """Format text with line breaks for better readability."""
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
    def apply_enhanced_table_styles():
        """Apply enhanced CSS styling with color coding for tiers."""
        table_styles = """
        <style>
            .candidate-results {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            }
            .tier-a { 
                background: linear-gradient(135deg, #4CAF50, #45a049) !important; 
                color: white !important; 
                font-weight: bold !important;
            }
            .tier-b { 
                background: linear-gradient(135deg, #2196F3, #1976D2) !important; 
                color: white !important; 
                font-weight: bold !important;
            }
            .tier-c { 
                background: linear-gradient(135deg, #FF9800, #F57C00) !important; 
                color: white !important; 
                font-weight: bold !important;
            }
            .tier-d { 
                background: linear-gradient(135deg, #f44336, #d32f2f) !important; 
                color: white !important; 
                font-weight: bold !important;
            }
            .candidate-header {
                font-size: 18px;
                font-weight: bold;
                margin-bottom: 15px;
                padding: 10px;
                border-radius: 8px;
                text-align: center;
            }
            .score-breakdown {
                display: flex;
                justify-content: space-between;
                margin: 10px 0;
                padding: 10px;
                background-color: rgba(255, 255, 255, 0.1);
                border-radius: 5px;
            }
            .score-item {
                text-align: center;
                flex: 1;
                margin: 0 5px;
            }
            .score-label {
                font-size: 12px;
                color: #666;
                margin-bottom: 3px;
            }
            .score-value {
                font-size: 16px;
                font-weight: bold;
            }
            .evidence-section {
                margin: 10px 0;
                padding: 10px;
                background-color: rgba(0, 0, 0, 0.05);
                border-radius: 5px;
                border-left: 4px solid #2196F3;
            }
            .strength-item {
                margin: 5px 0;
                padding: 5px 10px;
                background-color: rgba(76, 175, 80, 0.1);
                border-radius: 3px;
                border-left: 3px solid #4CAF50;
            }
            .concern-item {
                margin: 5px 0;
                padding: 5px 10px;
                background-color: rgba(255, 152, 0, 0.1);
                border-radius: 3px;
                border-left: 3px solid #FF9800;
            }
            .outreach-strategy {
                margin: 10px 0;
                padding: 10px;
                background-color: rgba(33, 150, 243, 0.1);
                border-radius: 5px;
                border: 1px solid #2196F3;
            }
            .tweet-quote {
                font-style: italic;
                padding: 8px;
                margin: 5px 0;
                background-color: rgba(0, 0, 0, 0.05);
                border-left: 3px solid #9C27B0;
                border-radius: 3px;
            }
            .rank-badge {
                display: inline-block;
                padding: 4px 8px;
                border-radius: 15px;
                font-size: 14px;
                font-weight: bold;
                color: white;
                margin-right: 10px;
            }
            .rank-1 { background: #FFD700; color: #000; }
            .rank-2 { background: #C0C0C0; color: #000; }
            .rank-3 { background: #CD7F32; color: #fff; }
            .rank-other { background: #666; color: #fff; }
        </style>
        """
        st.markdown(table_styles, unsafe_allow_html=True)
    
    @staticmethod
    def display_structured_candidate_results(ai_response: Dict[str, Any]) -> None:
        """
        Display structured, color-coded candidate results with systematic analysis.
        """
        if not ai_response or not ai_response.get("candidates"):
            st.info("No candidates found matching your criteria.")
            return
        
        candidates = ai_response["candidates"]
        analysis_summary = ai_response.get("analysis_summary", "")
        methodology = ai_response.get("methodology", "")
        
        # Apply enhanced styling
        TableVisualizer.apply_enhanced_table_styles()
        
        # Display header with summary
        st.markdown('<div class="candidate-results">', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            st.subheader("🎯 AI Candidate Analysis Results")
        with col2:
            st.metric("Candidates Found", len(candidates))
        with col3:
            if candidates:
                avg_score = sum(c.get("total_score", 0) for c in candidates) / len(candidates)
                st.metric("Average Score", f"{avg_score:.1f}/100")
        
        if analysis_summary:
            st.info(f"**Analysis Summary:** {analysis_summary}")
        
        if methodology:
            with st.expander("📋 Evaluation Methodology", expanded=False):
                st.write(methodology)
        
        # Display candidates in structured format
        st.markdown("### 🏆 Ranked Candidate Analysis")
        
        # Create tier summaries
        tier_counts = {"A": 0, "B": 0, "C": 0, "D": 0}
        for candidate in candidates:
            tier = candidate.get("tier", "D")
            tier_counts[tier] += 1
        
        # Show tier distribution
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
            TableVisualizer._display_single_candidate(candidate)
            st.markdown("---")
        
        # Create downloadable summary
        TableVisualizer._create_candidate_download(candidates)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    @staticmethod
    def _display_single_candidate(candidate: Dict[str, Any]) -> None:
        """Display a single candidate with structured analysis."""
        username = candidate.get("username", "Unknown")
        total_score = candidate.get("total_score", 0)
        rank = candidate.get("rank", 0)
        tier = candidate.get("tier", "D")
        scores = candidate.get("scores", {})
        analysis = candidate.get("analysis", {})
        strengths = candidate.get("strengths", [])
        concerns = candidate.get("concerns", [])
        outreach = candidate.get("outreach_strategy", {})
        
        # Tier colors
        tier_class = f"tier-{tier.lower()}"
        
        # Rank badge
        if rank == 1:
            rank_class = "rank-1"
        elif rank == 2:
            rank_class = "rank-2"
        elif rank == 3:
            rank_class = "rank-3"
        else:
            rank_class = "rank-other"
        
        # Header with rank and tier
        st.markdown(f'''
        <div class="candidate-header {tier_class}">
            <span class="rank-badge {rank_class}">#{rank}</span>
            @{username} - Tier {tier} - Score: {total_score}/100
        </div>
        ''', unsafe_allow_html=True)
        
        # Score breakdown
        if scores:
            st.markdown(f'''
            <div class="score-breakdown">
                <div class="score-item">
                    <div class="score-label">Role Fit</div>
                    <div class="score-value">{scores.get("role_fit", 0)}/40</div>
                </div>
                <div class="score-item">
                    <div class="score-label">Influence</div>
                    <div class="score-value">{scores.get("influence_network", 0)}/25</div>
                </div>
                <div class="score-item">
                    <div class="score-label">Technical</div>
                    <div class="score-value">{scores.get("technical_evidence", 0)}/25</div>
                </div>
                <div class="score-item">
                    <div class="score-label">Access</div>
                    <div class="score-value">{scores.get("accessibility", 0)}/10</div>
                </div>
            </div>
            ''', unsafe_allow_html=True)
        
        # Analysis sections
        col1, col2 = st.columns([3, 2])
        
        with col1:
            # Role fit evidence
            if analysis.get("role_fit_evidence"):
                st.markdown(f'''
                <div class="evidence-section">
                    <strong>🎯 Role Fit Analysis:</strong><br>
                    {analysis["role_fit_evidence"]}
                </div>
                ''', unsafe_allow_html=True)
            
            # Technical skills
            if analysis.get("technical_skills"):
                st.markdown(f'''
                <div class="evidence-section">
                    <strong>⚡ Technical Evidence:</strong><br>
                    {analysis["technical_skills"]}
                </div>
                ''', unsafe_allow_html=True)
            
            # Recent activity with tweet quotes
            if analysis.get("recent_activity"):
                st.markdown(f'''
                <div class="tweet-quote">
                    <strong>📱 Recent Activity:</strong><br>
                    {analysis["recent_activity"]}
                </div>
                ''', unsafe_allow_html=True)
        
        with col2:
            # Influence explanation
            if analysis.get("influence_explanation"):
                st.markdown(f'''
                <div class="evidence-section">
                    <strong>📊 Network Influence:</strong><br>
                    {analysis["influence_explanation"]}
                </div>
                ''', unsafe_allow_html=True)
            
            # Strengths
            if strengths:
                st.markdown("<strong>✅ Key Strengths:</strong>")
                for strength in strengths:
                    st.markdown(f'<div class="strength-item">• {strength}</div>', unsafe_allow_html=True)
            
            # Concerns
            if concerns:
                st.markdown("<strong>⚠️ Concerns:</strong>")
                for concern in concerns:
                    st.markdown(f'<div class="concern-item">• {concern}</div>', unsafe_allow_html=True)
        
        # Outreach strategy
        if outreach and any(outreach.values()):
            st.markdown(f'''
            <div class="outreach-strategy">
                <strong>📞 Outreach Strategy:</strong><br>
                <strong>Approach:</strong> {outreach.get("approach", "Not specified")}<br>
                <strong>Conversation Starter:</strong> {outreach.get("conversation_starter", "Not specified")}<br>
                <strong>Best Timing:</strong> {outreach.get("best_timing", "Not specified")}
            </div>
            ''', unsafe_allow_html=True)
    
    @staticmethod
    def _create_candidate_download(candidates: List[Dict[str, Any]]) -> None:
        """Create downloadable CSV of candidate analysis."""
        download_data = []
        for candidate in candidates:
            scores = candidate.get("scores", {})
            analysis = candidate.get("analysis", {})
            outreach = candidate.get("outreach_strategy", {})
            
            row = {
                "Rank": candidate.get("rank", 0),
                "Username": candidate.get("username", ""),
                "Total Score": candidate.get("total_score", 0),
                "Tier": candidate.get("tier", ""),
                "Role Fit Score": scores.get("role_fit", 0),
                "Influence Score": scores.get("influence_network", 0),
                "Technical Score": scores.get("technical_evidence", 0),
                "Accessibility Score": scores.get("accessibility", 0),
                "Role Fit Evidence": analysis.get("role_fit_evidence", ""),
                "Technical Skills": analysis.get("technical_skills", ""),
                "Influence Explanation": analysis.get("influence_explanation", ""),
                "Recent Activity": analysis.get("recent_activity", ""),
                "Key Strengths": "; ".join(candidate.get("strengths", [])),
                "Concerns": "; ".join(candidate.get("concerns", [])),
                "Outreach Approach": outreach.get("approach", ""),
                "Conversation Starter": outreach.get("conversation_starter", ""),
                "Best Timing": outreach.get("best_timing", "")
            }
            download_data.append(row)
        
        if download_data:
            df = pd.DataFrame(download_data)
            csv_data = df.to_csv(index=False)
            
            st.download_button(
                label="📄 Download Detailed Analysis (CSV)",
                data=csv_data,
                file_name="candidate_analysis_detailed.csv",
                mime="text/csv",
                help="Download complete candidate analysis with scores and evidence"
            )

    @staticmethod
    def display_top_accounts_table(
        nodes: Dict[str, Dict],
        importance_scores: Dict[str, float],
        cloutrank_scores: Dict[str, float],
        in_degrees: Dict[str, int],
        top_accounts: List[Dict[str, Any]],
        original_id: str,
        show_tweet_summaries: bool = False,
        importance_metric: str = "CloutRank",
        max_display: int = 50
    ) -> None:
        """Display table of top accounts with enhanced formatting."""
        st.subheader(f"Top Accounts by {importance_metric}")
        
        display_accounts = top_accounts[:max_display]
        if len(top_accounts) > max_display:
            st.info(f"Displaying top {max_display} of {len(top_accounts)} accounts for performance")
        
        rows_list = []
        normalized_original_id = original_id.replace('@', '').lower() if original_id else ""

        for idx, account_data in enumerate(display_accounts, 1):
            node_id = account_data.get('username')
            
            if not node_id:
                continue

            node_details = nodes.get(node_id, {})
            if not node_details:
                continue

            if node_id.lower() == normalized_original_id:
                connection = "Original"
            else:
                connection = account_data.get('connection_type', 'Other')

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
                "Description": TableVisualizer.format_text_with_line_breaks(
                    node_details.get('description', '')[:200]
                ),
            }
            
            if 'community_name' in account_data:
                 row['Community'] = account_data['community_name']

            if show_tweet_summaries:
                tweet_summary = account_data.get("tweet_summary", "No tweet summary available")
                row["Tweet Summary"] = TableVisualizer.format_text_with_line_breaks(
                    tweet_summary[:300]
                )
            
            rows_list.append(row)
        
        if rows_list:
            df = pd.DataFrame(rows_list)
            with st.expander(f"Top {len(rows_list)} Accounts by {importance_metric}", expanded=True):
                st.dataframe(df, use_container_width=True, height=400)
        else:
            st.info("No accounts to display.")

    @staticmethod
    def display_network_summary(network_data, importance_scores: Dict[str, float]) -> None:
        """Display a summary of the network statistics."""
        if not network_data or not network_data.nodes:
            return
        
        st.subheader("Network Summary")
        
        total_nodes = len(network_data.nodes)
        first_degree = len(network_data.get_first_degree_nodes())
        second_degree = len(network_data.get_second_degree_nodes())
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Accounts", total_nodes)
        
        with col2:
            st.metric("First Degree", first_degree)
        
        with col3:
            st.metric("Second Degree", second_degree)
        
        with col4:
            avg_importance = sum(importance_scores.values()) / len(importance_scores) if importance_scores else 0
            st.metric("Avg Importance", f"{avg_importance:.3f}")
        
        nodes_with_tweets = sum(1 for node in network_data.nodes.values() 
                               if node.get('tweets') and len(node['tweets']) > 0)
        
        if nodes_with_tweets > 0:
            st.info(f"Tweet data available for {nodes_with_tweets} accounts")