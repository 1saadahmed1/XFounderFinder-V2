"""
Community detection and management for X Network Visualization.
"""
import logging
import random
import asyncio
from typing import Dict, List, Tuple, Optional, Any
import colorsys

import streamlit as st

from api.ai_client import AIClient
from config import DEFAULT_NUM_COMMUNITIES, DEFAULT_BATCH_SIZE, MIN_ACCOUNTS_FOR_COMMUNITIES

logger = logging.getLogger(__name__)

class CommunityManager:
    """
    Manager for community detection and classification.
    """
    
    def __init__(self, ai_client: AIClient):
        """
        Initialize the community manager.
        
        Args:
            ai_client: AIClient instance for generating labels and classifications
        """
        self.ai_client = ai_client
        self.community_labels: Dict[str, str] = {}
        self.community_colors: Dict[str, str] = {}
        self.node_communities: Dict[str, str] = {}
    
    # New method to combine label generation and account classification
    async def detect_and_label_communities(self, node_descriptions: Dict[str, str], num_communities: int = DEFAULT_NUM_COMMUNITIES) -> Tuple[Dict[str, str], Dict[str, str]]:
        """
        Detects communities and labels them, then classifies all nodes into
        those communities. This method orchestrates the entire process.
        
        Args:
            node_descriptions: A dictionary of node IDs and their descriptions.
            num_communities: The desired number of communities.
            
        Returns:
            A tuple containing:
            - A dictionary mapping community IDs to their human-readable labels.
            - A dictionary mapping account usernames to their assigned community IDs.
        """
        # Convert the descriptions into a format the other methods can use.
        accounts = [{"screen_name": username, "description": desc} for username, desc in node_descriptions.items()]
        
        # 1. Generate human-readable labels for the communities
        community_labels = await self.generate_community_labels(
            accounts=accounts,
            num_communities=num_communities
        )
        
        # 2. Classify each account into one of the newly generated communities
        node_communities = await self.classify_accounts(
            accounts=accounts
        )
        
        return community_labels, node_communities
        
    def generate_n_colors(self, n: int) -> List[str]:
        """
        Generate n visually distinct colors.
        
        Args:
            n: Number of colors to generate
            
        Returns:
            List of hex color codes
        """
        colors = []
        for i in range(n):
            hue = i / n
            saturation = 0.7 + random.uniform(-0.2, 0.2)
            value = 0.9 + random.uniform(-0.2, 0.2)
            rgb = colorsys.hsv_to_rgb(hue, saturation, value)
            hex_color = '#{:02x}{:02x}{:02x}'.format(
                int(rgb[0] * 255),
                int(rgb[1] * 255),
                int(rgb[2] * 255))
            colors.append(hex_color)
        return colors
    
    def make_color_more_distinct(self, hex_color: str) -> str:
        """
        Make colors more distinct by increasing saturation and adjusting value.
        
        Args:
            hex_color: Hex color code
            
        Returns:
            Enhanced hex color code
        """
        # Convert hex to RGB
        hex_color = hex_color.lstrip('#')
        r, g, b = tuple(int(hex_color[i:i+2], 16)/255 for i in (0, 2, 4))
        
        # Convert RGB to HSV
        h, s, v = colorsys.rgb_to_hsv(r, g, b)
        
        # Increase saturation, ensure good value
        s = min(1.0, s * 1.3)  # Increase saturation by 30%
        v = max(0.6, min(0.95, v))  # Keep value in a good range
        
        # Convert back to RGB
        r, g, b = colorsys.hsv_to_rgb(h, s, v)
        
        # Convert to hex
        return '#{:02x}{:02x}{:02x}'.format(
            int(r * 255),
            int(g * 255),
            int(b * 255))
    
    def categorize_communities(self) -> Dict[str, str]:
        """
        Group communities into broader categories for better organization.
        
        Returns:
            Dictionary mapping community IDs to category names
        """
        if not self.community_labels:
            return {}
            
        # Define category keywords
        categories = {
            "Technology": ["ai", "software", "dev", "tech", "computer", "engineer", "automation", 
                          "robotics", "open source", "security", "privacy", "uiux", "design"],
            "Business": ["vc", "investor", "startup", "founder", "ceo", "business", "marketing", 
                         "growth", "sales", "careers", "jobs", "y combinator"],
            "Finance": ["financial", "trading", "crypto", "web3"],
            "Politics": ["political", "government", "regulation", "regulatory", "conflict", 
                         "ukraine", "russia", "israel", "palestine"],
            "Science": ["research", "academic", "neuroscience", "bci", "science", "stem", 
                       "space", "exploration", "health", "longevity", "theoretical"],
            "Creative": ["artist", "designer", "music", "arts", "food"],
            "Social": ["community", "support", "personal", "sports", "culture", "family", 
                       "e/acc", "reflection"],
            "Media": ["news", "journalism", "publisher", "book"],
            "Geographic": ["indian", "chinese", "irish"],
            "Other": ["other", "random"]
        }
        
        # Create mapping from community ID to category
        community_categories = {}
        
        for comm_id, label in self.community_labels.items():
            assigned = False
            label_lower = label.lower()
            
            # Try to find matching category
            for category, keywords in categories.items():
                if any(keyword in label_lower for keyword in keywords):
                    community_categories[comm_id] = category
                    assigned = True
                    break
            
            # If no category matches, use "Other"
            if not assigned:
                community_categories[comm_id] = "Other"
        
        return community_categories
    
    def chunk_accounts_for_processing(self, accounts: List[Dict], 
                                     max_chunk_size: int = DEFAULT_BATCH_SIZE) -> List[List[Dict]]:
        """
        Split accounts into manageable chunks for API processing.
        
        Args:
            accounts: List of account dictionaries
            max_chunk_size: Maximum chunk size
            
        Returns:
            List of account chunks
        """
        chunks = []
        current_chunk = []
        
        for account in accounts:
            current_chunk.append(account)
            if len(current_chunk) >= max_chunk_size:
                chunks.append(current_chunk)
                current_chunk = []
        
        # Add any remaining accounts
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks
    
    def merge_community_labels(self, label_groups: List[Dict[str, str]]) -> Dict[str, str]:
        """
        Merge multiple sets of community labels, resolving duplicates.
        
        Args:
            label_groups: List of community label dictionaries
            
        Returns:
            Merged community labels
        """
        if not label_groups:
            return {}
        
        # If only one group, just return it
        if len(label_groups) == 1:
            return label_groups[0]
        
        # Start with the first group
        merged_labels = label_groups[0].copy()
        
        # Track label frequencies to identify the most common themes
        label_counts = {}
        for labels in label_groups:
            for community, label in labels.items():
                if label not in label_counts:
                    label_counts[label] = 0
                label_counts[label] += 1
        
        # Sort labels by frequency (most common first)
        sorted_labels = sorted(label_counts.items(), key=lambda x: x[1], reverse=True)
        
        # Create a comprehensive set of labels that covers all communities
        next_community_id = max([int(c) for c in merged_labels.keys()]) + 1 if merged_labels else 0
        
        # Add all labels from subsequent groups
        for group_idx, labels in enumerate(label_groups[1:], 1):
            for community, label in labels.items():
                # If this exact label is already in merged_labels, skip
                if label in merged_labels.values():
                    continue
                    
                # Otherwise add as a new community
                merged_labels[str(next_community_id)] = label
                next_community_id += 1
        
        return merged_labels
    
    async def generate_community_labels(self, accounts: List[Dict], 
                                     num_communities: int = DEFAULT_NUM_COMMUNITIES) -> Dict[str, str]:
        """
        Generate community labels using both account descriptions and tweet summaries with parallel processing.
        
        Args:
            accounts: List of account dictionaries
            num_communities: Number of communities to generate
            
        Returns:
            Dictionary mapping community IDs to labels
        """
        # Validate inputs
        if len(accounts) < MIN_ACCOUNTS_FOR_COMMUNITIES:
            logger.warning(f"Not enough accounts ({len(accounts)}) to form communities")
            return {}
        
        # Prepare accounts with tweets
        accounts_with_info = []
        for account in accounts:
            tweet_summary = account.get("tweet_summary", "")
            # Only include accounts that have either a description or tweet summary
            if account.get("description") or tweet_summary:
                account_info = {
                    "screen_name": account.get("screen_name", ""),
                    "description": account.get("description", ""),
                    "tweet_summary": tweet_summary
                }
                accounts_with_info.append(account_info)
        
        if not accounts_with_info:
            logger.warning("No account information available for community label generation")
            return {}
        
        # Use progress indicators
        progress = st.progress(0)
        status_text = st.empty()
        status_text.text("Generating community labels...")
        
        # Check if we need to chunk (roughly estimate token count)
        estimated_tokens = sum(len(a["description"].split()) + len(a.get("tweet_summary", "").split()) 
                                for a in accounts_with_info)
        need_chunking = estimated_tokens > 20000
        
        if need_chunking:
            # Chunk accounts
            chunks = self.chunk_accounts_for_processing(accounts_with_info)
            
            # Process chunks in parallel
            async def process_chunk(chunk_idx, chunk):
                chunk_labels = await self.ai_client.generate_community_labels(chunk, num_communities)
                return chunk_idx, chunk_labels
            
            # Create tasks for all chunks
            chunk_tasks = [
                process_chunk(i, chunk) for i, chunk in enumerate(chunks)
            ]
            
            # Process chunks with semaphore to limit concurrency
            semaphore = asyncio.Semaphore(3)  # Limit to 3 concurrent AI requests
            
            async def process_chunk_with_semaphore(chunk_idx, chunk):
                async with semaphore:
                    return await process_chunk(chunk_idx, chunk)
                
            # Create tasks with semaphore
            chunk_tasks = [
                process_chunk_with_semaphore(i, chunk) for i, chunk in enumerate(chunks)
            ]
            
            # Wait for all tasks to complete
            label_groups = []
            completed = 0
            
            for completed_task in asyncio.as_completed(chunk_tasks):
                chunk_idx, chunk_labels = await completed_task
                if chunk_labels:
                    label_groups.append(chunk_labels)
                
                completed += 1
                progress.progress(completed / len(chunks))
                status_text.text(f"Processed chunk {completed}/{len(chunks)}")
            
            # Merge results
            self.community_labels = self.merge_community_labels(label_groups)
        else:
            # Process all accounts at once
            self.community_labels = await self.ai_client.generate_community_labels(
                accounts_with_info, 
                num_communities
            )
        
        # Generate colors for communities
        if self.community_labels:
            color_list = self.generate_n_colors(len(self.community_labels))
            self.community_colors = {
                community_id: self.make_color_more_distinct(color) 
                for community_id, color in zip(self.community_labels.keys(), color_list)
            }
            
        status_text.text("Community label generation complete!")
        progress.progress(1.0)
        
        return self.community_labels
    
    async def classify_accounts(self, accounts: List[Dict]) -> Dict[str, str]:
        """
        Classify accounts into communities with parallel processing.
        
        Args:
            accounts: List of account dictionaries
            
        Returns:
            Dictionary mapping usernames to community IDs
        """
        # Skip if no communities
        if not self.community_labels:
            logger.warning("No community labels available for classification")
            return {}
        
        # Show progress indicator
        progress_bar = st.progress(0)
        status_text = st.empty()
        status_text.text("Classifying accounts into communities...")
        
        # Filter accounts that need classification
        accounts_to_classify = [
            acc for acc in accounts
            if acc.get("screen_name") and (acc.get("description") or acc.get("tweet_summary", ""))
        ]
        
        if not accounts_to_classify:
            logger.warning("No accounts with content found for classification")
            return {}
        
        # Divide into chunks for batch processing
        chunks = self.chunk_accounts_for_processing(accounts_to_classify)
        
        # Process chunks in parallel
        results = {}
        
        # Define function to process a chunk
        async def process_chunk(chunk_idx, chunk):
            chunk_results = await self.ai_client.classify_accounts(chunk, self.community_labels)
            return chunk_idx, chunk_results
        
        # Create semaphore to limit concurrency
        semaphore = asyncio.Semaphore(3)  # Limit to 3 concurrent AI requests
        
        async def process_chunk_with_semaphore(chunk_idx, chunk):
            async with semaphore:
                return await process_chunk(chunk_idx, chunk)
        
        # Create tasks for all chunks
        chunk_tasks = [
            process_chunk_with_semaphore(i, chunk) for i, chunk in enumerate(chunks)
        ]
        
        # Process all chunks with progress tracking
        completed = 0
        
        for completed_task in asyncio.as_completed(chunk_tasks):
            chunk_idx, chunk_results = await completed_task
            
            # Merge results
            results.update(chunk_results)
            
            # Update progress
            completed += 1
            progress_bar.progress(completed / len(chunks))
            status_text.text(f"Classified {len(results)}/{len(accounts_to_classify)} accounts")
        
        # Store results
        self.node_communities = results
        
        # Complete progress
        progress_bar.progress(1.0)
        status_text.text(f"Classification complete! Assigned {len(results)} accounts to communities")
        
        return results
    
    def get_top_accounts_by_community(self, nodes: Dict[str, Dict], 
                                     importance_scores: Dict[str, float], 
                                     top_n: int = 20) -> Dict[str, List]:
        """
        Get top accounts for each community based on importance scores.
        
        Args:
            nodes: Dictionary of node data
            importance_scores: Dictionary mapping node IDs to importance scores
            top_n: Number of top accounts to include per community
            
        Returns:
            Dictionary mapping community IDs to lists of (node_id, node_data, score) tuples
        """
        logger.info(f"Getting top accounts by community. Node communities: {len(self.node_communities)}")
        
        if not self.node_communities:
            logger.warning("No node communities found")
            return {}
            
        # Group accounts by community
        community_accounts = {}
        node_count = 0
        
        for node_id, node in nodes.items():
            if node_id.startswith("orig_"):
                continue
                
            username = node["screen_name"]
            node_count += 1
            
            if username in self.node_communities:
                community = self.node_communities[username]
                if community not in community_accounts:
                    community_accounts[community] = []
                community_accounts[community].append((node_id, node, importance_scores.get(node_id, 0)))
        
        logger.info(f"Processed {node_count} nodes, found {len(community_accounts)} communities")
        
        # Sort accounts within each community by importance
        top_accounts = {}
        for community, accounts in community_accounts.items():
            sorted_accounts = sorted(accounts, key=lambda x: x[2], reverse=True)[:top_n]
            top_accounts[community] = sorted_accounts
            logger.info(f"Community {community}: {len(sorted_accounts)} accounts after sorting")
        
        return top_accounts
