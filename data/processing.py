"""
Data processing utilities for X Network Visualization - Complete Fixed Version.
"""
import logging
import asyncio
from typing import Dict, List, Tuple, Optional, Any, Set

import aiohttp
import pandas as pd
import streamlit as st

from api.twitter_client import TwitterClient
from api.ai_client import AIClient
from data.network import NetworkData
from config import DEFAULT_BATCH_SIZE

logger = logging.getLogger(__name__)

class DataProcessor:
    """
    Enhanced processor for network data collection with fixed None handling.
    """
    
    def __init__(self, twitter_client: TwitterClient, ai_client: AIClient):
        self.twitter_client = twitter_client
        self.ai_client = ai_client

    async def find_candidates_with_ai(self, network: NetworkData, user_prompt: str) -> Dict[str, Any]:
        """Enhanced AI candidate search with detailed analysis."""
        st.info("Analyzing candidates with AI...")
        
        profiles_to_analyze = []
        for user_id, profile in network.nodes.items():
            if user_id != network.original_id and "screen_name" in profile:
                follower_count = profile.get('followers_count') or 0
                tweet_count = profile.get('statuses_count') or 0
                
                enhanced_profile = profile.copy()
                enhanced_profile.update({
                    'activity_level': 'High' if tweet_count > 1000 else 'Moderate' if tweet_count > 100 else 'Low',
                    'engagement_tier': 'High' if follower_count > 10000 else 'Medium' if follower_count > 1000 else 'Basic'
                })
                profiles_to_analyze.append(enhanced_profile)
        
        if not profiles_to_analyze:
            return {"candidates": [], "analysis_summary": "No profiles available for analysis"}

        try:
            ai_response = await self.ai_client.search_for_candidates(profiles_to_analyze, user_prompt)
            return ai_response
        except Exception as e:
            logger.error(f"Error during AI candidate search: {e}")
            return {"candidates": [], "analysis_summary": f"Error: {e}"}

    async def collect_network_data(self, 
                                   username: str, 
                                   following_pages: int = 2, 
                                   second_degree_pages: int = 1,
                                   max_first_degree: int = 200,
                                   max_second_degree_per_account: int = 30) -> NetworkData:
        """Optimized network collection with smart limits."""
        network = NetworkData()
        original_id = network.add_original_node(username)
        
        progress = st.progress(0)
        status_text = st.empty()
        
        try:
            async with TwitterClient.create_session() as session:
                status_text.text("Fetching first-degree connections...")
                
                first_hop_accounts = await self._get_following_pages_limited(
                    username=username, 
                    session=session, 
                    max_pages=following_pages,
                    max_accounts=max_first_degree
                )
                
                first_degree_ids = []
                for account in first_hop_accounts:
                    uid = str(account.get("user_id", ""))
                    if not uid:
                        continue
                    
                    account["degree"] = 1
                    network.add_node(uid, account)
                    network.add_edge(original_id, uid)
                    first_degree_ids.append((uid, account.get("screen_name", "")))
                
                progress.progress(0.3)
                status_text.text(f"Found {len(first_degree_ids)} first-degree connections")
                
                if second_degree_pages > 0 and first_degree_ids:
                    status_text.text("Collecting second-degree connections in parallel...")
                    
                    semaphore = asyncio.Semaphore(8)
                    
                    async def fetch_second_degree_limited(source_id, source_name):
                        async with semaphore:
                            return await self._fetch_second_degree_limited(
                                source_id, source_name, session, 
                                second_degree_pages, max_second_degree_per_account
                            )
                    
                    tasks = [
                        fetch_second_degree_limited(source_id, source_name)
                        for source_id, source_name in first_degree_ids
                    ]
                    
                    completed = 0
                    for completed_task in asyncio.as_completed(tasks):
                        source_id, connections = await completed_task
                        
                        for sid, node_data in connections:
                            if sid not in network.nodes:
                                network.add_node(sid, node_data)
                            network.add_edge(source_id, sid)
                        
                        completed += 1
                        progress.progress(0.3 + (0.7 * completed / len(tasks)))
                        status_text.text(f"Processed {completed}/{len(tasks)} first-degree accounts")
                
                progress.progress(1.0)
                total_accounts = len(network.nodes)
                status_text.text(f"Collection complete! Found {total_accounts} total accounts")
                
        except Exception as e:
            logger.error(f"Error in network collection: {e}")
            status_text.text(f"Collection error: {e}")
        
        return network

    async def _get_following_pages_limited(self, 
                                           username: str, 
                                           session: aiohttp.ClientSession, 
                                           max_pages: int,
                                           max_accounts: int) -> List[Dict]:
        """Get following pages with strict account limits."""
        all_accounts = []
        cursor = None
        
        for page in range(max_pages):
            if len(all_accounts) >= max_accounts:
                break
                
            accounts, cursor = await self.twitter_client.get_following(username, session, cursor)
            
            if not accounts:
                break
                
            remaining_slots = max_accounts - len(all_accounts)
            all_accounts.extend(accounts[:remaining_slots])
            
            if not cursor or cursor == "-1":
                break
        
        return all_accounts

    async def _fetch_second_degree_limited(self, 
                                           source_id: str, 
                                           source_name: str, 
                                           session: aiohttp.ClientSession, 
                                           max_pages: int,
                                           max_accounts: int) -> Tuple[str, List[Tuple[str, Dict]]]:
        """Fetch limited second-degree connections for performance."""
        try:
            accounts = await self._get_following_pages_limited(
                username=source_name,
                session=session,
                max_pages=max_pages,
                max_accounts=max_accounts
            )

            connections = []
            for account in accounts:
                sid = str(account.get("user_id", ""))
                if sid:
                    account["degree"] = 2
                    connections.append((sid, account))
            
            return (source_id, connections)
            
        except Exception as e:
            logger.error(f"Error fetching second-degree for {source_name}: {e}")
            return source_id, []

    async def process_tweet_data(self, 
                                 network: NetworkData, 
                                 selected_nodes: Set[str], 
                                 batch_size: int = 20) -> NetworkData:
        """Ultra-fast tweet processing with intelligent prioritization."""
        priority_nodes = self._prioritize_nodes_for_tweet_processing(network, selected_nodes)
        
        if not priority_nodes:
            st.info("No priority accounts selected for tweet processing")
            return network
        
        max_tweet_accounts = 50
        if len(priority_nodes) > max_tweet_accounts:
            priority_nodes = priority_nodes[:max_tweet_accounts]
            st.warning(f"Limited to top {max_tweet_accounts} most important accounts for performance")
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        status_text.text(f"Processing tweets for {len(priority_nodes)} priority accounts...")
        
        try:
            async with TwitterClient.create_session() as session:
                status_text.text("Fetching tweets in parallel...")
                
                semaphore = asyncio.Semaphore(10)
                
                async def fetch_with_semaphore(node_id, node):
                    async with semaphore:
                        return await self._fetch_tweets_for_node(node_id, node, session)
                
                fetch_tasks = [
                    fetch_with_semaphore(node_id, network.nodes[node_id])
                    for node_id in priority_nodes
                ]
                
                fetch_results = await asyncio.gather(*fetch_tasks, return_exceptions=True)
                progress_bar.progress(0.6)
                
                accounts_with_tweets = []
                for i, result in enumerate(fetch_results):
                    if isinstance(result, Exception):
                        continue
                    
                    node_id, tweets, status = result
                    if tweets and len(tweets) > 0:
                        accounts_with_tweets.append((node_id, tweets))
                    else:
                        network.nodes[node_id]["tweet_fetch_status"] = status or "No tweets"
                
                status_text.text(f"Processing AI summaries for {len(accounts_with_tweets)} accounts...")
                
                if accounts_with_tweets:
                    await self._batch_process_tweet_summaries(
                        network, accounts_with_tweets, progress_bar, status_text, 0.6
                    )
                
                progress_bar.progress(1.0)
                status_text.text(f"Tweet processing complete! Processed {len(accounts_with_tweets)} accounts")
                
        except Exception as e:
            logger.error(f"Error in tweet processing: {e}")
            status_text.text(f"Processing error: {e}")
        
        return network

    def _prioritize_nodes_for_tweet_processing(self, network: NetworkData, selected_nodes: Set[str]) -> List[str]:
        """FIXED: Intelligent prioritization with None handling."""
        priority_accounts = []
        importance_scores = getattr(self, '_cached_importance_scores', {})
        
        for node_id in selected_nodes:
            if node_id not in network.nodes:
                continue
                
            node = network.nodes[node_id]
            
            if node.get("tweet_summary"):
                continue
            
            # FIXED: Safe comparison with None values
            tweet_count = node.get("statuses_count") or 0
            if tweet_count < 5:
                continue
            
            if not node.get("description", "").strip():
                continue
            
            priority_score = 0
            
            # FIXED: Safe follower count handling
            followers = node.get("followers_count") or 0
            if followers > 50000:
                priority_score += 5
            elif followers > 10000:
                priority_score += 4
            elif followers > 1000:
                priority_score += 3
            elif followers > 100:
                priority_score += 1
            
            # FIXED: Safe tweet count handling
            if tweet_count > 5000:
                priority_score += 3
            elif tweet_count > 1000:
                priority_score += 2
            elif tweet_count > 100:
                priority_score += 1
            
            # FIXED: Safe importance score handling
            if importance_scores:
                importance = importance_scores.get(node_id) or 0
                priority_score += importance * 15
            
            # FIXED: Safe verified check
            if node.get("verified"):
                priority_score += 2
            
            priority_accounts.append((node_id, priority_score))
        
        priority_accounts.sort(key=lambda x: x[1], reverse=True)
        return [node_id for node_id, _ in priority_accounts]

    async def _batch_process_tweet_summaries(self, 
                                           network: NetworkData,
                                           accounts_with_tweets: List[Tuple[str, List]], 
                                           progress_bar, 
                                           status_text,
                                           base_progress: float) -> None:
        """Process AI summaries in efficient batches."""
        batch_size = 8
        batches = [accounts_with_tweets[i:i+batch_size] 
                   for i in range(0, len(accounts_with_tweets), batch_size)]
        
        for i, batch in enumerate(batches):
            try:
                batch_prompt = self._create_batch_tweet_summary_prompt(network, batch)
                batch_response = await self.ai_client._make_api_call_with_retries(batch_prompt)
                
                if batch_response:
                    self._parse_batch_summary_response(network, batch, batch_response)
                
                progress = base_progress + (0.4 * (i + 1) / len(batches))
                progress_bar.progress(progress)
                status_text.text(f"AI processing batch {i + 1}/{len(batches)}")
                
            except Exception as e:
                logger.error(f"Error processing AI batch {i}: {e}")
                for node_id, _ in batch:
                    network.nodes[node_id]["tweet_fetch_status"] = f"AI processing failed: {e}"

    def _create_batch_tweet_summary_prompt(self, network: NetworkData, batch: List[Tuple[str, List]]) -> str:
        """Create optimized batch prompt for tweet summarization."""
        
        prompt = """Analyze these Twitter accounts and provide professional summaries focused on expertise and career relevance.

For each account, provide a 2-3 sentence summary covering:
1. Primary expertise/industry focus
2. Professional role and key interests  
3. Communication style and thought leadership level

Return response in this exact JSON format:
```json
{
  "summaries": {
    "username1": "Concise professional summary...",
    "username2": "Another professional summary..."
  }
}
```

Accounts to analyze:

"""
        
        for node_id, tweets in batch:
            username = network.nodes[node_id].get("screen_name", "unknown")
            
            # FIXED: Safe sorting with None handling
            top_tweets = sorted(tweets, key=lambda t: t.get("total_engagement") or 0, reverse=True)[:8]
            tweet_texts = [t.get("text", "")[:150] for t in top_tweets if t.get("text")]
            
            if tweet_texts:
                prompt += f"\n@{username} recent tweets:\n"
                for i, text in enumerate(tweet_texts[:5], 1):
                    prompt += f"{i}. {text}\n"
                prompt += "\n"
        
        return prompt

    def _parse_batch_summary_response(self, network: NetworkData, batch: List[Tuple[str, List]], response: str) -> None:
        """FIXED: Parse batch AI response with JSON cleaning."""
        try:
            import json
            import re
            
            json_match = re.search(r'```json\s*(\{.*\})\s*```', response, re.DOTALL)
            if json_match:
                json_string = json_match.group(1)
                
                # Fix common JSON issues
                json_string = re.sub(r',\s*}', '}', json_string)
                json_string = re.sub(r',\s*]', ']', json_string)
                
                response_data = json.loads(json_string)
                summaries = response_data.get("summaries", {})
                
                for node_id, tweets in batch:
                    username = network.nodes[node_id].get("screen_name", "")
                    
                    if username in summaries:
                        summary = summaries[username]
                        network.update_node_tweet_data(node_id, tweets, summary)
                        network.nodes[node_id]["tweet_fetch_status"] = ""
                    else:
                        fallback_summary = f"Active Twitter user with {len(tweets)} recent tweets in their field"
                        network.update_node_tweet_data(node_id, tweets, fallback_summary)
                        network.nodes[node_id]["tweet_fetch_status"] = ""
            else:
                for node_id, tweets in batch:
                    fallback_summary = f"Professional with {len(tweets)} recent tweets"
                    network.update_node_tweet_data(node_id, tweets, fallback_summary)
                    network.nodes[node_id]["tweet_fetch_status"] = ""
                    
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error: {e}")
            for node_id, tweets in batch:
                network.update_node_tweet_data(node_id, tweets, "Summary generation failed")
                network.nodes[node_id]["tweet_fetch_status"] = "JSON parsing failed"
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            for node_id, tweets in batch:
                network.update_node_tweet_data(node_id, tweets, "Summary generation failed")
                network.nodes[node_id]["tweet_fetch_status"] = "Processing failed"

    async def _fetch_tweets_for_node(self, 
                                     node_id: str, 
                                     node: Dict, 
                                     session: aiohttp.ClientSession) -> Tuple[str, List[Dict], str]:
        """Fetch tweets for a single node with error handling."""
        try:
            tweets, _ = await self.twitter_client.get_user_tweets(node_id, session)
            if tweets:
                return node_id, tweets, ""
            else:
                return node_id, [], "No tweets available"
        except Exception as e:
            logger.error(f"Error fetching tweets for {node.get('screen_name', 'unknown')}: {e}")
            return node_id, [], f"Error: {str(e)}"