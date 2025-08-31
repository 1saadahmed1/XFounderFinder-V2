"""
Enhanced AI client for Gemini API with structured candidate analysis.
"""
import json
import re
import asyncio
import logging
from typing import Dict, List, Any, Optional, Tuple

import google.generativeai as genai
import streamlit as st
from google.api_core.exceptions import GoogleAPIError, RetryError

from config import GEMINI_API_KEY, MAX_CONCURRENT_REQUESTS, DEFAULT_BATCH_SIZE

logger = logging.getLogger(__name__)

class AIClient:
    """Enhanced client for Google's Gemini API with structured analysis."""
    
    def __init__(self, api_key: str = GEMINI_API_KEY):
        self.api_key = api_key
        self.model_name = "gemini-1.5-pro-latest"
        self._max_concurrent = MAX_CONCURRENT_REQUESTS
        self._semaphores = {}
        
    def _get_semaphore(self):
        loop = asyncio.get_event_loop()
        loop_id = id(loop)
        
        if loop_id not in self._semaphores:
            self._semaphores[loop_id] = asyncio.Semaphore(self._max_concurrent)
            
        return self._semaphores[loop_id]
    
    def _initialize_client(self) -> Any:
        genai.configure(api_key=self.api_key)
        return genai.GenerativeModel(self.model_name)
    
    async def _make_api_call_with_retries(self, prompt: str, max_retries: int = 3) -> Optional[str]:
        base_delay = 1.0

        for retry_count in range(max_retries):
            try:
                client = self._initialize_client()
                response = await client.generate_content_async(prompt)
                
                if not response.text:
                    raise ValueError("API returned an empty response.")
                
                return response.text.strip()
                
            except (GoogleAPIError, ValueError, RetryError) as e:
                logger.error(f"API call failed (attempt {retry_count + 1}/{max_retries}): {e}")
                if retry_count < max_retries - 1:
                    wait_time = base_delay * (2 ** retry_count)
                    logger.info(f"Retrying in {wait_time:.1f}s...")
                    await asyncio.sleep(wait_time)
                else:
                    logger.error(f"All {max_retries} retries failed. Giving up.")
                    return None
            except Exception as e:
                logger.error(f"Unexpected error: {e}")
                return None
    
    async def search_for_candidates(self, profiles: List[Dict], user_prompt: str) -> Dict[str, Any]:
        """Enhanced candidate search with structured analysis and tweet evidence."""
        async with self._get_semaphore():
            profiles_for_ai = []
            total_profiles = len(profiles)
            
            for p in profiles:
                influence_raw = p.get("influence_score", 0)
                influence_percentage = influence_raw * 100 if influence_raw else 0
                
                # Extract tweet examples if available
                tweet_examples = []
                if hasattr(p, 'tweets') and p.get('tweets'):
                    tweets = p.get('tweets', [])
                    top_tweets = sorted(tweets, key=lambda t: t.get("total_engagement", 0), reverse=True)[:3]
                    tweet_examples = [t.get("text", "")[:180] for t in top_tweets if t.get("text")]
                elif p.get("tweet_summary"):
                    tweet_examples = [f"Summary: {p.get('tweet_summary')[:180]}"]
                
                profile_info = {
                    "username": p.get("screen_name", ""),
                    "name": p.get("name", ""),
                    "bio": p.get("description", ""),
                    "location": p.get("location", ""),
                    "followers_count": p.get("followers_count", 0),
                    "following_count": p.get("friends_count", 0),
                    "tweet_count": p.get("statuses_count", 0),
                    "verified": p.get("verified", False),
                    "influence_score": influence_raw,
                    "influence_percentage": f"{influence_percentage:.3f}%",
                    "estimated_influence_reach": int(influence_percentage * total_profiles / 100) if total_profiles > 0 else 0,
                    "mutual_connections": p.get("mutual_connections", 0),
                    "tweet_summary": p.get("tweet_summary", ""),
                    "recent_tweet_examples": tweet_examples,
                    "connection_path": p.get("connection_path", ""),
                    "activity_level": "High" if p.get("statuses_count", 0) > 1000 else "Moderate" if p.get("statuses_count", 0) > 100 else "Low"
                }
                profiles_for_ai.append(profile_info)

            # Enhanced prompt with systematic scoring
            prompt = f"""You are a senior executive recruiter analyzing Twitter profiles for: "{user_prompt}"

SCORING FRAMEWORK (Rate each candidate 1-100):
- Role Fit (40 points): Experience, skills, industry match from bio
- Influence & Network (25 points): Followers, network position, thought leadership
- Technical Evidence (25 points): Skills shown in tweets/bio content  
- Accessibility (10 points): Response likelihood, seniority level

INFLUENCE CONTEXT:
- Scores show % of THIS {total_profiles}-person network that follows each person
- 1.6% = ~{int(1.6 * total_profiles / 100)} followers in this sample
- Higher % = more recognition in this specific community

PROFILES WITH EVIDENCE:
{json.dumps(profiles_for_ai, indent=2)}

REQUIREMENTS:
1. Score each section systematically (role_fit, influence_network, technical_evidence, accessibility)
2. Quote specific tweet content as evidence when available
3. Reference exact bio details that match requirements
4. Rank candidates by total score (highest first)
5. Assign tiers: A (80-100), B (60-79), C (40-59), D (<40)
6. Include red flags and concerns

Return ONLY this JSON structure:
```json
{{
    "candidates": [
        {{
            "username": "candidate_username",
            "total_score": 87,
            "scores": {{
                "role_fit": 35,
                "influence_network": 22,
                "technical_evidence": 23,
                "accessibility": 7
            }},
            "rank": 1,
            "tier": "A",
            "reasoning": "Detailed 3-4 sentence analysis covering specific bio details, influence explanation (e.g., 'Their 1.6% influence means ~8 people in this network follow them'), and tweet evidence. Example: 'Recent tweet: \\"Just shipped our ML pipeline with 40% better performance\\" shows hands-on technical skills.'",
            "key_strengths": ["Specific strength with evidence", "Another strength with metrics", "Third strength"],
            "concerns": ["Specific concern with reasoning"],
            "outreach_approach": "LinkedIn message referencing their recent work on [specific topic from tweets/bio]"
        }}
    ],
    "analysis_summary": "Overview of candidate pool quality and key insights"
}}
```

Focus on SPECIFIC EVIDENCE from actual tweets and bio content, not generic assessments. Rank by total_score descending."""

            response_text = await self._make_api_call_with_retries(prompt)

        if response_text:
            try:
                json_match = re.search(r'```json\s*(\{.*\})\s*```', response_text, re.DOTALL)
                if json_match:
                    response_json_string = json_match.group(1)
                else:
                    response_json_string = response_text.strip()
                    
                ai_response = json.loads(response_json_string)
                
                if "candidates" in ai_response and isinstance(ai_response["candidates"], list):
                    # Sort by total_score and re-assign ranks
                    candidates = ai_response["candidates"]
                    candidates.sort(key=lambda x: x.get("total_score", 0), reverse=True)
                    
                    for i, candidate in enumerate(candidates, 1):
                        candidate["rank"] = i
                        score = candidate.get("total_score", 0)
                        if score >= 80:
                            candidate["tier"] = "A"
                        elif score >= 60:
                            candidate["tier"] = "B"
                        elif score >= 40:
                            candidate["tier"] = "C"
                        else:
                            candidate["tier"] = "D"
                    
                    ai_response["candidates"] = candidates
                    logger.info(f"Successfully parsed and ranked {len(candidates)} candidates")
                    return ai_response
                else:
                    logger.error("Invalid response format - missing candidates array")
                    return {"candidates": [], "analysis_summary": "Failed to parse candidates"}
                    
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON: {e}")
                return {"candidates": [], "analysis_summary": f"JSON parsing error: {e}"}
        else:
            logger.error("No response received from AI API")
            return {"candidates": [], "analysis_summary": "No AI response received"}

    async def generate_tweet_summary(self, tweets: List[Dict], username: str) -> str:
        """Generate an AI summary of tweets for profile analysis."""
        if not tweets:
            return ""
        
        async with self._get_semaphore():
            tweet_texts = []
            for tweet in tweets[:15]:
                text = tweet.get("text", "")
                engagement = tweet.get("total_engagement", 0)
                if text:
                    tweet_texts.append(f"[{engagement} eng] {text}")
            
            if not tweet_texts:
                return "No tweet content available"
            
            prompt = f"""Analyze tweets from @{username} for professional assessment:

Tweets:
{chr(10).join(tweet_texts)}

Provide 2-3 sentence summary covering:
1. Primary expertise/industry focus
2. Technical skills demonstrated  
3. Thought leadership level

Focus on career-relevant insights:"""

            response = await self._make_api_call_with_retries(prompt)
            return response if response else "Unable to generate summary"

    async def generate_community_labels(self, accounts: List[Dict], 
                                       num_communities: int) -> Dict[str, str]:
        """Generate community labels from account descriptions and tweets."""
        async with self._get_semaphore():
            account_summaries = []
            for acc in accounts[:150]:
                summary_parts = []
                if acc.get('description'):
                    summary_parts.append(acc['description'])
                if acc.get('tweet_summary'):
                    summary_parts.append(acc['tweet_summary'])
                
                if summary_parts:
                    combined_summary = f"@{acc.get('screen_name', '')}: {' | '.join(summary_parts)}"
                    account_summaries.append(combined_summary)
            
            prompt = f"""Analyze these accounts and create {num_communities} distinct professional communities.

Focus on:
- Industry/domain (AI/ML, SaaS, Climate Tech)
- Role type (Engineers, Founders, Product Managers)
- Specialization (Research, Growth, Infrastructure)

Account data:
{chr(10).join(account_summaries[:100])}

Return JSON with specific, professional labels:
```json
{{
    "0": "AI/ML Engineers & Researchers",
    "1": "B2B SaaS Founders & Executives", 
    "2": "Product & Growth Leaders",
    ...
}}
```

Make labels specific and mutually exclusive."""

            response = await self._make_api_call_with_retries(prompt)
            
            if response:
                try:
                    json_match = re.search(r'```json\s*(\{.*\})\s*```', response, re.DOTALL)
                    if json_match:
                        labels = json.loads(json_match.group(1))
                        return labels
                except Exception as e:
                    logger.error(f"Error parsing community labels: {e}")
            
            return {str(i): f"Professional Group {i+1}" for i in range(num_communities)}

    async def classify_accounts(self, accounts: List[Dict], 
                               community_labels: Dict[str, str]) -> Dict[str, str]:
        """Classify accounts into detected communities."""
        async with self._get_semaphore():
            classifications = {}
            
            batch_size = 40
            for i in range(0, len(accounts), batch_size):
                batch = accounts[i:i+batch_size]
                
                account_data = []
                for acc in batch:
                    account_info = {
                        "username": acc.get("screen_name", ""),
                        "bio": acc.get("description", ""),
                        "tweet_summary": acc.get("tweet_summary", "")
                    }
                    account_data.append(account_info)
                
                prompt = f"""Classify accounts into communities based on bios and activity.

Communities:
{json.dumps(community_labels, indent=2)}

Accounts:
{json.dumps(account_data, indent=2)}

Guidelines:
- Match job titles, companies, skills in bios
- Use tweet summaries for current focus
- Choose single best-fitting community

Return JSON mapping usernames to community IDs:
```json
{{
    "username1": "0",
    "username2": "1",
    ...
}}
```"""

                response = await self._make_api_call_with_retries(prompt)
                
                if response:
                    try:
                        json_match = re.search(r'```json\s*(\{.*\})\s*```', response, re.DOTALL)
                        if json_match:
                            batch_classifications = json.loads(json_match.group(1))
                            classifications.update(batch_classifications)
                    except Exception as e:
                        logger.error(f"Error parsing batch classifications: {e}")
                        continue
            
            return classifications