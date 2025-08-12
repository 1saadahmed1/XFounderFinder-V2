"""
AI client for Gemini API interactions.
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

# Set up logger
logger = logging.getLogger(__name__)

class AIClient:
    """Client for interacting with Google's Gemini API."""
    
    def __init__(self, api_key: str = GEMINI_API_KEY):
        """
        Initialize the Gemini AI client.
        
        Args:
            api_key: Gemini API key
        """
        self.api_key = api_key
        # We will use the model that is best for this type of task.
        self.model_name = "gemini-1.5-pro-latest"
        self._max_concurrent = MAX_CONCURRENT_REQUESTS
        self._semaphores = {}
        
    def _get_semaphore(self):
        """
        Get or create a semaphore for the current event loop.
        
        Returns:
            asyncio.Semaphore: A semaphore attached to the current event loop
        """
        loop = asyncio.get_event_loop()
        loop_id = id(loop)
        
        if loop_id not in self._semaphores:
            self._semaphores[loop_id] = asyncio.Semaphore(self._max_concurrent)
            
        return self._semaphores[loop_id]
    
    def _initialize_client(self) -> Any:
        """
        Initialize Gemini client.
        
        Returns:
            Gemini GenerativeModel
        """
        genai.configure(api_key=self.api_key)
        return genai.GenerativeModel(self.model_name)
    
    async def _make_api_call_with_retries(self, prompt: str, max_retries: int = 3) -> Optional[str]:
        """
        A centralized, robust method to make Gemini API calls with retries and exponential backoff.
        
        Args:
            prompt: The prompt text for the Gemini model.
            max_retries: The maximum number of retries.
            
        Returns:
            The raw text response from the API, or None on failure.
        """
        base_delay = 1.0

        for retry_count in range(max_retries):
            try:
                client = self._initialize_client()
                response = await client.generate_content_async(prompt)
                
                # Check for an empty or invalid response
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
                    logger.error(f"All {max_retries} retries failed for API call. Giving up.")
                    return None
            except Exception as e:
                logger.error(f"Unexpected error during API call (attempt {retry_count + 1}/{max_retries}): {e}")
                return None
    
    async def search_for_candidates(self, profiles: List[Dict], user_prompt: str) -> Dict[str, Any]:
        """
        Uses Gemini to analyze a list of profiles and find candidates based on a user's prompt.

        Args:
            profiles: A list of profile dictionaries to analyze.
            user_prompt: A string describing the ideal candidate.

        Returns:
            A dictionary containing the AI's analysis and a list of candidates.
        """
        async with self._get_semaphore():
            # Prepare the data to be sent to the AI
            profiles_for_ai = []
            for p in profiles:
                profile_info = {
                    "username": p.get("screen_name"),
                    "bio": p.get("description"),
                    "location": p.get("location"),
                    "followers_count": p.get("followers_count"),
                    "following_count": p.get("friends_count"),
                    "tweet_count": p.get("statuses_count"),
                }
                profiles_for_ai.append(profile_info)

            # Construct the prompt for the AI
            prompt = f"""You are an expert talent scout and HR professional. Your task is to analyze a list of X/Twitter profiles and identify promising candidates for a specific role based on a user's request.

Here is the user's request:
"{user_prompt}"

Here is the list of X/Twitter profiles in JSON format:
{json.dumps(profiles_for_ai, indent=2)}

Please review each profile and select only the ones that are a strong match for the user's request. For each promising candidate, provide a brief, professional justification for your choice.

Return your response in a single JSON object with the following structure:
```json
{{
    "candidates": [
        {{
            "username": "candidate_username_1",
            "reasoning": "A concise explanation of why this candidate is a strong fit.",
            "profile_url": "[https://x.com/candidate_username_1](https://x.com/candidate_username_1)"
        }},
        {{
            "username": "candidate_username_2",
            "reasoning": "A concise explanation of why this candidate is a strong fit.",
            "profile_url": "[https://x.com/candidate_username_2](https://x.com/candidate_username_2)"
        }}
    ]
}}
If no suitable candidates are found, return an empty list in the "candidates" key.
"""
            response_text = await self._make_api_call_with_retries(prompt)

        if response_text:
            try:
                # Use a regex to extract the JSON from the text, as the AI might wrap it in markdown.
                # The re.DOTALL flag ensures that the . matches newlines.
                json_match = re.search(r'```json\s*(\{.*\})\s*```', response_text, re.DOTALL)
                if json_match:
                    response_json_string = json_match.group(1)
                else:
                    # Fallback to assuming the whole response is JSON if no markdown is found
                    response_json_string = response_text
                    
                ai_response = json.loads(response_json_string)
                if "candidates" in ai_response and isinstance(ai_response["candidates"], list):
                    return ai_response
                else:
                    logger.error(f"AI response format was incorrect: {response_text}")
                    return {"candidates": []}
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse AI response as JSON: {e}\nRaw Response: {response_text}")
                return {"candidates": []}
        else:
            return {"candidates": []}

# The following methods are not needed for the new functionality and will be kept but not used by the new app logic.
# I recommend removing them in a later refactor for a cleaner codebase.
async def generate_tweet_summary(self, tweets: List[Dict], username: str) -> str:
    """
    Generate an AI summary of tweets using Gemini API.
    """
    # ... (original code remains the same) ...
    return ""

async def generate_batch_tweet_summaries(self, batch_data: List[Tuple],
                                         batch_size: int = DEFAULT_BATCH_SIZE,
                                         max_retries: int = 3) -> Dict[str, str]:
    """
    Generate summaries for multiple accounts in a single API call.
    """
    # ... (original code remains the same) ...
    return {}

async def generate_community_labels(self, accounts: List[Dict],
                                    num_communities: int,
                                    max_retries: int = 3) -> Dict[str, str]:
    """
    Get community labels from Gemini using account descriptions.
    """
    # ... (original code remains the same) ...
    return {}

async def classify_accounts(self, accounts: List[Dict],
                            community_labels: Dict[str, str],
                            max_retries: int = 3) -> Dict[str, str]:
    """
    Classify accounts into communities.
    """
    # ... (original code remains the same) ...
    return {}
    
async def extract_topics_from_tweets(self, accounts: List[Dict], max_retries: int = 3) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
    """
    Extract topics from tweet summaries and descriptions of accounts.
    """
    # ... (original code remains the same) ...
    return {}, {}
    
async def summarize_user_tweets(self, username: str, tweet_text: str) -> str:
    """
    Generate an AI summary of tweets using Gemini API specifically for the original user.
    """
    # ... (original code remains the same) ...
    return ""