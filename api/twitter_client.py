"""
Twitter API client for the X Network Visualization application.
"""
import json
import asyncio
import time
from datetime import datetime
import logging
from typing import Dict, List, Tuple, Optional, Any

import aiohttp
import streamlit as st

from config import RAPIDAPI_KEY, RAPIDAPI_HOST, MAX_CONCURRENT_REQUESTS, DEFAULT_FETCH_TIMEOUT
from data.network import NetworkData

logger = logging.getLogger(__name__)

class TwitterClient:
    """Client for interacting with Twitter API through RapidAPI."""

    def _masked_api_key(self):
        key = self.api_key
        if len(key) > 4:
            return f"{'*' * (len(key) - 4)}{key[-4:]}"
        return key

    def __init__(self, api_key: str = RAPIDAPI_KEY, api_host: str = RAPIDAPI_HOST):
        """
        Initialize the Twitter API client.

        Args:
            api_key: RapidAPI key
            api_host: RapidAPI host
        """
        self.api_key = api_key
        self.api_host = api_host
        self.headers = {
            "x-rapidapi-key": self.api_key,
            "x-rapidapi-host": self.api_host
        }
        self._max_concurrent = MAX_CONCURRENT_REQUESTS
        self._semaphores = {}

        logger.debug(f"Initialized TwitterClient with RapidAPI Key: {self._masked_api_key()}")
        logger.debug(f"Using API Host: {self.api_host}")

    def _get_semaphore(self):
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        loop_id = id(loop)

        if loop_id not in self._semaphores:
            self._semaphores[loop_id] = asyncio.Semaphore(self._max_concurrent)

        return self._semaphores[loop_id]

    async def get_following(self, screenname: str, session: aiohttp.ClientSession,
                            cursor: Optional[str] = None) -> Tuple[List[Dict], Optional[str]]:

        semaphore = self._get_semaphore()
        async with semaphore:
            endpoint = f"/FollowingLight?username={screenname}&count=20"
            if cursor and cursor != "-1":
                endpoint += f"&cursor={cursor}"

            url = f"https://{self.api_host}{endpoint}"
            logger.debug(f"get_following: Request URL: {url}")
            logger.debug(f"get_following: Using headers: {self.headers}")

            max_retries = 5
            for attempt in range(max_retries):
                try:
                    async with session.get(url, headers=self.headers) as response:
                        logger.debug(f"get_following: HTTP status: {response.status}")
                        if response.status == 200:
                            data = await response.text()
                            logger.debug(f"get_following: Response data length: {len(data)}")
                            return self._parse_following_response(data)

                        elif response.status == 429:
                            reset_timestamp = response.headers.get("x-ratelimit-requests-reset")
                            logger.warning(f"get_following: Rate limited (429). Headers: {response.headers}")
                            if reset_timestamp:
                                try:
                                    reset_time = int(reset_timestamp)
                                    current_time = int(time.time())
                                    wait_seconds = reset_time - current_time
                                    if wait_seconds > 0:
                                        logger.warning(f"Rate limited on get_following for {screenname}. Waiting {wait_seconds} seconds before retrying...")
                                        await asyncio.sleep(wait_seconds + 1)
                                        continue
                                except ValueError:
                                    logger.error("Invalid x-ratelimit-requests-reset header value.")
                            else:
                                logger.warning("Rate limited but no reset header found, waiting 60 seconds by default.")
                                await asyncio.sleep(60)
                                continue
                        else:
                            logger.error(f"Failed to get following for {screenname}: HTTP {response.status}")
                            return [], None
                except Exception as e:
                    logger.error(f"Error fetching following for {screenname}: {str(e)}")
                    return [], None

            logger.error(f"Exceeded max retries getting following for {screenname}.")
            return [], None

    async def get_following_network_async(self, screenname: str) -> Optional[NetworkData]:

        all_accounts = []
        next_cursor = None

        try:
            async with self.create_session() as session:
                logger.debug(f"Fetching user info for: {screenname}")
                user_info = await self.get_user_info(screenname, session)
                if not user_info:
                    logger.error(f"Failed to get info for original user {screenname}.")
                    return None

                network_data = NetworkData()
                network_data.nodes[user_info['user_id']] = user_info

                logger.debug(f"Fetching following accounts for user_id: {user_info['user_id']}")
                while True:
                    accounts, next_cursor = await self.get_following(screenname, session, next_cursor)
                    logger.debug(f"Fetched {len(accounts)} accounts, next_cursor={next_cursor}")
                    if not accounts:
                        break
                    all_accounts.extend(accounts)
                    if not next_cursor or next_cursor == "-1":
                        break
        except Exception as e:
            logger.error(f"An error occurred while fetching the full network for {screenname}: {e}")
            return None

        for acc in all_accounts:
            network_data.nodes[acc["user_id"]] = acc
            network_data.edges.append({
                "source": user_info['user_id'],
                "target": acc['user_id'],
                "connection_type": "1st Degree"
            })

        logger.debug(f"Total nodes collected: {len(network_data.nodes)}")
        logger.debug(f"Total edges collected: {len(network_data.edges)}")

        return network_data

    async def get_user_info(self, screenname: str, session: aiohttp.ClientSession) -> Optional[Dict[str, Any]]:

        semaphore = self._get_semaphore()
        async with semaphore:
            endpoint = f"/UserByScreenName?username={screenname}"
            url = f"https://{self.api_host}{endpoint}"

            logger.debug(f"get_user_info: Request URL: {url}")
            logger.debug(f"get_user_info: Using headers: {self.headers}")

            max_retries = 5
            for attempt in range(max_retries):
                try:
                    async with session.get(url, headers=self.headers) as response:
                        logger.debug(f"get_user_info: HTTP status: {response.status}")
                        if response.status == 200:
                            data_str = await response.text()
                            logger.debug(f"get_user_info: Response data length: {len(data_str)}")
                            data = json.loads(data_str)

                            user_result = data.get("data", {}).get("user_result", {}).get("result", {})
                            if not user_result:
                                logger.warning(f"No user info found for {screenname} in API response.")
                                return None

                            legacy = user_result.get("legacy", {})

                            return {
                                "user_id": user_result.get("rest_id"),
                                "screen_name": legacy.get("screen_name"),
                                "name": legacy.get("name"),
                                "followers_count": legacy.get("followers_count"),
                                "friends_count": legacy.get("friends_count"),
                                "description": legacy.get("description"),
                                "statuses_count": legacy.get("statuses_count"),
                                "created_at": legacy.get("created_at"),
                                "location": legacy.get("location"),
                                "profile_image_url": legacy.get("profile_image_url_https"),
                            }
                        elif response.status == 429:
                            reset_timestamp = response.headers.get("x-ratelimit-requests-reset")
                            logger.warning(f"get_user_info: Rate limited (429). Headers: {response.headers}")
                            if reset_timestamp:
                                try:
                                    reset_time = int(reset_timestamp)
                                    current_time = int(time.time())
                                    wait_seconds = reset_time - current_time
                                    if wait_seconds > 0:
                                        logger.warning(f"Rate limited on get_user_info for {screenname}. Waiting {wait_seconds} seconds before retrying...")
                                        await asyncio.sleep(wait_seconds + 1)
                                        continue
                                except ValueError:
                                    logger.error("Invalid x-ratelimit-requests-reset header value.")
                            else:
                                logger.warning("Rate limited but no reset header found, waiting 60 seconds by default.")
                                await asyncio.sleep(60)
                                continue
                        else:
                            logger.error(f"Failed to get user info for {screenname}: HTTP {response.status}")
                            return None
                except (aiohttp.ClientError, json.JSONDecodeError, Exception) as e:
                    logger.error(f"Error fetching user info for {screenname}: {str(e)}")
                    return None

            logger.error(f"Exceeded max retries getting user info for {screenname}.")
            return None

    async def get_user_tweets(self, user_id: str, session: aiohttp.ClientSession,
                             cursor: Optional[str] = None) -> Tuple[List[Dict], Optional[str]]:

        async with self._get_semaphore():
            endpoint = f"/UserTweets?user_id={user_id}"
            if cursor:
                endpoint += f"&cursor={cursor}"

            url = f"https://{self.api_host}{endpoint}"
            logger.debug(f"get_user_tweets: Request URL: {url}")
            logger.debug(f"get_user_tweets: Using headers: {self.headers}")

            max_retries = 5
            retry_delay = 1.0

            for retry in range(max_retries):
                try:
                    async with session.get(url, headers=self.headers) as response:
                        logger.debug(f"get_user_tweets: HTTP status: {response.status}")
                        if response.status == 200:
                            data = await response.text()
                            logger.debug(f"get_user_tweets: Response data length: {len(data)}")
                            return self._parse_tweet_data(data)

                        elif response.status == 429:
                            reset_timestamp = response.headers.get("x-ratelimit-requests-reset")
                            logger.warning(f"get_user_tweets: Rate limited (429). Headers: {response.headers}")
                            if reset_timestamp:
                                try:
                                    reset_time = int(reset_timestamp)
                                    current_time = int(time.time())
                                    wait_seconds = reset_time - current_time
                                    if wait_seconds > 0:
                                        logger.warning(f"Rate limited when getting tweets for {user_id}. Waiting {wait_seconds} seconds before retrying...")
                                        await asyncio.sleep(wait_seconds + 1)
                                        continue
                                except ValueError:
                                    logger.error("Invalid x-ratelimit-requests-reset header value.")
                            else:
                                logger.warning("Rate limited but no reset header found, waiting 60 seconds by default.")
                                await asyncio.sleep(60)
                                continue

                        else:
                            logger.error(f"Failed to get tweets for {user_id}: HTTP {response.status}")
                            if retry < max_retries - 1:
                                wait_time = retry_delay * (2 ** retry)
                                logger.warning(f"Retrying in {wait_time:.1f}s (attempt {retry+1}/{max_retries})")
                                await asyncio.sleep(wait_time)
                                continue
                            return [], None
                except Exception as e:
                    logger.error(f"Error fetching tweets for {user_id}: {str(e)}")
                    if retry < max_retries - 1:
                        wait_time = retry_delay * (2 ** retry)
                        logger.warning(f"Retrying after error in {wait_time:.1f}s (attempt {retry+1}/{max_retries})")
                        await asyncio.sleep(wait_time)
                    else:
                        return [], None

            return [], None

    def _parse_following_response(self, json_str: str) -> Tuple[List[Dict], Optional[str]]:
        try:
            data = json.loads(json_str)
            logger.debug(f"_parse_following_response: Loaded JSON data")

            accounts = []
            next_cursor = data.get("next_cursor_str")
            logger.debug(f"_parse_following_response: next_cursor: {next_cursor}")

            users = data.get("users", [])
            logger.debug(f"_parse_following_response: Number of users found: {len(users)}")

            for user in users:
                account = {
                    "user_id": user.get("id_str"),
                    "screen_name": user.get("screen_name", ""),
                    "name": user.get("name", ""),
                    "followers_count": user.get("followers_count", 0),
                    "friends_count": user.get("friends_count", 0),
                    "statuses_count": user.get("statuses_count", 0),
                    "media_count": user.get("media_count", 0),
                    "created_at": user.get("created_at", ""),
                    "location": user.get("location", ""),
                    "blue_verified": user.get("verified", False),
                    "verified": user.get("verified", False),
                    "website": user.get("url", ""),
                    "business_account": False,
                    "description": user.get("description", ""),
                }
                accounts.append(account)

            logger.debug(f"_parse_following_response: Parsed {len(accounts)} accounts")
            return accounts, next_cursor

        except json.JSONDecodeError as e:
            logger.error(f"Error parsing following response: {str(e)}")
            return [], None
        except Exception as e:
            logger.error(f"Unexpected error parsing following response: {str(e)}")
            return [], None

    def _parse_tweet_data(self, json_str: str) -> Tuple[List[Dict], Optional[str]]:
        try:
            if not json_str or len(json_str.strip()) == 0:
                logger.debug("_parse_tweet_data: Empty or whitespace-only response")
                return [], None

            tweet_data = json.loads(json_str)
            logger.debug("_parse_tweet_data: Loaded JSON data")

            tweets = []
            next_cursor = None

            if not isinstance(tweet_data, dict):
                logger.warning("_parse_tweet_data: Tweet data not a dict")
                return [], None

            if "errors" in tweet_data:
                error_msgs = [error.get("message", "Unknown error") for error in tweet_data.get("errors", [])]
                if error_msgs:
                    logger.error(f"API returned errors: {', '.join(error_msgs)}")
                return [], None

            if "data" not in tweet_data:
                logger.warning("_parse_tweet_data: No 'data' key in response")
                return [], None

            timeline = tweet_data.get("data", {}).get("user_result_by_rest_id", {}).get("result", {}).get("profile_timeline_v2", {}).get("timeline", {})

            if not timeline:
                user_result = tweet_data.get("data", {}).get("user_result_by_rest_id", {}).get("result", {})
                if user_result and user_result.get("__typename") == "UserUnavailable":
                    reason = user_result.get("reason", "Account unavailable")
                    logger.warning(f"User unavailable: {reason}")
                return [], None

            all_entries = []

            for instruction in timeline.get("instructions", []):
                if instruction.get("__typename") == "TimelinePinEntry":
                    entry = instruction.get("entry", {})
                    if entry:
                        all_entries.append(entry)
                elif instruction.get("__typename") == "TimelineAddEntries":
                    entries = instruction.get("entries", [])
                    for entry in entries:
                        if entry.get("content", {}).get("__typename") == "TimelineTimelineCursor":
                            cursor_type = entry.get("content", {}).get("cursor_type")
                            if cursor_type == "Bottom":
                                next_cursor = entry.get("content", {}).get("value")
                            continue
                        all_entries.append(entry)

            logger.debug(f"_parse_tweet_data: Found {len(all_entries)} timeline entries")

            for entry in all_entries:
                tweet_content = entry.get("content", {}).get("content", {}).get("tweet_results", {}).get("result", {})
                if not tweet_content:
                    continue

                is_retweet = False
                tweet_to_parse = tweet_content

                if "retweeted_status_results" in tweet_content.get("legacy", {}):
                    is_retweet = True
                    tweet_to_parse = tweet_content.get("legacy", {}).get("retweeted_status_results", {}).get("result", {})

                legacy = tweet_to_parse.get("legacy", {})
                if not legacy:
                    continue

                text = legacy.get("full_text", "")
                urls = legacy.get("entities", {}).get("urls", [])
                for url in urls:
                    text = text.replace(url.get("url", ""), url.get("display_url", ""))

                likes = legacy.get("favorite_count", 0)
                retweets = legacy.get("retweet_count", 0)
                replies = legacy.get("reply_count", 0)
                quotes = legacy.get("quote_count", 0)

                created_at = legacy.get("created_at", "")
                try:
                    date_obj = datetime.strptime(created_at, "%a %b %d %H:%M:%S %z %Y")
                    date_str = date_obj.strftime("%Y-%m-%d %H:%M")
                except:
                    date_str = created_at

                tweets.append({
                    "text": text,
                    "date": date_str,
                    "likes": likes,
                    "retweets": retweets,
                    "replies": replies,
                    "quotes": quotes,
                    "total_engagement": likes + retweets + replies + quotes,
                    "is_retweet": is_retweet
                })

            logger.debug(f"_parse_tweet_data: Parsed {len(tweets)} tweets")
            return tweets, next_cursor

        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON response from API: {str(e)}")
            return [], None
        except Exception as e:
            logger.error(f"Error parsing tweet data: {str(e)}")
            return [], None

    async def get_user_id_from_username(self, username: str, session: aiohttp.ClientSession) -> Optional[str]:

        semaphore = self._get_semaphore()
        async with semaphore:
            endpoint = f"/UsernameToUserId?username={username}"
            url = f"https://{self.api_host}{endpoint}"

            logger.debug(f"get_user_id_from_username: Request URL: {url}")
            logger.debug(f"get_user_id_from_username: Using headers: {self.headers}")

            try:
                async with session.get(url, headers=self.headers) as response:
                    logger.debug(f"get_user_id_from_username: HTTP status: {response.status}")
                    if response.status != 200:
                        logger.error(f"Error fetching user ID for {username}: HTTP {response.status}")
                        return None

                    json_str = await response.text()
                    logger.debug(f"get_user_id_from_username: Response data length: {len(json_str)}")
                    data = json.loads(json_str)

                    return data.get("id_str")
            except Exception as e:
                logger.error(f"Error fetching user ID for {username}: {str(e)}")
                return None

    @staticmethod
    def create_session(connector_limit: int = MAX_CONCURRENT_REQUESTS,
                       timeout: int = DEFAULT_FETCH_TIMEOUT) -> aiohttp.ClientSession:

        logger.debug(f"Creating aiohttp session with connector_limit={connector_limit} and timeout={timeout}")
        conn = aiohttp.TCPConnector(
            limit=connector_limit,
            force_close=False,
            ttl_dns_cache=600,
            use_dns_cache=True,
            ssl=False
        )
        timeout_obj = aiohttp.ClientTimeout(total=timeout)
        session = aiohttp.ClientSession(connector=conn, timeout=timeout_obj)
        logger.debug("aiohttp session created")
        return session
